# ember design

Living document. Every non-obvious decision gets a short entry: **what**, **alternatives**, **why**.

## Architecture (target, after M9)

```
prompt ──> tokenizer (byte-level BPE) ──> token ids
                                              │
weights (safetensors, mmap) ──upload once──>  GPU buffers (f32 / f16 / q4 blocks)
                                              │
            per token: one command encoder ──> [embed → N × (norm → attn(KV cache) → norm → MLP) → norm → lm_head]
                                              │        WGSL compute kernels, buffers reused, no allocation
                                              ▼
                         logits ──readback──> sampler (greedy / seeded top-k/top-p) ──> next token
                                              │
CPU reference forward pass (plain Rust f32) ──┘  oracle: every kernel and the full model are
                                                 differential-tested against it (D3, D5)
```

Today (M4, Tier 1 done): GPT-2 124M runs end to end on the GPU with naive kernels and a KV
cache, matching the CPU reference (itself checked against float64 numpy). Not yet built: the
per-token command encoder and buffer reuse (M7; today every op submits its own dispatch and
allocates its output), f16 / quantized weights (M9), and the sampler beyond greedy.

## Decisions

### D1: Stack: Rust + wgpu + WGSL
- **Alternatives:** TypeScript on the browser WebGPU API; C++ on Dawn (Chrome's WebGPU).
- **Why:** wgpu is the WebGPU implementation Firefox uses. It runs natively on Vulkan / Metal / DX12
  and compiles to wasm, so the same kernels run on the laptop and in a browser (M10). GPU code is
  testable headless with `cargo test`, which a browser stack makes hard on Linux. Same toolchain
  and workflow as lsmkv and lob. Trade-off: WebGPU lacks CUDA features (no tensor cores, limited
  subgroup ops, no float atomics), and that is the interesting constraint to talk about.
- **Hardware:** Intel Iris Xe (TigerLake GT2, integrated, shares system RAM) via Mesa's Vulkan driver.

### D2: Model path: GPT-2 124M, then a small Llama-family model
- **Alternatives:** a FaceRestore U-Net (convolutions); jumping straight to Llama.
- **Why:** GPT-2 is the simplest real transformer (learned positions, LayerNorm, GELU, MHA), with
  public weights and well-known outputs, so the first milestones are about correctness, not
  architecture. The Llama family (M8) then adds what interviews ask about now: RoPE, RMSNorm,
  SwiGLU, grouped-query attention, and quantized weights (M9). The U-Net is a Tier 3 stretch.

### D3: The CPU reference is the oracle
- **What:** a plain, obviously correct Rust implementation of every op (and in M1 of the whole
  GPT-2 forward pass). Every GPU kernel is differential-tested against it on edge-case and random
  shapes. The reference is checked once against pinned outputs of the original model.
- **Alternatives:** compare GPU outputs only against goldens dumped from PyTorch.
- **Why:** goldens only cover the inputs they were dumped for; an oracle covers any shape and any
  input, and points to the exact kernel that broke. Same idea as lob's reference book.

### D4: Determinism
- **What:** same device + same inputs gives bit-identical outputs. Kernels never use atomics on
  floats, reductions have a fixed order (fixed workgroup size, fixed tree), sampling takes a seed,
  and engine code never reads a clock (only the measurement module does).
- **Why:** float addition is not associative, so a reduction whose order depends on scheduling
  gives different bits run to run. Then a test failure can't be reproduced, and "did my change
  alter the output?" has no answer. Bit-identical across *different* GPUs is not promised (D5).

### D5: Compare floats with explicit tolerances
- **What:** `compare::check(got, want, Tol { abs, rel })` passes when every element satisfies
  `|got - want| <= abs + rel * |want|`, and NaN or infinity mismatches always fail. On failure it
  reports the worst index, both values and the error.
- **Alternatives:** bitwise equality; mean error only.
- **Why:** CPU and GPU legitimately differ: the GPU may fuse multiply-add, sums run in a different
  order, and WGSL allows `exp`/`tanh` within a few ULP. Bitwise equality would fail on correct
  kernels; a mean would hide one broken element. Tolerances are set per op (exact ops such as `add`
  use 0) and any loosening gets a note here.

### D6: Device setup
- **What:** one headless device (no window surface), high-performance adapter preference,
  requesting the adapter's own limits instead of WebGPU defaults. Blocking API on native via
  `pollster`; the async core stays reachable for wasm later (M10).
- **Why limits:** the WebGPU default `max_storage_buffer_binding_size` is 128 MiB, but GPT-2's token
  embedding is 50257 × 768 × 4 B = 147 MiB. Asking for what the adapter supports avoids splitting
  that tensor. `ember info` prints the limits we got.

### D7: Tensor layout
- **What:** contiguous row-major `f32`, shape kept on the host (`Vec<usize>`). A `GpuTensor` is a
  storage buffer plus its shape. No strides or views yet.
- **Why:** every kernel indexes `row * cols + col` and nothing else, which keeps the WGSL readable.
  Transposes happen once at load time instead of through strided reads. f16 and quantized layouts
  arrive with their milestones (M7, M9).

### D8: Elementwise dispatch with a grid-stride loop
- **What:** workgroup size 256; dispatch `min(ceil(n / 256), 65535)` workgroups; each invocation
  loops `i += 256 * num_workgroups`.
- **Alternatives:** one invocation per element with a 2D dispatch for large `n`.
- **Why:** the per-dimension workgroup limit is 65535, so one-thread-per-element tops out at
  16.7M elements, and GPT-2's embedding has 38.6M. The loop handles any `n` with a 1D dispatch.
  256 is a multiple of every vendor's SIMD width (Intel 8/16/32, AMD 32/64, NVIDIA 32).

### D9: Out-of-bounds accesses are contained, not trusted
- **What:** WebGPU guarantees that a shader can't read or write outside a bound buffer: wgpu adds
  bounds checks (or relies on Vulkan's robust buffer access), so an out-of-bounds write is dropped
  or clamped to the end of the buffer. Kernels still bounds-check by `params.n`, never by buffer size.
- **Why it matters:** in M0 the mutation `i < n` -> `i <= n` survives. Every output buffer holds
  exactly `n` floats, so the extra write lands outside the binding and is discarded: on this
  platform it's an equivalent mutant. Once buffers are reused from an arena (M7) and hold more
  than `n` floats, the same bug would silently corrupt the neighbour. M7's tests must use
  oversized buffers with sentinel values to catch it.

### D10: GPT-2 weights: safetensors, hand-parsed over mmap
- **What:** `openai-community/gpt2` `model.safetensors` (548 MB) plus `config.json`, `vocab.json`
  and `merges.txt` in `data/gpt2/` (git-ignored). A small hand-written parser reads the format
  (u64 little-endian header length, a JSON header mapping names to dtype, shape and byte offsets,
  then raw tensor bytes) over a `memmap2` mapping.
- **Alternatives:** the `safetensors` crate; converting to a custom format.
- **Why:** the format is simple enough to parse in about a page, and owning the parser means every
  step from bytes to tensors is explainable. mmap avoids copying 548 MB before it is needed.
- **Details:** the JSON header itself is parsed with `serde_json` (JSON parsing is not the point).
  The reader validates everything the header claims before trusting it: header length within the
  file and under the format's 100 MB cap, known dtypes, `shape × dtype size == end - begin` with
  overflow checks, and tensors that tile the data section exactly (no gaps, overlaps, or trailing
  bytes). Values are decoded with `f32::from_le_bytes` because the data section starts at
  `8 + N` and has no alignment guarantee. Conv1D weights (`[in, out]`) are transposed to
  `[out, in]` at load (D7); the 12 `attn.bias` causal-mask buffers in the file are ignored.
  `scripts/fetch_gpt2.sh` pins the checkpoint's sha256
  (`248dfc39…d3a707`, same as Hugging Face's LFS hash).

### D11: Tokenizer: hand-written byte-level BPE
- **What:** GPT-2's algorithm from `vocab.json` + `merges.txt`: the bytes-to-unicode map, the regex
  pre-split, then repeatedly merging the lowest-rank adjacent pair.
- **Alternatives:** Hugging Face's `tokenizers` crate.
- **Why:** it's heavy and hides the algorithm, and the tokenizer is a likely interview question.
- **Details:** the pre-split regex is OpenAI's pattern verbatim, run by `fancy-regex` because
  `\s+(?!\S)` needs look-ahead, which the `regex` crate deliberately lacks. Merges work on token
  ids, `(left, right) -> (rank, merged)`, rather than on strings, so there is no string building
  in the inner loop. Each round merges the lowest-rank pair at every position, left to right, as
  `encoder.py` does. `<|endoftext|>` in the input is plain text, as in OpenAI's encoder; Hugging
  Face maps it to id 50256 instead. Verified against Hugging Face `tokenizers` 0.23.2 on 23
  strings covering contractions (both cases), whitespace runs, CRLF, NBSP, `\x1c`–`\x1f` and
  `\x85`, combining marks, CJK, emoji ZWJ sequences, and the empty string
  (`tests/fixtures/gpt2_tokenizer_cases.json`).

### D12: Pinned reference outputs from a numpy-only script
- **What:** a one-off Python script that uses only `numpy` (no torch, since PyPI is slow on this
  network) writes token ids and logits for 3 prompts to `data/gpt2/golden/`. The Rust reference
  must match the logits to 1e-4.
- **Alternatives:** hard-coding known greedy completions.
- **Why:** text alone can match while the logits are wrong. Matching logits checks every layer.
- **Details:** the script computes in float64 (the truth, as far as f32 is concerned) and
  stores float32. It is written vectorized (whole-sequence matmuls, all heads at once,
  `-inf` masking) so it shares little structure with the Rust loops. Two extra checks keep it
  from being "the same author's bug twice". Token ids come from Hugging Face `tokenizers`
  (installed in the venv next to numpy, a deliberate deviation from "numpy only"). And its greedy output
  must reproduce the continuation Hugging Face published for "I enjoy walking with my cute dog";
  the script refuses to write goldens otherwise.
- **Measured (2026-10-05):** GPT-2 124M logits on 3 prompts (6, 10 and 21 tokens): worst
  absolute error 6.4e-4 on logits of magnitude ~100, worst relative 1.0e-5. The tolerance is
  `abs 1e-4 + rel 1e-4 × |want|`, so the relative term carries it, with 10× headroom. "Match to
  1e-4" in the table therefore means relative: a 1e-4 absolute bound would fail on correct
  code, because f32 accumulates about 1e-6 relative error per layer over 12 layers. Greedy
  continuations match token for token; the smallest top-2 logit gap in them is 0.028, far above
  the error. Tiny model: worst absolute error 1.2e-6 at tolerance 1e-5.

### D13: Reference matmul is a naive triple loop
- **What:** row-major f32, weights transposed once at load (D7), no BLAS.
- **Alternatives:** `ndarray` or BLAS.
- **Why:** the oracle has to be obviously correct, and it's allowed to be slow.
- **Measured (2026-10-05, wall clock, `ember generate`, release build, one core):** 0.60
  tokens/s for a 7–23-token sequence with no KV cache. Each dot product is one serial f32
  dependency chain, which can't vectorize without reordering the sum (changing the bits). Unoptimized
  it is about 30× slower, so `[profile.test]` uses `opt-level = 3`. The GPT-2 greedy test
  (52 steps, ~100 s) is `#[ignore]`d and runs with `cargo test -- --ignored`.
- **Threaded:** `linear` splits big products (≥ 2^20 multiply-adds)
  across `available_parallelism()` threads, in contiguous chunks of the output. Each element is
  still one serial sum, so results are bitwise identical to the serial loop (a test compares
  them bit for bit) and independent of the thread count. Measured the same way: 3.29 tokens/s
  on 8 threads (5.5×). The GPT-2 greedy test is back in the default run (~24 s).

### D14: Model shape comes from config.json
- **What:** `n_layer`, `n_head`, `n_embd`, `n_ctx` and `vocab_size` are read into a config struct.
  No sizes are hard-coded.
- **Why:** hard-coded constants would break for gpt2-medium/large, and they hide shape bugs that
  the tiny test models (D15) would otherwise catch.

### D15: Tests run without the weights
- **What:** two tiny models with odd sizes, both 2 layers, 3 heads, E = 12, vocab 37, context 16.
  `Weights::random` (seeded) feeds property tests: prefix rows are bitwise equal, token order
  matters, `next_logits` equals the last row. A committed fixture
  (`tests/fixtures/tiny_gpt2/`, 22 KB, written by the numpy script, including its `attn.bias`
  buffers) is checked against float64 logits and a 12-token greedy continuation. So the whole
  forward pass and the loader are tested on every `cargo test`, with no download. Tests that need
  the real weights skip, with a message saying how to fetch them.
- **Alternatives:** always requiring the weights.
- **Why:** CI and fresh clones shouldn't need a 548 MB download, and tiny models with odd sizes catch
  indexing bugs that 768-wide tensors make hard to debug.

### D16: Greedy decoding semantics
- **What:** `argmax` returns the first index among equal maxima, and `None` if any logit is NaN
  (generation then fails with an error). Generation stops early at the context length.
  `<|endoftext|>` doesn't stop it: the caller asks for `n` tokens and gets `n`.
- **Why:** first-max makes ties deterministic (D4) and matches `np.argmax`. NaN compares false
  with everything, so a plain max loop starting on a NaN returns index 0 forever. A broken
  model would then look like a model that likes token 0 (a test caught exactly that).

## M2: GPU kernels

### D17: Kernel order, simplest first
- **What:** `gelu` (elementwise) → residual `add` (M0's kernel) → `embed` (gather) →
  `softmax_rows` (row reduction) → `layer_norm` (two reductions per row) → naive `linear` →
  causal attention. The bias add is fused into `linear` (one read of the output instead of two).
- **Alternatives:** matmul first, since it is most of the time.
- **Why:** each kernel adds one new idea (elementwise, gather, reduction, 2-D dispatch, a
  reduction inside a gather). A failure then points at the new idea, not at five at once.

### D18: Row reductions: one workgroup per row, fixed tree
- **What:** 256 threads per row. Thread `i` sums elements `i, i+256, i+512, …` serially, then the
  256 partial sums are combined in shared memory by a fixed halving tree (128, 64, …, 1) with a
  `workgroupBarrier()` between levels.
- **Alternatives:** one thread per row; atomics.
- **Why:** a thread per row leaves the GPU mostly idle (T rows, often < 100) and makes each thread
  walk 768 or 50257 elements alone. Float atomics don't exist in WebGPU, and their ordering would
  break determinism anyway (D4). The fixed tree gives the same bits every run. Its order differs
  from the CPU's serial sum, which is why reductions get a nonzero tolerance (D22).
- **Barriers can't be proven by tests.** Removing any of the three "read before overwrite" or
  "write before read" barriers (softmax, layer_norm, attention) leaves every test passing on the
  Iris Xe. Its SIMD groups within a workgroup run nearly in step, so the race window never
  opens. A race-free test run says nothing about another GPU (M10 runs these kernels in a
  browser on other hardware). So every barrier carries a comment naming the write and read it
  orders. Those three surviving mutants are recorded in D22.

### D19: Naive matmul: one thread per output element
- **What:** `linear` dispatches 16×16 workgroups over (output feature, row). Each thread does the
  serial dot product of `x[i, :]` and `w[o, :]` (both contiguous, D7), then adds the bias.
- **Alternatives:** tiling now.
- **Why:** it is the baseline M6 is measured against, and its accumulation order matches the CPU's
  exactly, so the only difference is fused multiply-add. `x` reads are shared by the 16 threads of
  a row, `w` reads are not coalesced (neighbouring threads read rows `n_in` apart). That's
  the known weakness M6 fixes.

### D20: Naive attention: one workgroup per (head, query row)
- **What:** dispatch `(T, H)` workgroups of 64 threads. Scores for keys `0..=t` go into a
  shared-memory array of `n_ctx` floats (4 KiB for 1024), the max and sum use the fixed tree
  (D18), then each thread produces output dimensions `c, c+64, …` as a serial weighted sum over j.
- **Alternatives:** fused online softmax (FlashAttention), which is M7.
- **Why:** it mirrors the CPU loop structure closely enough to debug, and it is causal for free:
  keys after t are never scored. The shared array caps T at the compiled `MAX_CTX` (1024); the
  op rejects longer inputs instead of corrupting memory.

### D21: Kernel parameters in a 16-byte-aligned uniform per dispatch
- **What:** every kernel takes its sizes in a small uniform struct (padded to a multiple of 16
  bytes, as M0's `add`). Pipelines are compiled once in `Gpu::new` (M0, D6).
- **Why:** sizes vary per call (T grows during generation), and uniforms are the WebGPU way to
  pass a few scalars. Per-dispatch buffers are allocated for now; M7's arena removes that.

### D22: Per-kernel tolerances, measured
- **What:** each GPU op is compared to `cpu.rs` with its own `Tol`, set from the measured worst
  error with headroom, and the measured value recorded in DESIGN.md. Every kernel is also run
  twice and must produce identical bits (D4).
- **Why:** sources of difference vary by op: none (add), `tanh`/`exp` implementations (gelu,
  softmax), summation order (reductions), fused multiply-add (linear). One global tolerance
  would be either too loose for the exact ops or failing for the reductions.
- **Measured (2026-10-05, Iris Xe, Mesa 26.2.2), worst over each op's test shapes:**

  | Op | Tolerance (abs, rel) | Worst measured | Main source of difference |
  |---|---|---|---|
  | `add`, `embed` | exact | 0 | one IEEE add on both sides |
  | `gelu` | 1e-6, 1e-5 | 9.5e-7 abs | the drivers' `tanh` vs libm's |
  | `softmax_rows` | 1e-7, 1e-5 | 6.7e-8 abs | `exp`, tree vs serial sum; subnormals flushed to 0 |
  | `layer_norm` | 2e-5, 1e-5 (was 1e-4) | 5.7e-6 abs (was 8.8e-5) | tree sums, `sqrt`; before D23 mostly the CPU's own error |
  | `linear` | 1e-5, 1e-5 | 2.9e-6 abs (n_in 3072) | fused multiply-add: same order, one rounding per step |
  | `causal_attention` | 5e-6, 1e-4 | 7.8e-7 abs | dot products, `exp`, tree softmax |

- **LayerNorm's tolerance measures the oracle, not the kernel.** Inputs around 100 with a spread
  of ~3, like GPT-2's residual stream: against a float64 LayerNorm, the CPU's serial f32 sum is
  5.2e-5 off (3072 columns) and the GPU's tree sum 4.4e-6. A serial sum's rounding error grows
  with n; a tree's with log n. The error in the mean is then divided by the standard
  deviation. Headroom over the measured 8.8e-5 is small, but the inputs are seeded and the
  kernel deterministic, so the test can't flake. Resolved by D23: the tolerance is now 2e-5.
- **Mutation results (27 WGSL mutants):** 23 caught. The 4 survivors:
  - The three barrier removals (D18).
  - Removing gelu's clamp before `tanh`. Mesa's `tanh` already returns ±1 for large arguments, so
    on this driver the clamp is an equivalent mutant, like `i <= n` in D9. It stays for drivers
    that compute tanh as `(e^2u - 1) / (e^2u + 1)` = inf / inf.

  Four "wrong max" mutants (softmax ×3, attention ×1) first survived. Softmax is invariant to
  the value subtracted, so a slightly wrong max only shows when `exp` overflows. Tests with one
  dominant logit (100 among ±5, positioned so each mutant misses it) now catch all four.

### D23: The CPU LayerNorm computes in f64
- **What:** mean, variance and every output of `cpu::layer_norm` are computed in f64 and
  rounded to f32 once.
- **Alternatives:** pairwise f32 summation; leaving the tolerance at 1e-4.
- **Why:** D22 found the f32 oracle 12× *less* accurate than the GPU's tree sum. A tolerance
  that mostly measures the oracle hides real kernel errors. In f64 the oracle's error is ~0.5 ulp,
  so the GPU tolerance tightened from 1e-4 to 2e-5 (measured worst 5.7e-6). GPT-2 logits vs numpy
  improved slightly: worst relative error 1.0e-5 → 9.5e-6. The other reductions (softmax, attention)
  have small enough errors that they stay f32.

## M3: GPT-2 on the GPU

### D24: Weights are uploaded once
- **What:** `GpuWeights::upload(&Weights)` mirrors `Weights` with a `GpuTensor` per tensor,
  ~498 MB of f32 (124M parameters). The Iris Xe shares system RAM, and the biggest single
  binding (`wte`, 147 MiB) is under the 2047 MiB limit (D6).
- **Why:** weights are read every token and never change; uploading per forward pass would make
  PCIe/memcpy the bottleneck. f16 storage is a Tier 3 stretch.

### D25: One queue submit per op (for now)
- **What:** the GPU forward pass calls the M2 ops in order; each validates, dispatches and submits
  on its own, and wgpu orders submissions on the queue. No readback until the logits.
- **Alternatives:** one command encoder per forward pass.
- **Why:** the ops stay independently testable with the signatures M2 tested. Submit
  overhead is measured in M5, and M7 merges a token into one encoder.

### D26: The LM head runs on the last row only during generation
- **What:** `ops::row` copies one row into its own buffer with `copy_buffer_to_buffer` (no
  kernel), and `linear` runs on `[1, E]`. The full `[T, V]` logits are only computed to compare
  every position in tests.
- **Why:** the LM head (`V × E` = 38.6M multiply-adds per row) is the single largest matmul;
  only the last row matters for the next token.

### D27: Correctness bar for M3
- **What:** GPU logits are compared with the CPU reference (not the numpy goldens: the oracle is
  the reference, D3) on the tiny model and the 3 golden prompts, with a tolerance measured and
  recorded here. Greedy tokens must be identical on all 4 golden continuations, including
  HF's published one.
- **Why:** per-op tolerances (D22) compound over 12 layers. The end-to-end number shows how
  much, and identical greedy text shows it doesn't matter where it counts.
- **Measured (2026-10-05):** tiny model 1.2e-6 abs. GPT-2 124M on 3 prompts: worst 2.1e-4 abs
  on logits of magnitude ~100, 2.5e-6 relative; tolerance `abs 1e-4 + rel 1e-5`. The GPU is
  closer to the CPU reference than the reference is to float64 numpy (9.5e-6). All 4 greedy
  continuations are identical. Prefix rows, last-row logits and repeated runs are *bitwise*
  equal on the GPU, as on the CPU, because every kernel computes a row the same way whatever
  else is in the batch.

### D28: Readback only the next-token logits; argmax on the CPU
- **What:** each generation step reads back `[1, V]` (201 KB) and runs `cpu::argmax`.
- **Alternatives:** an argmax kernel (one more reduction), reading back 4 bytes.
- **Why:** simplest correct thing; whether 201 KB of readback matters is a question for M5's
  timings, not a guess.

### D29: `ember generate` runs on the GPU by default
- **What:** `ember generate [--cpu] [-n N] <prompt>`; both paths print tokens/s labelled "no KV
  cache" until M4.
- **First numbers (2026-10-05; wall clock from the first generated step to the last, weight
  upload excluded; release build; one run each; prompt "I enjoy walking with my cute dog", 7
  tokens + 40 generated):** GPU 1.86 tokens/s (21.5 s), CPU reference on 8 threads 1.20
  tokens/s (33.4 s). Over 16 tokens: 3.18 vs 3.29. Without a KV cache step t costs O(t), so
  rates fall as sequences grow. This is not a fair "GPU vs CPU" claim yet: naive kernels
  (D19, D20), ~160 queue submits per step (D25), and a per-step readback. M4 (KV cache) and M5
  (per-kernel timings) are where real numbers start.

## M4: KV cache and incremental decode

### D30: KV cache layout
- **What:** per layer, a K buffer and a V buffer of `[n_ctx, E]`: row = absolute position, heads
  side by side (head h owns columns `h*D..(h+1)*D`), the same layout as the K and V thirds of a
  `qkv` row. Allocated once at full context: 12 × 2 × 1024 × 768 × 4 B = 75.5 MB for GPT-2 124M.
- **Alternatives:** grow per token; head-major `[H, n_ctx, D]`.
- **Why:** fixed buffers mean no reallocation or copying as the sequence grows, and one layout
  for `qkv` and cache keeps the indexing identical in both. Head-major gives each head a
  contiguous `[n_ctx, D]` block, which a fused kernel (M7) may want. That's decided there, with
  measurements.

### D31: One attention kernel for prefill and decode
- **What:** the M2 kernel now takes queries from `qkv` rows and keys/values from the cache. Query
  row i sits at absolute position `start + i` and attends cache rows `0..=start+i`. Prefill is
  `start = 0` with T rows; a decode step is `start = t` with 1 row. `ops::causal_attention` (the
  M2 op) becomes "write a temporary cache, then attend", so its tests still cover the kernel.
- **Alternatives:** a separate decode kernel.
- **Why:** one kernel to test and reason about. Decode with one query row is exactly prefill's
  last row, which is what makes D33's bitwise check possible.

### D32: `kv_write` kernel
- **What:** scatters the K and V columns of `qkv: [T, 3E]` into cache rows `start..start+T`. The
  host checks `start + T <= n_ctx`. The positional embedding takes the same `start` (`ops::embed`).
- **Why:** prefill writes T rows per layer in one dispatch instead of 2T buffer copies.

### D33: Decode is checked bitwise against full recompute
- **What:** at every step, the cached path's next-token logits must have the same *bits* as the
  uncached GPU forward pass on the same prefix. Greedy output on the 4 golden continuations must
  be identical too.
- **Alternatives:** a tolerance.
- **Why:** every kernel computes a row the same way whatever else is in the batch (D27), and
  cached K/V rows are exactly the rows full recompute would produce. So any difference is a bug:
  a wrong position, a stale or misplaced cache row, an off-by-one in `start`. A tolerance could
  hide an off-by-one whose logits happen to be close.

### D34: The CPU reference stays full-recompute
- **Why:** the oracle checks *what* the model computes. D33 already checks the cache against full
  recompute, which the CPU oracle checks in turn. A cached CPU path would add code to the
  oracle that nothing needs.

### D35: Prefill and decode are measured separately
- **What:** `ember bench`: 1 warm-up run, then 5 measured runs; report the median prefill
  latency (prompt → first logits) and the median decode rate (tokens/s over n steps), wall
  clock, with prompt and output lengths stated. The uncached rate is measured the same way for
  comparison.
- **Why:** prefill is a batch of matrix-matrix products (compute-bound); decode is one row per
  step, matrix-vector (memory-bound). One mixed tokens/s hides both. Medians resist the
  occasional slow run; the warm-up excludes pipeline and driver first-use costs.
- **First results (2026-10-05, `ember bench -n 32 "I enjoy walking with my cute dog"`, Iris Xe /
  Vulkan / Mesa 26.2.2, release build, prompt 7 tokens, 32 decode steps):**

  | | Median |
  |---|---|
  | Prefill (7 tokens → first logits) | 134.7 ms |
  | Decode with KV cache | 50.6 ms/token = **19.76 tokens/s** |
  | No cache (M3, prompt + 33 tokens recomputed every step) | 462.9 ms/token = 2.16 tokens/s |

  So the cache is 9.1× faster at this length, and the gap grows with length (uncached step t
  costs O(t) of everything, cached only O(t) attention). A decode step reads all ~498 MB of
  f32 weights once, so 50.6 ms means ~10 GB/s effective. This laptop's LPDDR4x peak is around
  50–68 GB/s (unmeasured here; M6 measures the roofline), so naive decode reaches roughly a
  fifth of it. That's the gap the uncoalesced matrix-vector reads (D19) leave for M6.

## M5: Measurement

### D36: GPU time comes from timestamp queries
- **What:** with `Features::TIMESTAMP_QUERY` (requested when the adapter has it), every compute
  pass gets `timestamp_writes` at its beginning and end. Each dispatch is its own pass (D25), so
  each pair brackets exactly one kernel. The query set is resolved into a buffer, read back, and
  `(end - begin) × queue.get_timestamp_period()` gives nanoseconds. On the Iris Xe / Mesa 26.2.2
  one tick is 52.08 ns (a 19.2 MHz counter).
- **Alternatives:** wall clock around each op with a blocking `poll`.
- **Why:** the wall clock measures CPU + driver + GPU and, with a wait after every op, also
  destroys the CPU/GPU overlap it is trying to measure. Timestamps are written by the GPU
  itself. Buffer copies (`ops::row`, readback) aren't passes and aren't timed; they show up in
  the gap between kernel time and step wall time (D38).

### D37: The profiler is opt-in and lives in `src/profile.rs`
- **What:** `Gpu` holds an optional profiler behind a `Mutex`. `gpu.profile_start()` turns it on,
  every `dispatch` then records its kernel's name and two query slots, and
  `gpu.profile_finish()` resolves and returns `(kernel, ns)` per dispatch. Off (the default),
  `dispatch` creates no queries.
- **Why:** nothing the engine computes may depend on timing (D4), so the clock is read only by
  the profiler and the CLI. A test checks that logits are bitwise identical with profiling on.

### D38: Report: per-kernel table plus wall clock
- **What:** `ember profile` prints, for the prefill and for one decode step, each kernel's call
  count, GPU ms and share of the step, the kernels' total, and the step's wall-clock time. The
  difference is everything that isn't a kernel: submits, buffer allocation, copies, readback,
  CPU work.

### D39: Method: 1 warm-up, median of 5
- **What:** each kernel's per-step time is the median over 5 measured runs after 1 warm-up; the
  same for wall clock. Device, driver, prompt and lengths are printed with every table.

### D40: Context sweep
- **What:** decode ms/token measured at positions ≈ 8, 128, 512 and 1000 (the cache is filled by
  a prefill of that length first), with the attention kernel's share at each.
- **Why:** attention is O(t) per step while the matmuls are fixed; the sweep shows where
  attention starts to matter.

### D41: Measured bandwidth roofline
- **What:** a `copy` kernel (vec4 loads and stores, grid-stride) moves a 256 MiB buffer; bytes
  read + written over its timestamped GPU time gives achievable GB/s. Decode's weight bytes over
  its kernel time is then compared with that, not with a datasheet number.

### D42: First per-kernel results
`ember profile "I enjoy walking with my cute dog"` (2026-10-07, Iris Xe / Vulkan / Mesa 26.2.2,
release build, 7-token prompt, median of 5 after 1 warm-up; GPU ms from timestamps, step ms
from the wall clock):

| Step | `linear` | attention | everything else | all kernels | wall clock | not in kernels |
|---|---|---|---|---|---|---|
| Prefill (7 tokens) | 127.3 ms (94.5%) | 0.32 ms | 0.63 ms | 128.3 ms | 134.7 ms | 6.4 ms |
| Decode (position 7) | 37.3 ms (83.5%) | 0.36 ms | 0.53 ms | 38.3 ms | 44.7 ms | 6.4 ms |

| Decode at position | 8 | 128 | 512 | 1000 |
|---|---|---|---|---|
| Wall ms/token | 43.9 | 43.2 | 44.8 | 49.4 |
| Attention share of GPU time | 1.0% | 4.2% | 12.9% | 23.4% |

Bandwidth: the copy kernel moves 256 MiB in and 256 MiB out in 20.1 ms = **26.7 GB/s**. A decode
step reads 495 MB of weights: 12.9 GB/s over its kernel time, **48% of the measured roofline**
(42% over wall time).

A second invocation minutes later gave decode kernels 34.9 ms (52% of a 27.1 GB/s roofline),
prefill 120.9 ms, 5.2–6.0 ms outside kernels, attention 22.3% at 1000: whole runs vary by ~8%
(a laptop iGPU's clocks and thermals), while the shares and ratios hold. Comparisons between
kernels (M6) are therefore made within one invocation.

What this says:
- **Matmul is the whole story** (83–95% of every step), so M6 is the right next milestone. Prefill's
  `linear` time is 3.4× decode's for 7× the rows: the naive kernel rereads all of `w` for every
  row instead of reusing it, which is what tiling fixes.
- **Decode is far from the memory roof.** It reaches half of what a plain copy achieves, so a
  coalesced matrix-vector kernel (M6) has up to ~2× left before bandwidth, not compute, caps it.
- **6.4 ms per step (14% of decode) isn't GPU work:** 135 separate submits, 135 output
  allocations, the row copy and the readback. That is M7's target (one encoder per token, a buffer
  arena).
- **Attention is small until the context is long:** O(t) per step, a quarter of decode GPU time
  at 1000. The matmuls are fixed, so attention only becomes the target once they are fast.
- **Caveat on the roofline:** 26.7 GB/s is what this copy kernel achieves, a lower bound on the
  hardware's. The datasheet peak (LPDDR4x, ~50–68 GB/s) was not reached. The D35 note's "a fifth
  of the bandwidth" was against the datasheet; against the measured number it is about half.

## M6: Fast matmul

M5 found `linear` at 83–95% of every step (D42), so M6 replaces it. The naive kernel stays as
the baseline (D49).

### D43: Prefill matmul: shared-memory tiling with register blocks
- **What:** `matmul.wgsl`. A 16×16 workgroup computes a 64×64 block of the output, and each
  thread computes a 4×4 block of it in registers: rows `ty + 16r`, columns `tx + 16c`, so
  neighbouring threads touch neighbouring columns. K is walked 16 at a time: the workgroup stages
  a 64×16 slice of `x` and a 16×64 slice of `w` in shared memory (8 KiB), and every thread
  does its 16×4×4 multiply-adds from there.
- **Alternatives:** register blocking without shared memory; one thread per output (D19).
- **Why:** naive does one global load per multiply-add. Here each staged value is used by 16
  threads, and each loaded register value by 4 multiply-adds, which turns a memory-bound loop
  into a compute-bound one. The strided 4×4 assignment, instead of a contiguous 4×4 block, makes
  the inner loop's shared-memory reads and the final stores land on consecutive addresses.
- **Global loads are scalar, not vec4** (the M6 table recommended vec4). GPT-2's LM head has
  50257 columns, so `w` rows aren't 16-byte aligned. Consecutive threads read consecutive
  scalars, which the hardware coalesces anyway. The tile sweep (D48) measured whether vec4 would
  pay.

### D44: Decode (T = 1) gets a matrix-vector kernel
- **What:** `ops::linear` dispatches `matvec.wgsl` when `T = 1`, and `matmul` otherwise. The
  matvec has one thread per output. At every k, a workgroup's 256 threads read 256 consecutive
  floats of row k of `w`. `x` is staged in shared memory 256 values at a time, because every
  thread needs the same `x[k]`.
- **Alternatives:** one kernel for both.
- **Why:** with one row, a 64-row tile computes 63 rows of nothing, and decode is memory-bound
  anyway: what matters is reading each weight once, coalesced.

### D45: GPU weights are `[in, out]`; the token table is stored only transposed
- **What:** `GpuWeights::upload` transposes every linear weight back to `[in, out]` (the
  checkpoint's own Conv1D layout). The token table is uploaded once, as `wte_t: [E, V]`: that is
  the tied LM head's `[in, out]`, and `embed.wgsl` reads a token's embedding as column `id`
  (E reads, V floats apart). The CPU oracle keeps `[out, in]` and `[V, E]` (D7).
  `ops::linear_naive` takes the CPU layout.
- **Alternatives (owner's choice, 2026-10-07):** keep both `wte` and `wte_t` (+154 MB); keep
  `[V, E]` for the head and give the matvec a layout flag. Also considered: keep `[out, in]`
  and split K across a workgroup, which is coalesced too but needs a tree sum (D46).
- **Why:** in `[in, out]`, threads that own neighbouring outputs read neighbouring addresses at
  every k, in both kernels. The LM head is 154 MB of a 495 MB decode step, so it has to be in the
  fast layout too. One copy avoids 154 MB of extra memory on a GPU that shares system RAM. The
  embed lookup's strided reads cost 768 loads per token. Upload transposes on the CPU once per
  load.

### D46: Decode stays bitwise equal to full recompute
- **What:** `matmul`, `matvec` and `linear_naive` all compute each output as the serial sum
  `acc = fma(x[k], w[k, o], acc)` for k = 0, 1, …, n_in − 1, then add the bias. The tiled kernel
  stops its last K step at n_in instead of multiplying zero padding. A test checks that every row
  has the same bits from all three kernels, in a batch or alone.
- **Alternatives:** split-K in the matvec (more parallelism, a different summation order) with
  a tolerance between decode and recompute.
- **Why:** D33's bitwise check catches cache bugs a tolerance would hide, and it only holds if
  a row's bits don't depend on the batch it ran in. The kernels use WGSL's `fma()` explicitly.
  WGSL lets an implementation evaluate `fma` as a fused or an unfused multiply-add, so equal bits
  across kernels are a property this driver is *tested* for, not one the language guarantees.
  If a driver breaks it, the test says so.
- **Revisit if:** the matvec is under ~70% of the bandwidth roofline. Only `n_out` threads run
  (768 for `attn_out` and `fc_out`), which may be too few to hide memory latency. The fix would
  be split-K with a fixed combine order, and the tiled kernel would combine in the same order.
  That's a design change and goes back to the owner.

### D47: Measuring M6
- **What:** `ember matmul` times naive vs tiled vs matvec on GPT-2's shapes with timestamp
  queries (median of 5 after 1 warm-up). It reports GFLOP/s for the matmuls, against a measured
  compute roof (a kernel of independent FMA chains), and GB/s of weights for the matvec, against
  the copy roofline (D41). The model-level before/after is `ember profile`, run back to back on
  the same machine state.
- **Why:** a matmul is judged against what the hardware can do, not against the old kernel only.

### D48: Tile sizes from a small sweep, fixed as constants
- **What:** a few configurations were measured with `ember matmul` and the best is compiled
  in. No runtime autotuning.
- **Why:** one target GPU for now; autotuning is machinery without a second device to justify it.

### D49: The naive kernel stays, as `linear_naive`
- **Why:** it's the benchmark's baseline, and an independent implementation to compare the
  tiled kernels with bit for bit (D46).
