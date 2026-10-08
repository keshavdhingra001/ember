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

Today (M6): GPT-2 124M runs end to end on the GPU with a KV cache, a tiled matmul for prefill
and a matrix-vector kernel for decode, matching the CPU reference (itself checked against
float64 numpy). Not yet built: the per-token command encoder and buffer reuse (M7; today every
op submits its own dispatch and allocates its output), f16 / quantized weights (M9), and the
sampler beyond greedy.

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
- **Registers are four named `vec4`s, not arrays.** The first version kept the 4×4 block in
  `array<array<f32, 4>, 4>`, and its loaded values in two `array<f32, 4>`. Mesa didn't keep
  those loop-indexed private arrays in registers. The kernel ran at 113 GFLOP/s, and the
  compute probe written the same way measured the same 116 GFLOP/s, so it looked like a
  kernel at 97% of its roof. Rewriting the probe with plain variables gave 1431 GFLOP/s
  (D47). With `acc0..acc3: vec4` (lane c = column `tx + 16c`) and shared memory packed so
  that one vec4 load gives a thread its 4 values for one k, the matmul went from 113 to
  455 GFLOP/s. Each output still runs the same serial fma sequence (D46).
- **Global loads are scalar, not vec4** (the M6 table recommended vec4). GPT-2's LM head has
  50257 columns, so `w` rows aren't 16-byte aligned. Consecutive threads read consecutive
  scalars, which the hardware coalesces anyway. Whether vec4 loads would pay where the
  shapes allow them is **not measured** yet.

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
- **Padding the last K step would not change any bits.** A padded step adds `fma(0, 0, acc)`,
  and `acc + 0 = acc` for every acc the kernel can hold (acc starts at +0, and a sum that
  rounds to zero is +0). So stopping at n_in only saves work. Mutation check: D50.
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
- **Matvec sweep (2026-10-07):** workgroup size × weight loads in flight per thread (a
  "lookahead": issue 8 loads, then their 8 fmas, in k order, so D46 holds). One decode step's
  matrices, ms of GPU time (`ember matmul`, decode table):

  | Workgroup | Loads in flight | Step ms | LM head GB/s | `fc_out` GB/s |
  |---|---|---|---|---|
  | 64 | 1 | 42.2 | 26.4 | 5.8 |
  | 64 | 4 | 29.0 | 16.5 | 12.0 |
  | 64 | 8 | 27.2 | 16.3 | 13.9 |
  | 256 | 1 | 38.9 | 30.3 | 6.4 |
  | 256 | 4 | 28.3 | 16.1 | 12.7 |
  | 256 | 8 | 25.7 | 16.5 | 15.5 |

  No single configuration wins everywhere. The narrow matrices (768–3072 outputs, so 768–3072
  threads) need several loads in flight per thread to hide memory latency. The LM head (50257
  outputs) has enough threads without them and loses 45% with them; the likely cause is the
  extra registers lowering occupancy (not verified). So the matvec is compiled twice from one
  source with an `override LOOKAHEAD: bool`: workgroup 256, lookahead below 16384 outputs, none
  from there on (`matvec_wide`). The 16384 boundary sits between the two measured sizes; it is
  not itself measured. Result: 22.2 ms per step's matrices, against 25.7 for the best single
  configuration.
- **Not swept:** the matmul's tile sizes (64×64×16, 4×4 per thread) and vec4 global loads.

### D49: The naive kernel stays, as `linear_naive`
- **Why:** it's the benchmark's baseline, and an independent implementation to compare the
  tiled kernels with bit for bit (D46).

### D50: M6 results
`ember matmul` (2026-10-07, Iris Xe / Vulkan / Mesa 26.2.2, GPU time from timestamp queries,
median of 5 after 1 warm-up, random weights):

- **Roofs.** Compute: **1431 GFLOP/s** (`fma_peak`, 32 independent fma chains per invocation).
  Read bandwidth: **29.6 GB/s** (`read_peak`, a 256 MiB stream of vec4 reads). Copy: 25.3 GB/s
  read + written (D41). A read-only stream beats copy's reads + writes, so decode, which only
  reads weights, is measured against the read roof.
- **Compute probe pitfalls.** Written with an array of chains, the probe measured 116 GFLOP/s
  (the D43 register problem). Per invocation: 8 scalar chains 900, 4 vec4s 1177, 8 vec4s 1431,
  16 vec4s 489 (out of registers).
- **Prefill (tiled vs naive), GFLOP/s:**

  | Matrix (in → out) | T = 7 | T = 128 | T = 512 |
  |---|---|---|---|
  | qkv (768 → 2304) | 7.8 → 26.6 | 12.3 → 340 | 11.9 → 457 (32% of roof) |
  | attn_out (768 → 768) | 8.5 → 17.8 | 11.9 → 238 | 12.3 → 405 |
  | fc (768 → 3072) | 10.5 → 35.2 | 12.2 → 376 | 12.1 → 466 |
  | fc_out (3072 → 768) | 8.5 → 17.8 | 10.4 → 281 | 10.5 → 411 |
  | lm_head (768 → 50257) | 12.7 → 46.7 | | |

  25–39× at T ≥ 128. At T = 7 only 2–4×: a 7-row batch uses 7 of each 64-row tile, and
  with one tile row there are only `n_out / 64` workgroups (12 for a 768-output matrix), so
  the GPU is mostly idle.
- **Decode (one step's matrices, 12 distinct copies of each block matrix plus the head, in
  model order):** naive 34.3 ms → matvec 22.8 ms; **21.7 GB/s = 73% of the read roof** (49%
  naive). Per matrix: qkv 86%, fc 93%, LM head 92%, but attn_out and fc_out (768 outputs)
  only 49–50%: 768 threads can't keep enough reads in flight. An earlier version timed one
  matrix over and over, which kept it in the last-level cache (shared with the CPU) and
  reported up to 187% of the bandwidth roof.

Model level: `ember profile` and `ember bench -n 32` on the 7-token prompt, the pre-M6 binary
(`ca030de`) and M6 run back to back on an idle machine:

| | Before | After | |
|---|---|---|---|
| Prefill, kernels | 123.3 ms | 52.2 ms | 2.4× |
| Prefill, wall | 129.5 ms | 58.3 ms | 2.2× |
| Decode step, kernels | 35.9 ms | 23.5 ms | 1.53× |
| Decode step, wall | 41.5 ms | 28.5 ms | 1.46× |
| Decode, `ember bench` | 25.3 tokens/s | **35.8 tokens/s** | 1.42× |
| No KV cache, `ember bench` | 2.81 tokens/s | 16.9 tokens/s | 6.0× |
| Decode weight bandwidth, kernels | 13.8 GB/s | 21.1 GB/s (71% of the read roof) | |

- The decode wall clock now has ~5 ms outside kernels out of 28.5 (submits, per-op buffers,
  readback): M7's target. Attention reaches 30% of GPU time at position 1000 (21% before),
  because everything else got faster.
- The 7-token prefill's matmuls take 46 ms, almost 3 decode steps for 7 rows. They should
  cost about one weight read (~17 ms) since the batch is far too small to be compute-bound.
  That's the same parallelism problem as the narrow matvecs (D46).
- `embed` went from 0.006 to 0.07 ms in the prefill (strided reads of `wte_t`, D45):
  negligible, as expected.
- The before numbers differ by a few % from D42's (same binary, another day). The gains above
  compare runs taken minutes apart.
- **Mutation pass (27 mutants in matmul, matvec, embed and the weight upload; `--test gpu_ops
  --test gpt2_gpu`): 24 caught.** All four barrier removals were caught, unlike M2's (D18):
  a 64×64 tile spans many SIMD groups, so the race window opens. A swapped pair of fmas in
  the matvec's lookahead (same numbers, a different summation order) is caught only by the
  bitwise tests (D46), not by the tolerance against the CPU. The 3 survivors:
  - matmul running padded K steps: equivalent, as D46 predicts.
  - matvec's lookahead bound `k + 8 <= steps` → `<`: the last 8 values move to the plain loop,
    in the same order. Equivalent.
  - matmul writing row `t` (one past the end): this driver drops out-of-bounds writes. WebGPU
    also allows them to land elsewhere in the same buffer, so the check stays (like D22's gelu
    clamp).

- **Later (D56, D57):** the split-K follow-up took decode to 37.6 tokens/s and the 7-token
  prefill to 34 ms, measured on a quiet machine against the same pre-split binary (table at the
  end of D57). The numbers above are M6's own and stay as they were.

## M6 follow-up: split-K (approved and built 2026-10-07)

D50 left two gaps with one cause, too few threads: the 768-output matvecs run at 50% of the
read roof, and the 7-token prefill's matmuls cost about 3 decode steps.

### D51: The linear kernels compute a chunked sum
- **What:** per output, K is split into chunks of 256. Each chunk is a serial fma chain starting
  from 0, and the chunk partials are added in chunk order: `((p0 + p1) + p2) + …`, then the
  bias. `matmul` folds its accumulators into running totals every 16 BK steps; `matvec` computes
  the partials in parallel and combines them in the same order.
- **Alternatives:** split-K with a tolerance between decode and full recompute.
- **Why:** it gives the matvec K/256 times more threads and keeps D33's bitwise decode check,
  because both kernels still run one identical sequence of operations per output.

### D52: Matvec: 64 outputs × 4 K slices per workgroup (amended by D56)
- **What:** 256 threads = 64 consecutive outputs × 4 slices. Slice threads compute the partials
  of their chunks; the workgroup adds them in chunk order through shared memory, in rounds of 4
  chunks, so any K works. `attn_out` goes from 768 to 2304 threads, `fc_out` to 9216.
- **Alternatives:** more loads in flight (D48 already found 8 to be the best).

### D53: The naive kernel stays a plain serial sum
- **What:** `linear_naive` leaves the bitwise test (its order now differs) and keeps the
  tolerance test against the CPU.

### D54: The CPU oracle stays serial
- **What:** the tolerance stays 1e-5, and the measured error under the chunked order is recorded.
- **Measured (`linear_matches_cpu`, random x in ±1, w in ±0.1):** the worst absolute error grew
  from 2.9e-6 to 1.24e-5 at n_in = 3072 (LM head shape: 1.7e-6 → 5.0e-6). Against the
  tolerance `1e-5 + 1e-5 · |want|`, the worst element uses 69% of it (before: at most 29%, since
  2.9e-6 < 0.29 · 1e-5). What grew is the distance between two orders of the same sum, not
  necessarily the error against the true value: a serial sum's error bound grows with n, a
  chunked sum's with 256 + n / 256, so the CPU's order is usually the less accurate one (not
  measured against float64 here).
- **Also tested:** `linear_is_bitwise_the_chunked_sum` writes D51 out on the CPU with
  `f32::mul_add` and requires the GPU's bits to match exactly, for all three matvec
  configurations and the tiled matmul. Without it, a wrong chunk size in both kernels would pass
  the matmul-vs-matvec bitwise test and the tolerance.

### D55: Small-T prefill is decided after measuring D51–D52 (decided: D57)
- **What:** measure first, then choose between split-K in the matmul for small T and a narrower
  tile. Expected gain from D51–D52 (an estimate, to be measured): attn_out + fc_out from 9.6 to
  ~5.5 ms per step, decode ~28.5 → ~24 ms (~41 tokens/s).

### D56: The matvec splits K only up to 1024 outputs
- **What:** `matvec.wgsl` takes the slice count as an override constant (`SLICES`, like
  `LOOKAHEAD`, D48) and is compiled three times: `matvec_split` (4 slices, lookahead) up to 1024
  outputs, `matvec` (1 slice, lookahead) below 16384, `matvec_wide` (1 slice, none) above. The
  slice count only decides which thread computes a chunk's partial, never the order the partials
  are added in, so all three give the same bits (tested).
- **Alternatives (owner's choice, 2026-10-07):** D52 as approved (4 slices everywhere); the best
  variant per matrix from the sweep below (overfits three noisy runs).
- **Why:** D52 as approved made the decode step *slower* (22.7 → 25.5 ms). Splitting K helps the
  768-output matrices, which lacked threads, and hurts the wide ones, which didn't: a slice's 64
  threads read 256-byte runs of a row of `w` instead of 1 KiB. That explanation fits the numbers
  but isn't verified.
- **Sweep (2026-10-07, `ember matmul` decode table, GPU ms per step's matrices, median of 3 runs
  alternated with the pre-split binary; CPU busy with another job, GPU idle):**

  | Variant (slices, lookahead) | qkv | attn_out | fc | fc_out | lm_head | Step |
  |---|---|---|---|---|---|---|
  | Before (serial sum, D48) | 3.32 | 1.83 | 3.97 | 7.65 | 5.94 | 22.70 |
  | 1, on | 3.17 | 1.80 | 4.21 | 7.35 | 10.16 | 26.69 |
  | 1, off | 4.91 | 4.49 | 6.09 | 18.26 | 5.20 | 38.94 |
  | 2, on | 3.69 | 1.35 | 4.96 | 4.86 | 9.83 | 24.70 |
  | 2, off | 3.61 | 3.01 | 5.12 | 9.28 | 6.60 | 27.62 |
  | 4, on (D52) | 3.84 | 1.23 | 4.99 | 5.27 | 10.16 | 25.48 |
  | 4, off | 2.81 | 1.76 | 4.76 | 5.72 | 8.08 | 23.13 |

  Single runs vary by up to ~15% per matrix (the read roof itself measured 28.7–30.3 GB/s), so
  differences under that are noise. The rule's estimate from this table: 3.17 + 1.23 + 4.21 +
  5.27 + 5.20 = 19.1 ms. The 1024 boundary sits between the measured 768 and 2304; it isn't
  itself measured.
- **Mutation pass (20 mutants in matvec.wgsl, matmul.wgsl and the dispatch in ops.rs; `--test
  gpu_ops --test gpt2_gpu`): 17 caught.** Both barrier removals, a wrong chunk size in either
  kernel, a reversed combine order, a swapped lookahead pair, a dropped final fold, and a
  workgroup count that doesn't match the slice count were all caught. The 3 survivors:
  - Staging `x` with `<=` n_in: one extra value is staged and never read, and the read itself
    is out of bounds, which WebGPU clamps. Equivalent; the check stays (no out-of-bounds reads).
  - A partial last chunk's length taken from the round start instead of the chunk start: slices
    past n_in read `w` out of bounds, this driver returns 0, and `fma(x, 0, p) = p`. Another
    driver may return a clamped in-bounds value instead, so the check stays (like D50's row
    check). Not observable here.
  - Adding the empty chunks of a partial last round: each adds +0, and the total is never -0.
    Equivalent, as the shader comment says.
- **Model level (2026-10-07):** the pre-split binary (`06f483c`) and D56 (`5a59bc7`), run
  alternately twice: `ember profile`, `ember bench -n 32`, `ember matmul` on the 7-token prompt.
  The GPU was idle but the CPU was not: another project's test loop ran the whole time (load
  average 5–12). GPU times come from timestamp queries and hold up; wall-clock times don't.

  | | Before (2 runs) | After (2 runs) | |
  |---|---|---|---|
  | Decode step, kernels (`profile`) | 24.6, 26.7 ms | 22.1, 22.4 ms | −10 to −16% |
  | Block matrices per decode step (`profile`) | 17.5 ms (`matvec` ×48) | 15.1 ms (`matvec` 7.9 + `matvec_split` 7.2) | |
  | One step's matrices (`matmul`) | 23.3, 23.7 ms | 19.6, 21.1 ms | |
  | Decode weight bandwidth, kernels | 20.1, 18.5 GB/s | 22.4, 22.1 GB/s | |
  | Prefill, kernels | 53.9 ms | 53.9 ms | unchanged (the matmul's extra fold costs nothing visible) |
  | Decode, `bench` (wall, CPU loaded) | 24.1, 30.8 tokens/s | 37.4, 35.6 tokens/s | not a valid comparison |

  The gain is smaller than D55's estimate (attn_out + fc_out 9.6 → 5.5 ms, step ~24 ms):
  the split matrices went from ~9.5 to ~6.1 ms in `ember matmul`, but less inside the model.
  The wall-clock numbers were rerun on a quiet machine on 2026-10-08: see the table at the end
  of D57.

### D57: Up to 8 rows run a multi-row matvec
- **What:** `matvec_rows.wgsl` is the matvec with up to 8 rows per thread: one partial per row
  in two vec4s, each weight read once and applied to every row. Same layout, slices, lookahead
  and dispatch rule as D56 (`matvec_rows_split` / `matvec_rows` / `matvec_rows_wide`).
  `ops::linear`: T = 1 matvec, 2–8 `matvec_rows`, more the tiled matmul. x is staged in windows
  of 64 k values per chunk (a whole round for 8 rows would be 32 KiB of shared memory).
- **Alternatives (owner's choice, 2026-10-07):** split-K in the tiled matmul for small T (a
  scratch buffer and a second pass); a narrower tile.
- **Why:** a 7-row prefill is memory-bound like decode: 7 rows × 2 flops per weight is far below
  what the GPU can compute while one weight arrives. The tiled matmul ran it on `n_out / 64`
  workgroups (12 for 768 outputs). Each row is still D51's chunked sum, so bits don't depend
  on the batch (tested on sub-batches of 1, 2 and 8 rows against a 70-row matmul batch).
- **Measured (`ember profile`, 7-token prompt, GPU time, D56 binary and D57 alternately twice;
  CPU loaded by another job):** the prefill's block matrices 46.5, 46.3 ms (`matmul` ×48) →
  22.2, 26.9 ms (`matvec_rows` + `matvec_rows_split`); all prefill kernels 53.6, 53.0 → 28.5,
  37.0 ms. The second run was noisier (its wall clock had 17.8 ms outside kernels against
  ~8 normally). `ember matmul`'s T = 7 row now times `matvec_rows` too, but it repeats one
  matrix, which stays in the last-level cache (D50), so the model-level number is the one to
  quote.
- **The 8-row limit is not measured:** at some T the tiled matmul wins again (compute starts to
  matter, and 8 accumulators per thread is already 2 vec4s). Not swept.
- **Mutation pass (19 mutants in matvec_rows.wgsl and the dispatch; `--test gpu_ops --test
  gpt2_gpu`): 15 caught**, including all three barrier removals, a wrong row / k mapping in the
  staging, a swapped lookahead weight, a wrong half of the accumulators, and a row limit raised
  to 16 (caught by the GPT-2 tests, whose uncached recompute runs 9–16-row batches). The 4
  survivors:
  - Writing row t (one past the end): this driver drops out-of-bounds writes; the check stays
    (D50).
  - Staging rows past t: read out of bounds, never stored. Equivalent; the check stays.
  - No window bound inside a partial last chunk: the extra k values have x staged as 0, and
    `fma(0, w, p) = p` for any finite w. Equivalent; the bound only skips work (like D46).
  - Adding the empty chunks of a partial last round: +0 each. Equivalent (D56).
- **Clean wall-clock rerun (2026-10-08, D56 and D57 together):** the pre-split binary
  (`06f483c`), D56 (`5a59bc7`) and D57 (`e350141`), each running `ember profile` then `ember
  bench -n 32` on the 7-token prompt, in that order, for 16 rounds over two sessions (6 before, 5
  D56, 5 D57 with a bench line). No other job used the GPU or the CPU (load average 1.0–2.1 from
  the runs themselves; one last round at 3.3). Each number is the median over rounds of each
  binary's own median of 5 after 1 warm-up; the range across rounds is in brackets.

  | | Before | D56 | D57 |
  |---|---|---|---|
  | Decode, `bench` | 34.1 tokens/s [32.5–36.4] | **37.6 tokens/s** [35.5–39.1] | 37.8 tokens/s [31.7–38.5] |
  | Decode step, kernels (`profile`) | 25.0 ms [23.3–26.4] | 20.5 ms [19.6–22.3] | 20.9 ms [20.5–26.5] |
  | Decode step, wall (`profile`) | 30.6 ms | 25.8 ms | 26.7 ms |
  | Prefill (7 tokens), `bench` | 58.3 ms | 57.3 ms | **34.0 ms** [32.9–36.4] |
  | Prefill, kernels (`profile`) | 54.0 ms | 51.1 ms | 28.2 ms |

  - Decode: −18% kernel time, +10% tokens/s (D56). D57 doesn't touch the T = 1 path, so its
    decode column measures the same code as D56's; their difference (20.5 vs 20.9 ms) shows the
    noise between rounds. Pooled over both, decode is 37.7 tokens/s.
  - Prefill: D57 is 1.7× faster in wall clock (58.3 → 34.0 ms), 1.9× in kernel time. D56's
    3 ms prefill drop is within the range of the before column, so it isn't claimed.
  - About 5–6 ms of every decode step is still outside the kernels (M7).
  - The order was fixed (before, D56, D57), so D57 always ran last in a round, after two
    binaries had warmed the GPU. Its worst round (31.7 tokens/s, kernels 26.5 ms) was the last
    of the run, at load 3.3. A rotating order would remove that bias.
  - The copy kernel's bandwidth varied from 18.8 to 27.4 GB/s across rounds, so the
    "% of the roofline" lines of `ember profile` aren't quoted from this run (one round shows
    134%, which is impossible: its copy measurement was low).


## M7: Fusion and memory (approved 2026-10-08)

Starting point (D57's clean rerun): a decode step is 135 dispatches, each with its own output
buffer, uniform buffer, bind group, command encoder and submit; ~5.3 ms of each ~25.8 ms step is
outside the kernels. Attention runs 12 workgroups in decode and takes 10.8 ms of the step at
position 1000. The plan's estimates (measured in §4: D66): ~1.5 ms outside kernels, decode at
position 8 ~21 ms (~47 tokens/s), attention at position 1000 ~3–4 ms.

### D58: One command encoder and one submit per token
- **What:** ops record into a `Rec` (one `wgpu::CommandEncoder`) instead of submitting. A token's
  ~110 dispatches, the copy of the last row and the copy into the readback buffer go into one
  encoder and one `queue.submit`. Each dispatch keeps its own compute pass, so the profiler's
  per-pass timestamps work unchanged.
- **Alternatives:** one pass for the whole token (fewer passes, but per-kernel timing then needs
  `TIMESTAMP_QUERY_INSIDE_PASSES`); keep one submit per op.
- **Why:** a submit costs tens of microseconds of driver work; 135 of them are a large part of
  the 5.3 ms. Within one encoder wgpu still inserts the barriers between dependent dispatches.

### D59: The cached path runs on a fixed workspace
- **What:** a `Workspace` per model holds one buffer per intermediate (residual stream ×2,
  LayerNorm output, qkv, attention output and its chunk partials, MLP hidden, last row, logits,
  readback), each sized for `n_ctx` rows and allocated once. Tensors over it are views: shape
  `[T, …]`, buffer larger (kernels bound by their params, never the buffer size, D9). Bind groups
  and uniform buffers are cached by (kernel, bound buffers); a token only rewrites the uniforms
  whose contents changed (the position) and the token id. Reusing one cache entry with different
  contents within one recording is an error, because `write_buffer` lands before the whole
  submit.
- **Alternatives:** a size-keyed pool of free buffers (keeps the API, rebuilds bind groups every
  token); trace a step once and replay it (wgpu can't replay compute command buffers).
- **Why:** bind groups name buffers, so fixed buffers make every bind group reusable. `n_ctx`
  sizing lets prefill use the same buffers as decode.

### D60: "Zero allocations" means zero GPU object creations per token
- **What:** `Gpu` counts the buffers and bind groups it creates. After the first decode step,
  a step creates none (tested). Rust heap allocations and wgpu's internal staging memory (wgpu
  allocates a staging chunk per `write_buffer` on native) are reported, not required to be zero.
- **Alternatives:** zero heap allocations too (needs a counting allocator, and wgpu allocates
  while recording).

### D61: Ops record into a caller's `Rec`
- **What:** each op has a form that takes a `Rec` and its output tensor. The existing functions
  (`ops::linear(gpu, …)` etc.) become wrappers: allocate the output, record, submit. So per-kernel
  tests, the uncached M3 path and the cached workspace path all run the same recording code.
- **Alternatives:** a second implementation for the workspace path.

### D62: The linear kernels end with an epilogue
- **What:** an override constant `EPILOGUE` selects what happens after the bias: nothing, GELU
  (the text of `gelu.wgsl`'s function, shared), or adding a residual input. `matmul`, `matvec`
  and `matvec_rows` apply it the same way, so decode stays bitwise equal to recompute (D33).
  `fc` uses GELU; `attn_out` and `fc_out` add the residual (f32 addition commutes, so
  `x + y` is the same bits as the separate `add` kernel's). 135 → 99 dispatches before D63.
- **Alternatives:** fuse LayerNorm too (harder, small gain); no fusion.
- **Built (2026-10-08):** `epilogue.wgsl` (with `gelu_fn.wgsl`, now also prepended to
  `gelu.wgsl`) is prepended to the three kernels; `res` is binding 4, the params moved to 5.
  Kernels without a residual bind `x` there (as a missing bias binds `w`). Each linear kernel
  is compiled 3 times (21 pipelines instead of 7): `ember info` 47 → 59 ms (3 runs each, wall
  clock; Mesa may cache compiled shaders across runs). Tested bitwise against `gelu` and `add`
  run separately for all 7 kernel configurations, and through the whole model (tiny and GPT-2
  124M) against the one-op path: equal on this driver. The profiler reports the fused kernels
  as `matvec+gelu`, `matvec_split+res` and so on.

### D63: Attention splits keys into chunks of 64 with an online softmax
- **What:** pass 1, one workgroup per (query row, head, key chunk of 64): stage the chunk's keys
  in shared memory with coalesced loads, score them, and write the chunk's max, sum of
  exponentials and unnormalized output. Pass 2, one workgroup per (row, head): combine the chunks
  in chunk order (rescale by `exp(m_c - m)`), then divide by the total. Chunks are fixed by
  absolute key index, so a row's bits don't depend on the batch, and D33 still holds. The
  1024-position shared-memory limit of D20 goes away.
- **Alternatives:** one pass where the last workgroup combines (an atomic counter); full
  FlashAttention-2 with tiles of queries for long prompts (prefill is short today; deferred).
- **Why:** decode attention runs 12 workgroups today; at position 1000 this is 192.
- **Built (2026-10-08):** `attention.wgsl` (pass 1, `reduce.wgsl` prepended) and
  `attention_combine.wgsl` (pass 2). Pass 1 stages 32 keys at a time with rows padded to 65
  floats (8.1 KiB of keys, 8.9 KiB of workgroup memory in all, under WebGPU's default 16 KiB;
  all 64 keys would be 16.6 KiB). That padding caps the head dimension at 64
  (`ATTENTION_MAX_D`, checked on the host; GPT-2, SmolLM2 and Qwen2.5 all use 64). Partials are
  `[T, heads, n_chunks, d + 2]` with `n_chunks = ceil((start + T) / 64)`; the workspace's
  `Role::Parts` is sized for a full prefill: 1024 × 12 × 16 × 66 floats = 52 MB for GPT-2
  124M (workspace 35 → 87 MB). `ATTENTION_MAX_CTX` is gone; tested to 1300 positions. Tests:
  T = 63, 64, 65, 1023, 1024, 1025, 1300 against the CPU at the unchanged tolerance, a dominant
  key in chunk 2 (the combine must rescale to the global max), the d > 64 and short-scratch
  rejections, and the existing bitwise decode-vs-recompute tests.
- **Fix (§4):** the first version staged both 32-key halves for every chunk, with an integer
  divide and modulo per element, even when the row had only a few keys there. A 7-token prefill's
  attention took 0.79 ms against the old kernel's 0.31 (12 layers, timestamp queries), and the
  uncached path, which recomputes short sequences, slowed down. Now only the halves that hold
  keys are staged, one key row per step (thread l loads dim l, d <= 64 = WG). Same arithmetic,
  same bits.
- **The no-cache "regression" was the benchmark (§4, approved 2026-10-08):** `ember bench`'s
  uncached rate went 64 → 84 ms/token from §3 on, but the uncached path's kernel time did not
  change (51–56 ms per call at T = 7..40, s2 and HEAD alike), and `generate --no-cache` was not
  slower. `bench` kept its `KvCache` (KV rows + workspace, 155 MiB, 52 MB more than before
  because of `Role::Parts`) alive while timing the uncached loop. wgpu 30 sub-allocates with
  gpu-allocator in 128–256 MiB blocks and frees a spare block as soon as it empties. With the
  cache allocated, one uncached call's temporaries stop fitting the existing blocks at T = 20:
  each call then creates and frees a 256 MiB block (allocator report polled during the calls:
  5 blocks at T = 19, a 6th at T = 20; 68 → 101 ms per call), about +60 ms per call over a
  T = 7..40 sweep. Fix: `bench` drops the cache before the uncached run, so it measures the M3
  path as `generate --no-cache` runs it. Paired runs after the fix (alternating, on battery, so
  slower than the §4 table): s2 66.7 / 67.5 / 68.0 vs HEAD 70.5 / 71.7 / 72.1 ms/token. The
  remaining +4 ms (6%) is partly attention on short prefills: at T = 40, 12 layers of attention
  + combine take 3.6–3.7 ms vs 1.9–2.7 ms for the old one-pass kernel (timestamp queries, 2
  paired runs), so +1–1.8 ms; the rest is within the run-to-run spread of the kernel totals
  (57–63 ms) and not explained further. Not fixed: the
  uncached path is the M3 reference, not a serving path. A long-running engine that allocates
  per call would hit the same block churn; the cached path allocates nothing per token (D60).

### D64: M7 tests
- **What:** outputs into oversized buffers pre-filled with sentinels (D9): the elements past the
  view must survive. Workspace path vs op-by-op path bitwise. Attention at chunk boundaries
  (63, 64, 65, 1023, 1024 keys) and beyond 1024 against the CPU at the unchanged 1e-5. The
  creation counter is zero for a steady-state decode step.

### D65: `Gpu::alloc` creates buffers initialized (approved 2026-10-08)
- **Problem:** with several threads on one `Gpu` (the test binaries), a kernel's output buffer
  sometimes held another buffer's data: `gpu_ops` failed 4 of 20 runs, each time in a different
  test, with an output whose first few floats were right. Present since at least M4 (a stress
  test fails at `727daab`); §3's three new tests added enough parallel traffic to show it.
  Narrowed down with `tests/concurrency.rs` (8 threads × 300 `add`s): 1–5 bad outputs per run;
  none on one thread (0 of 7200), none for upload + readback alone, none when the output was
  initialized before the kernel. So the race is in wgpu 30.0.1's lazy zero-init of new buffers
  (or below it, Mesa ANV), not in ember's recording or readback. The engine itself runs on one
  thread, which never showed it.
- **What:** `alloc` creates its buffer `mapped_at_creation` and unmaps it at once: the zeros
  are written then, and wgpu has nothing to initialize lazily. Staging buffers keep lazy init
  (readback alone never failed).
- **Alternatives:** run GPU tests on one thread (hides a real race); report upstream only.
- **Result:** the stress test 0 bad in 6 runs, `gpu_ops` 0 failures in 20 runs. Creation counts
  (D60) are unchanged; the cached decode step allocates nothing, so it can't be slower. Worth
  an upstream report with the stress test as the repro.

### D66: M7 results (measured 2026-10-08)
- **Method:** four binaries, `before` (`e452e1a`, M6 + split-K), s2 (`09b2117`, §1 + §2), s3
  (`194e416`, §3) and s4 (`c4e21b6`, §3 with the staging fix), each running `ember profile` and
  `ember bench -n 32` on the 7-token prompt, 4 rounds in rotating order; medians of the 4
  rounds (each round is itself a median of 5 after a warm-up). The machine was under load
  (load average 10 → 2.5, another project's fuzzer), so differences under ~5% are noise.
  Raw output: `target/tmp/m7/bench2.out` (not committed).

| median of 4 rounds | before | s2 | s3 | s4 |
|---|---|---|---|---|
| decode tokens/s | 36.8 | 42.4 | 41.2 | 42.0 |
| 7-token prefill ms | 34.9 | 31.2 | 31.5 | 30.8 |
| decode: outside kernels ms | 5.43 | 4.24 | 4.84 | 4.64 |
| decode at position 1000: attention ms | 11.2 | 10.5 | 5.7 | 5.5 |
| decode at position 1000: wall ms | 37.8 | 34.3 | 29.3 | 29.0 |
| 7-token prefill: attention ms (12 layers) | 0.32 | 0.30 | 0.82 | 0.49 |

- **Decode:** 36.8 → 42.0 tokens/s (+14%), mostly from §1 + §2 (one submit per token, the
  fixed workspace, 135 → 99 dispatches). §3 adds 12 dispatches (the combine, 111 in all) and
  does not speed up a decode step at position 8, where attention is 2% of the step.
- **Long context:** at position 1000 attention takes 11.2 → 5.5 ms and the step 37.8 → 29.0 ms
  (−23%): 192 workgroups instead of 12. Attention is then 21% of the GPU time (35% before).
- **Where a decode step goes now (s4, 4 rounds):** the linear kernels are 71–82% of the wall
  time and 91–97% of the GPU time. They stream the 495 MB of weights at 24.5–25.5 GB/s: above
  the copy kernel's 23.7–24.4 GB/s measured in the same runs (a copy reads and writes the same
  number of bytes; decode mostly reads), and about 83–86% of the 29.6 GB/s read-only peak
  measured in M6 (a different session, so approximate).
- **No-cache path:** the 64 → 84 ms/token in the first measurement was the benchmark holding
  the KV cache allocated (D63, fixed in `bench`). Paired after the fix: +4 ms (6%) over s2.
- **Against the plan (stated before building):**
  - Time outside the kernels: planned ~1.5 ms, measured 4.6 ms. §1 + §2 together saved
    ~1.2 ms, not ~4. Not broken down further; the remaining parts (the 201 KB logits readback
    and its map/poll wait, recording 111 passes, the host argmax) have not been timed
    separately, so which one dominates is open.
  - Decode at position 8: planned ~21 ms (~47 tokens/s), measured 23.8 ms (42.0 tokens/s); the
    kernels came in as planned, the miss is the outside time above.
  - Attention at position 1000: planned 3–4 ms, measured 5.5 ms. Not profiled inside the
    kernel; candidates are the values read from global memory (not staged) and the second pass's
    launch. Unverified.
  - Short prefills got slower in attention (0.30 → 0.49 ms over 12 layers; 0.82 before the
    staging fix): a 7-token row still runs a 64-thread workgroup per chunk and a second pass.
    0.2 ms of a 31 ms prefill.

## M8: A Llama-family model (approved 2026-10-08)

SmolLM2-135M (Hugging Face `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f`): Llama
architecture, E = 576, 30 layers, 9 query heads and 3 key/value heads of d = 64, SwiGLU MLP of
1536, vocabulary 49152, RoPE θ = 100000, RMSNorm ε = 1e-5, tied embeddings, bf16 weights.

### D67: SmolLM2-135M first, 360M as a scale check, Qwen2.5 deferred
- **What:** build for SmolLM2-135M; then run SmolLM2-360M (E = 960, 32 layers, 15/5 heads) by
  config alone, with no code change.
- **Alternatives:** Qwen2.5-0.5B: a 151936-token vocabulary (545 MB for the embedding alone in
  f32, ~2 GB in all), biases on q/k/v, and a different pre-tokenizer pattern.
- **Why:** 135M is GPT-2's size class, so the numbers compare directly, and it is the plain
  Llama layout. Qwen fits better after M9 makes weights smaller.

### D68: bf16 weights become f32 on load
- **What:** the loader converts bf16 to f32 (exact: the bf16 bits are the top half of the f32).
  Kernels stay f32.
- **Alternatives:** bf16 on the GPU, unpacked in the matmul (half the bytes per token: that is
  M9's job).

### D69: The oracle chain for Llama
- **What:** a plain-Rust f32 CPU reference (`src/llama/`), checked against a numpy float64
  implementation (`scripts/llama_golden.py`, same approach as D12), and token ids checked
  against the `tokenizers` library. No PyTorch (PyPI is too slow on this network).
- **Alternatives:** transformers + torch goldens; llama.cpp's output (comes with M11).
- **Built (§2):** `scripts/llama_golden.py` (numpy float64, vectorized; RoPE angles in f64)
  writes a committed tiny model (`tests/fixtures/tiny_llama`: 2 layers, 6 query heads over 2
  key/value heads, d = 4, bf16 weights; the seed search requires a varied greedy continuation,
  since a random model with tied embeddings keeps predicting its last token) and goldens for
  SmolLM2-135M. The numpy model continues "I enjoy walking with my cute dog" with ", and I
  love to watch him play.": fluent text, which a wrong RoPE pairing or head grouping would
  not give. The Rust CPU reference matches the tiny model at 1.3e-6 and SmolLM2's greedy
  continuations exactly (4 prompts, 52 tokens; closest top-2 gap 0.024).
- **Tolerance:** SmolLM2's logits differ from float64 by up to 4.8e-4 (GPT-2: 6.4e-4 on logits
  5x larger). An independent float32 numpy forward is off by the same amount (up to 3.9e-4),
  so it is f32 rounding over 30 layers. The test allows 2e-3 absolute (4x).

### D70: RoPE pairs dimension i with i + d/2, from one shared table
- **What:** Hugging Face's Llama layout ("rotate half"): for i < d/2,
  `x'[i] = x[i] cos - x[i + d/2] sin`, `x'[i + d/2] = x[i + d/2] cos + x[i] sin`, angle
  `pos * θ^(-2i/d)`. A `[n_ctx, d/2]` cos table and sin table are computed once on the CPU in
  f64, rounded to f32, and used by both the CPU reference and the GPU kernel. Q and K are
  rotated before K goes into the cache.
- **Alternatives:** sin/cos inside the shader (WGSL's sin and cos have loose accuracy
  guarantees, worse for large arguments, and differ from the CPU's); interleaved pairs (GPT-J
  layout: wrong for these weights).

### D71: Grouped-query attention in the existing kernel
- **What:** query head h reads key/value head `h / (n_head / n_kv_head)`. The cache holds only
  the key/value heads: `[n_ctx, n_kv_head * d]`. GPT-2 is the case `n_kv_head = n_head`, and
  its outputs must stay bit-for-bit the same (tested).
- **Alternatives:** a second attention kernel; expanding K and V to all heads (3× the cache and
  the reads).
- **Built (§3):** `ops::Heads { q, kv }` goes to `kv_write_into` and `attention_into`; the
  kernels take the cache row width and the group size where they had padding. GPT-2's GPU
  output is bit-identical before and after (a 70-token forward plus 9 cached decode steps,
  3.97 M floats compared with `cmp`). Tested against the CPU for 9/3, 6/2, 4/1 and 4/4 heads
  across chunk boundaries (worst 7.8e-7 absolute), and bitwise for a 50-row prefill followed
  by 20 decode steps against one 70-row pass.

### D72: Fused projection matrices
- **What:** at load, q, k and v become one `[E, (n_head + 2 n_kv_head) d]` matrix and gate
  and up one `[E, 2 I]` matrix, so each is one matmul, as GPT-2's `c_attn`.

### D73: RMSNorm and SiLU-multiply kernels
- **What:** `rms_norm.wgsl` (the fixed reduction tree of `reduce.wgsl`, sum of squares, gain,
  no bias) and `silu_mul.wgsl` (`silu(gate) * up` over the fused `[T, 2I]` buffer). The down
  projection adds the residual in its epilogue (D62).
- **Alternatives:** SwiGLU as a matmul epilogue (each thread would need its gate and up
  columns together); deferred until a profile says it matters.
- **Built (§3):** `rms_norm.wgsl` (with `reduce.wgsl`), `silu_mul.wgsl`, and `rope.wgsl`, which
  rotates in place (one invocation per pair, reading both halves before writing either), so
  the workspace needs no second qkv buffer. Worst errors against the CPU: RMSNorm 4.8e-7
  (CPU in f64), SiLU-multiply 8.7e-7 relative (exp differs by a few ulp), RoPE 2.4e-7 (a
  driver may fuse `a c - b s` into an fma). Value heads pass through RoPE bit for bit.

### D74: `src/llama/` beside `src/gpt2/`, shared ops and cache
- **What:** config, weights, CPU forward and GPU forward per model; the KV cache and workspace
  take their sizes from the model; the CLI gets `--model gpt2 | smollm2-135m`.
- **Alternatives:** one generic model abstraction over both (more indirection than two models
  justify).

### D75: The tokenizer learns digits and special tokens
- **What:** SmolLM2 uses GPT-2's byte-level BPE with two additions: digits are split one by
  one before the byte-level step, and added tokens (`<|endoftext|>`, `<|im_start|>`, …) are
  matched whole before BPE. Both are read from `tokenizer.json`; GPT-2's ids don't change.
- **Built (§1):** `Tokenizer::load_hf` reads `tokenizer.json` and refuses every setting it
  doesn't implement (a normalizer, `add_prefix_space`, an unknown token, byte fallback,
  `ignore_merges`, subword prefixes, non-special or stripping added tokens). Found on the
  way: SmolLM2's vocabulary has no token for 21 bytes (six control characters and bytes valid
  UTF-8 almost never uses: 0xC0, 0xC1, 0xF1, 0xF2, 0xF5–0xFF). Hugging Face, with no unknown
  token, drops such a byte before merging, so ember does too; `load` (GPT-2) still requires
  all 256. Special tokens: earliest match first, the longest at one position. Matches
  `tokenizers` 0.23.2 on 31 strings (GPT-2's 23 plus digits, non-ASCII digits and special
  tokens next to text).

### D76: The KV cache is sized for 2048 positions by default
- **What:** `n_ctx` for the cache and workspace defaults to 2048 (configurable), below the
  model's 8192: at 8192 the cache alone would be 30 × 2 × 8192 × 192 × 4 B = 377 MB.
