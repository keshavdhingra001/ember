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

Today (M1): the CPU reference GPT-2 end to end (safetensors loader, BPE tokenizer, forward
pass, greedy generation, `ember generate`), matching a float64 numpy implementation. The
GPU side is still M0's: a headless device, upload and readback, and one `add` kernel.

## Decisions

### D1: Stack: Rust + wgpu + WGSL (owner chose 2026-10-04)
- **Alternatives:** TypeScript on the browser WebGPU API; C++ on Dawn (Chrome's WebGPU).
- **Why:** wgpu is the WebGPU implementation Firefox uses. It runs natively on Vulkan / Metal / DX12
  and compiles to wasm, so the same kernels run on the laptop and in a browser (M10). GPU code is
  testable headless with `cargo test`, which a browser stack makes hard on Linux. Same toolchain
  and workflow as lsmkv and lob. Trade-off: WebGPU lacks CUDA features (no tensor cores, limited
  subgroup ops, no float atomics), and that is the interesting constraint to talk about.
- **Hardware:** Intel Iris Xe (TigerLake GT2, integrated, shares system RAM) via Mesa's Vulkan driver.

### D2: Model path: GPT-2 124M, then a small Llama-family model (owner chose 2026-10-04)
- **Alternatives:** the owner's FaceRestore U-Net (convolutions); jumping straight to Llama.
- **Why:** GPT-2 is the simplest real transformer (learned positions, LayerNorm, GELU, MHA), with
  public weights and well-known outputs, so the first milestones are about correctness, not
  architecture. The Llama family (M8) then adds what interviews ask about now: RoPE, RMSNorm,
  SwiGLU, grouped-query attention, and quantized weights (M9). The U-Net is a Tier 3 stretch.

### D3: The CPU reference is the oracle (Claude, M0; owner review pending)
- **What:** a plain, obviously correct Rust implementation of every op (and in M1 of the whole
  GPT-2 forward pass). Every GPU kernel is differential-tested against it on edge-case and random
  shapes. The reference is checked once against pinned outputs of the original model.
- **Alternatives:** compare GPU outputs only against goldens dumped from PyTorch.
- **Why:** goldens only cover the inputs they were dumped for; an oracle covers any shape and any
  input, and points to the exact kernel that broke. Same idea as lob's reference book.

### D4: Determinism (Claude, M0; owner review pending)
- **What:** same device + same inputs gives bit-identical outputs. Kernels never use atomics on
  floats, reductions have a fixed order (fixed workgroup size, fixed tree), sampling takes a seed,
  and engine code never reads a clock (only the measurement module does).
- **Why:** float addition is not associative, so a reduction whose order depends on scheduling
  gives different bits run to run. Then a test failure can't be reproduced, and "did my change
  alter the output?" has no answer. Bit-identical across *different* GPUs is not promised (D5).

### D5: Compare floats with explicit tolerances (Claude, M0; owner review pending)
- **What:** `compare::check(got, want, Tol { abs, rel })` passes when every element satisfies
  `|got - want| <= abs + rel * |want|`, and NaN or infinity mismatches always fail. On failure it
  reports the worst index, both values and the error.
- **Alternatives:** bitwise equality; mean error only.
- **Why:** CPU and GPU legitimately differ: the GPU may fuse multiply-add, sums run in a different
  order, and WGSL allows `exp`/`tanh` within a few ULP. Bitwise equality would fail on correct
  kernels; a mean would hide one broken element. Tolerances are set per op (exact ops such as `add`
  use 0) and any loosening gets a note here.

### D6: Device setup (Claude, M0; owner review pending)
- **What:** one headless device (no window surface), high-performance adapter preference,
  requesting the adapter's own limits instead of WebGPU defaults. Blocking API on native via
  `pollster`; the async core stays reachable for wasm later (M10).
- **Why limits:** the WebGPU default `max_storage_buffer_binding_size` is 128 MiB, but GPT-2's token
  embedding is 50257 × 768 × 4 B = 147 MiB. Asking for what the adapter supports avoids splitting
  that tensor. `ember info` prints the limits we got.

### D7: Tensor layout (Claude, M0; owner review pending)
- **What:** contiguous row-major `f32`, shape kept on the host (`Vec<usize>`). A `GpuTensor` is a
  storage buffer plus its shape. No strides or views yet.
- **Why:** every kernel indexes `row * cols + col` and nothing else, which keeps the WGSL readable.
  Transposes happen once at load time instead of through strided reads. f16 and quantized layouts
  arrive with their milestones (M7, M9).

### D8: Elementwise dispatch with a grid-stride loop (Claude, M0; owner review pending)
- **What:** workgroup size 256; dispatch `min(ceil(n / 256), 65535)` workgroups; each invocation
  loops `i += 256 * num_workgroups`.
- **Alternatives:** one invocation per element with a 2D dispatch for large `n`.
- **Why:** the per-dimension workgroup limit is 65535, so one-thread-per-element tops out at
  16.7M elements, and GPT-2's embedding has 38.6M. The loop handles any `n` with a 1D dispatch.
  256 is a multiple of every vendor's SIMD width (Intel 8/16/32, AMD 32/64, NVIDIA 32).

### D9: Out-of-bounds accesses are contained, not trusted (Claude, M0; owner review pending)
- **What:** WebGPU guarantees that a shader can't read or write outside a bound buffer: wgpu adds
  bounds checks (or relies on Vulkan's robust buffer access), so an out-of-bounds write is dropped
  or clamped to the end of the buffer. Kernels still bounds-check by `params.n`, never by buffer size.
- **Why it matters:** in M0 the mutation `i < n` -> `i <= n` survives. Every output buffer holds
  exactly `n` floats, so the extra write lands outside the binding and is discarded: on this
  platform it's an equivalent mutant. Once buffers are reused from an arena (M7) and hold more
  than `n` floats, the same bug would silently corrupt the neighbour. M7's tests must use
  oversized buffers with sentinel values to catch it.

### D10: GPT-2 weights: safetensors, hand-parsed over mmap (owner approved 2026-10-05, M1)
- **What:** `openai-community/gpt2` `model.safetensors` (548 MB) plus `config.json`, `vocab.json`
  and `merges.txt` in `data/gpt2/` (git-ignored). A small hand-written parser reads the format
  (u64 little-endian header length, a JSON header mapping names to dtype, shape and byte offsets,
  then raw tensor bytes) over a `memmap2` mapping.
- **Alternatives:** the `safetensors` crate; converting to a custom format.
- **Why:** the format is simple enough to parse in about a page, and owning the parser means the
  owner can explain exactly how bytes become tensors. mmap avoids copying 548 MB before it is needed.
- **Details:** the JSON header itself is parsed with `serde_json` (JSON parsing is not the point).
  The reader validates everything the header claims before trusting it: header length within the
  file and under the format's 100 MB cap, known dtypes, `shape × dtype size == end - begin` with
  overflow checks, and tensors that tile the data section exactly (no gaps, overlaps, or trailing
  bytes). Values are decoded with `f32::from_le_bytes` because the data section starts at
  `8 + N` and has no alignment guarantee. Conv1D weights (`[in, out]`) are transposed to
  `[out, in]` at load (D7); the 12 `attn.bias` causal-mask buffers in the file are ignored.
  `scripts/fetch_gpt2.sh` pins the checkpoint's sha256
  (`248dfc39…d3a707`, same as Hugging Face's LFS hash).

### D11: Tokenizer: hand-written byte-level BPE (owner approved 2026-10-05, M1)
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

### D12: Pinned reference outputs from a numpy-only script (owner approved 2026-10-05, M1)
- **What:** a one-off Python script that uses only `numpy` (no torch, since PyPI is slow on this
  network) writes token ids and logits for 3 prompts to `data/gpt2/golden/`. The Rust reference
  must match the logits to 1e-4.
- **Alternatives:** hard-coding known greedy completions.
- **Why:** text alone can match while the logits are wrong. Matching logits checks every layer.
- **Details:** the script computes in float64 (the truth, as far as f32 is concerned) and
  stores float32. It is written vectorized (whole-sequence matmuls, all heads at once,
  `-inf` masking) so it shares little structure with the Rust loops. Two extra checks keep it
  from being "the same author's bug twice". Token ids come from Hugging Face `tokenizers`
  (installed in the venv next to numpy, a deviation from "numpy only"). And its greedy output
  must reproduce the continuation Hugging Face published for "I enjoy walking with my cute dog";
  the script refuses to write goldens otherwise.
- **Measured (2026-10-05):** GPT-2 124M logits on 3 prompts (6, 10 and 21 tokens): worst
  absolute error 6.4e-4 on logits of magnitude ~100, worst relative 1.0e-5. The tolerance is
  `abs 1e-4 + rel 1e-4 × |want|`, so the relative term carries it, with 10× headroom. "Match to
  1e-4" in the table therefore means relative: a 1e-4 absolute bound would fail on correct
  code, because f32 accumulates about 1e-6 relative error per layer over 12 layers. Greedy
  continuations match token for token; the smallest top-2 logit gap in them is 0.028, far above
  the error. Tiny model: worst absolute error 1.2e-6 at tolerance 1e-5.

### D13: Reference matmul is a naive triple loop (owner approved 2026-10-05, M1)
- **What:** row-major f32, weights transposed once at load (D7), no BLAS.
- **Alternatives:** `ndarray` or BLAS.
- **Why:** the oracle has to be obviously correct, and it's allowed to be slow.
- **Measured (2026-10-05, wall clock, `ember generate`, release build, one core):** 0.60
  tokens/s for a 7–23-token sequence with no KV cache. Each dot product is one serial f32
  dependency chain, which can't vectorize without reordering the sum (changing the bits). Unoptimized
  it is about 30× slower, so `[profile.test]` uses `opt-level = 3`. The GPT-2 greedy test
  (52 steps, ~100 s) is `#[ignore]`d and runs with `cargo test -- --ignored`.
- **Open (owner):** threading `linear` across output rows would give ~4–6× on this laptop and keep
  every element's bits identical (each output is still one serial sum). Not done yet; it needs
  the owner's call.

### D14: Model shape comes from config.json (owner approved 2026-10-05, M1)
- **What:** `n_layer`, `n_head`, `n_embd`, `n_ctx` and `vocab_size` are read into a config struct.
  No sizes are hard-coded.
- **Why:** hard-coded constants would break for gpt2-medium/large, and they hide shape bugs that
  the tiny test models (D15) would otherwise catch.

### D15: Tests run without the weights (owner approved 2026-10-05, M1)
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

### D16: Greedy decoding semantics (Claude, M1; owner review pending)
- **What:** `argmax` returns the first index among equal maxima, and `None` if any logit is NaN
  (generation then fails with an error). Generation stops early at the context length.
  `<|endoftext|>` doesn't stop it: the caller asks for `n` tokens and gets `n`.
- **Why:** first-max makes ties deterministic (D4) and matches `np.argmax`. NaN compares false
  with everything, so a plain max loop starting on a NaN returns index 0 forever. A broken
  model would then look like a model that likes token 0 (a test caught exactly that).
