# ember

An LLM inference engine on WebGPU, written in Rust with [wgpu](https://wgpu.rs) and hand-written
WGSL compute shaders. It runs GPT-2 124M and the Llama-family SmolLM2-135M / 360M natively on
Vulkan (developed on an Intel Iris Xe integrated GPU); a browser build via wasm is planned.

Every GPU kernel is differential-tested against a plain-Rust CPU reference, every cached decode
step is bit-for-bit equal to recomputing the whole sequence, and every performance number below
says how it was measured.

## Results

GPT-2 124M, f32, Intel Iris Xe (TGL GT2), Mesa 26.2.2, prompt "I enjoy walking with my cute
dog" (7 tokens). Wall clock, median of 5 runs after a warm-up (`ember bench`); milestones after
M5 were measured back to back against the previous build, in rotating order (the M7 row under
some background load, so ±5%).

| Milestone | What changed | Decode | 7-token prefill |
|---|---|---|---|
| M3 | Whole model on the GPU, no cache | 1.9 tokens/s | n/a |
| M4 | KV cache, incremental decode | 19.8 tokens/s | 135 ms |
| M6 | Tiled matmul for prefill, matrix-vector kernel for decode | 35.8 tokens/s | 58 ms |
| M6+ | Split-K matvec, multi-row matvec for short prompts | 37.6 tokens/s | 34 ms |
| M7 | One submit per token, fixed workspace, fused epilogues, chunked attention | 42.0 tokens/s | 31 ms |

Decode at position 1000 (attention reads the whole cache): 37.8 → 29.0 ms
per token with M7's chunked attention (attention itself 11.2 → 5.5 ms,
GPU timestamp queries).

Where the time goes (`ember profile`, timestamp queries per kernel): the weight matrices are
71–82% of a decode step's wall time. They stream the 495 MB of weights at about 25 GB/s, roughly
85% of the read bandwidth measured on the same GPU, so decode is memory-bound and close to the
hardware. The rest is attention, LayerNorm and about 4.6 ms per token outside the kernels
(driver, readback, CPU), which is where M7 fell short of its plan (DESIGN.md D66).

**SmolLM2-135M (M8)**, same kernels, same prompt (7 tokens), measured in one session against
GPT-2 in rotating order (3 rounds, on AC power, load under 2; DESIGN.md D77):

| | GPT-2 124M | SmolLM2-135M |
|---|---|---|
| Decode | 38.9 tokens/s | 30.1 tokens/s |
| 7-token prefill | 29 ms | 37 ms |
| Weights read per token | 495 MB | 538 MB |
| Kernel dispatches per token | 111 | 333 |
| Decode at position 1000 (attention) | 30.1 ms (5.0 ms) | 41.2 ms (9.2 ms) |

The GPU time follows the bytes (+7% for +9% weights; the block matrices stream at the
measured read roof), but SmolLM2's 30 layers record three times as many dispatches, and the
time outside the kernels grows from 3.9 to 9.3 ms per token: that, not the math, is most of
the gap. Grouped-query attention makes the cache 3× smaller, yet attention is not faster,
because each query head still reads its shared keys separately. Both are the next things to
fix (D77). SmolLM2-360M runs from its config with no code of its own. (GPT-2's 38.9 here vs
42.0 in the M7 row is mostly the session: the M7 binary gives 39.8 in the same session, and
M8 costs GPT-2 about 2%.)

For prefill, M6's shared-memory tiled matmul reaches up to 466 GFLOP/s (39× the naive kernel)
against a measured compute roof of 1431 GFLOP/s (`ember matmul`).

Correctness: GPU logits are within 2.5e-6 (relative) of the CPU reference, and greedy text is
identical. The CPU reference matches a float64 numpy implementation and Hugging Face's
published output. SmolLM2-135M: GPU logits within 4.2e-4 of the CPU reference (the size of
f32 rounding over 30 layers, measured with an independent float32 numpy run), and the greedy
continuations of 4 prompts match the float64 numpy model token for token.

## How it works

- **CPU oracle** (`src/cpu.rs`, `src/gpt2/forward.rs`, `src/llama/forward.rs`). Plain Rust,
  f32, no SIMD tricks: the source of truth. Each GPU op has a CPU twin and a test with an
  explicit, measured tolerance. Each CPU model is in turn checked against a float64 numpy
  implementation.
- **Two model families on the same kernels** (`src/gpt2/`, `src/llama/`). Llama adds RMSNorm,
  rotary position embeddings (in place, from a table computed once in f64), SwiGLU and
  grouped-query attention (query head h reads key/value head h / group, so SmolLM2's cache is
  3× smaller). GPT-2 is the group-1 case, and its output stayed bit-for-bit identical through
  the change. bf16 weights widen to f32 exactly on load; the tokenizer adds SmolLM2's digit
  split and special tokens, checked against Hugging Face `tokenizers`.
- **Kernels** (`src/shaders/`). Row reductions by a fixed tree (LayerNorm, softmax);
  a shared-memory tiled matmul with register blocks for prefill; a matrix-vector kernel for
  decode, splitting the reduction over K for narrow matrices; GELU and the residual add as the
  matmul's epilogue; attention split into chunks of 64 keys with an online softmax, merged in a
  fixed order.
- **KV cache and workspace** (`src/cache.rs`, shared by both models). Keys and values for every position; decode
  runs one token through the model. All intermediates live in fixed buffers allocated once, so
  bind groups are reused and a decode step creates no GPU objects. A token is one command
  encoder and one submit.
- **Determinism.** Same device and inputs give the same bits: no float atomics, fixed reduction
  orders, and chunk boundaries fixed by absolute position, so a row's result never depends on
  how the sequence was batched.
- **Measurement** (`src/profile.rs`). GPU timestamp queries per dispatch, measured bandwidth and
  compute roofs, wall clock per step.

The reasoning behind each choice, with the alternatives and the numbers, is in
[DESIGN.md](DESIGN.md) (D1–D77).

## Testing

`cargo test` runs 149 tests. Beyond the differential tests: decode vs full recompute
bitwise, attention at chunk boundaries and past 1024 positions, outputs written into oversized
buffers filled with sentinels (nothing past the view may change), zero GPU object creations per
decode step, and a multi-threaded stress test. Each milestone gets a mutation pass: bugs are
planted in the Rust and WGSL code one at a time to check that a test fails.

## Limitations and next steps

- f32 weights only. Next: int8 / 4-bit weights with dequantization inside the matmul, then the
  browser build and a comparison with llama.cpp.
- The KV cache holds 2048 positions by default for SmolLM2 (trained for 8192): attention's
  per-chunk partials grow with the square of the context, and a full 8192 would not fit in one
  buffer on this GPU (the cache refuses it with an error).
- Head dimension at most 64 (the attention kernel's shared-memory layout).
- Measured on one integrated GPU; the numbers are for that machine.
- Greedy decoding only.

## Try it

```bash
cargo run --release -- info       # which GPU and limits wgpu gave us
cargo run --release -- selftest   # GPU add on 1M floats vs the CPU reference
cargo test
scripts/fetch_gpt2.sh                                    # GPT-2 124M into data/gpt2/ (550 MB)
scripts/fetch_smollm2.sh                                 # SmolLM2-135M into data/smollm2-135m/ (272 MB; `360m` for the larger one)
cargo run --release -- tokenize "Hello world"            # BPE, step by step
cargo run --release -- generate -n 16 "I enjoy walking with my cute dog"        # on the GPU
cargo run --release -- generate --cpu -n 16 "I enjoy walking with my cute dog"  # CPU reference
cargo run --release -- bench -n 32 "I enjoy walking with my cute dog"           # prefill / decode timings
cargo run --release -- profile "I enjoy walking with my cute dog"               # per-kernel GPU times
cargo run --release -- matmul                                                   # matmul kernels vs measured roofs
cargo run --release -- generate --model smollm2-135m -n 16 "I enjoy walking with my cute dog"
cargo run --release -- tokenize --model smollm2-135m "<|im_start|>user 2024"    # special tokens, digit split
# every model command takes --model gpt2 (default) | smollm2-135m | smollm2-360m
```
