# ember

An LLM inference engine on WebGPU, written in Rust with [wgpu](https://wgpu.rs) and hand-written
WGSL compute shaders. It runs GPT-2 124M natively on Vulkan (developed on an Intel Iris Xe
integrated GPU); a browser build via wasm is planned.

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

For prefill, M6's shared-memory tiled matmul reaches up to 466 GFLOP/s (39× the naive kernel)
against a measured compute roof of 1431 GFLOP/s (`ember matmul`).

Correctness: GPU logits are within 2.5e-6 (relative) of the CPU reference, and greedy text is
identical. The CPU reference matches a float64 numpy implementation and Hugging Face's
published output.

## How it works

- **CPU oracle** (`src/cpu.rs`, `src/gpt2/forward.rs`). Plain Rust, f32, no SIMD tricks: the
  source of truth. Each GPU op has a CPU twin and a test with an explicit, measured tolerance.
- **Kernels** (`src/shaders/`). Row reductions by a fixed tree (LayerNorm, softmax);
  a shared-memory tiled matmul with register blocks for prefill; a matrix-vector kernel for
  decode, splitting the reduction over K for narrow matrices; GELU and the residual add as the
  matmul's epilogue; attention split into chunks of 64 keys with an online softmax, merged in a
  fixed order.
- **KV cache and workspace** (`src/gpt2/gpu.rs`). Keys and values for every position; decode
  runs one token through the model. All intermediates live in fixed buffers allocated once, so
  bind groups are reused and a decode step creates no GPU objects. A token is one command
  encoder and one submit.
- **Determinism.** Same device and inputs give the same bits: no float atomics, fixed reduction
  orders, and chunk boundaries fixed by absolute position, so a row's result never depends on
  how the sequence was batched.
- **Measurement** (`src/profile.rs`). GPU timestamp queries per dispatch, measured bandwidth and
  compute roofs, wall clock per step.

The reasoning behind each choice, with the alternatives and the numbers, is in
[DESIGN.md](DESIGN.md) (D1–D66).

## Testing

`cargo test` runs 116 tests. Beyond the differential tests: decode vs full recompute
bitwise, attention at chunk boundaries and past 1024 positions, outputs written into oversized
buffers filled with sentinels (nothing past the view may change), zero GPU object creations per
decode step, and a multi-threaded stress test. Each milestone gets a mutation pass: bugs are
planted in the Rust and WGSL code one at a time to check that a test fails.

## Limitations and next steps

- GPT-2 only, f32 only. Next: a Llama-family model (RMSNorm, RoPE, SwiGLU, grouped-query
  attention), then int8 / 4-bit weights, then the browser build and a comparison with llama.cpp.
- Head dimension at most 64 (the attention kernel's shared-memory layout).
- Measured on one integrated GPU; the numbers are for that machine.
- Greedy decoding only.

## Try it

```bash
cargo run --release -- info       # which GPU and limits wgpu gave us
cargo run --release -- selftest   # GPU add on 1M floats vs the CPU reference
cargo test
scripts/fetch_gpt2.sh                                    # GPT-2 124M into data/gpt2/ (550 MB)
cargo run --release -- tokenize "Hello world"            # BPE, step by step
cargo run --release -- generate -n 16 "I enjoy walking with my cute dog"        # on the GPU
cargo run --release -- generate --cpu -n 16 "I enjoy walking with my cute dog"  # CPU reference
cargo run --release -- bench -n 32 "I enjoy walking with my cute dog"           # prefill / decode timings
cargo run --release -- profile "I enjoy walking with my cute dog"               # per-kernel GPU times
cargo run --release -- matmul                                                   # matmul kernels vs measured roofs
```
