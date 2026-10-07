# ember

An LLM inference engine on WebGPU, written in Rust with [wgpu](https://wgpu.rs) and WGSL compute
shaders. It runs natively on Vulkan (developed on an Intel Iris Xe iGPU), and the plan is to run
in the browser via wasm too.

**Status:** M0–M6 done. GPT-2 124M runs end to end on the GPU (Intel Iris Xe, Vulkan) with a
KV cache, at **35.8 tokens/s decode** and 56 ms prefill for a 7-token prompt (`ember bench`,
median of 5 after a warm-up, wall clock). Logits are within 2.5e-6 (relative) of a plain-Rust CPU
reference, and greedy text is identical. Every cached decode step is bit-for-bit equal to
recomputing the whole sequence. The reference itself matches a float64 numpy implementation and
Hugging Face's published output. Every kernel is differential-tested against the reference.

M6 replaced the naive matmul with a shared-memory tiled kernel for prefill (up to 466 GFLOP/s,
39× naive, against a measured 1431 GFLOP/s compute roof) and a coalesced matrix-vector kernel
for decode (73% of the measured 29.6 GB/s read bandwidth over a step's weights). Against the
pre-M6 build, run back to back: decode 25.3 → 35.8 tokens/s, prefill 129 → 58 ms (`ember
profile`, `ember bench`, `ember matmul`; DESIGN.md D50). Next: M7, one command encoder per
token and reused buffers (~5 ms of each 28 ms decode step is outside the kernels).

## Approach
- **CPU oracle.** A plain-Rust reference implementation of every op (and of the whole model) is
  the source of truth. Every GPU kernel is differential-tested against it with explicit tolerances.
- **Deterministic.** Same device and inputs give bit-identical outputs: no float atomics, fixed
  reduction order, seeded sampling.
- **Measured.** Every performance claim states its method: wall clock (median of 5 after a
  warm-up) per step, and GPU timestamp queries per kernel.

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
