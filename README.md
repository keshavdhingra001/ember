# ember

An LLM inference engine on WebGPU, written in Rust with [wgpu](https://wgpu.rs) and WGSL compute
shaders. It runs natively on Vulkan (developed on an Intel Iris Xe iGPU), and the plan is to run
in the browser via wasm too.

**Status:** M3 done: GPT-2 124M runs end to end on the GPU (Intel Iris Xe, Vulkan), with logits
within 2.5e-6 (relative) of a plain-Rust CPU reference and identical greedy text. The reference
itself matches a float64 numpy implementation and Hugging Face's published output. Every kernel
(GELU, embedding, softmax, LayerNorm, naive matmul, causal attention) is differential-tested
against the reference. Next: M4, a KV cache.

## Approach
- **CPU oracle.** A plain-Rust reference implementation of every op (and of the whole model) is
  the source of truth. Every GPU kernel is differential-tested against it with explicit tolerances.
- **Deterministic.** Same device and inputs give bit-identical outputs: no float atomics, fixed
  reduction order, seeded sampling.
- **Measured.** Performance claims come with GPU timestamp numbers and the method used.

## Try it
```bash
cargo run --release -- info       # which GPU and limits wgpu gave us
cargo run --release -- selftest   # GPU add on 1M floats vs the CPU reference
cargo test
scripts/fetch_gpt2.sh                                    # GPT-2 124M into data/gpt2/ (550 MB)
cargo run --release -- tokenize "Hello world"            # BPE, step by step
cargo run --release -- generate -n 16 "I enjoy walking with my cute dog"        # on the GPU
cargo run --release -- generate --cpu -n 16 "I enjoy walking with my cute dog"  # CPU reference
```
