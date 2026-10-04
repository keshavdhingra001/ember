# ember

An LLM inference engine on WebGPU, written in Rust with [wgpu](https://wgpu.rs) and WGSL compute
shaders. It runs natively on Vulkan (developed on an Intel Iris Xe iGPU), and the plan is to run
in the browser via wasm too.

**Status:** M2 done: every GPT-2 op has a WGSL kernel (GELU, embedding, softmax, LayerNorm,
naive matmul, causal attention) differential-tested against the CPU reference. M1's CPU reference
loads GPT-2 124M, tokenizes with a hand-written BPE, and generates greedily, matching a float64
numpy reference. Next: M3, the whole model on the GPU. See [CHECKPOINT.md](CHECKPOINT.md) for the roadmap and
[DESIGN.md](DESIGN.md) for every design decision.

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
cargo run --release -- generate -n 16 "I enjoy walking with my cute dog"
```
