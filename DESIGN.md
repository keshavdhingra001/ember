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

Today (M0): a headless wgpu device, host/GPU tensors, upload and readback, one elementwise
kernel (`add`) checked against the CPU reference with the tolerance comparator, and `ember info`.

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
