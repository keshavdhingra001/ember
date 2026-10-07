// GPT-2's GELU, tanh approximation (see cpu::gelu_scalar). Prepended to gelu.wgsl and to the
// linear kernels' epilogue (D62): one text, so the fused and the separate GELU are the same
// operations.

const SQRT_2_OVER_PI: f32 = 0.7978846;

fn gelu(v: f32) -> f32 {
    let u = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
    // tanh(u) is exactly 1.0 or -1.0 in f32 once |u| > 9.01, and some drivers compute tanh as
    // (e^2u - 1) / (e^2u + 1), which is inf / inf = NaN for large u. Clamping is exact and safe.
    return 0.5 * v * (1.0 + tanh(clamp(u, -10.0, 10.0)));
}
