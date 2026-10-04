// out[i] = gelu(x[i]) for i in 0..n, GPT-2's tanh approximation (see cpu::gelu_scalar).
// Grid-stride loop as in add.wgsl (D8).

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const SQRT_2_OVER_PI: f32 = 0.7978846;

fn gelu(v: f32) -> f32 {
    let u = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
    // tanh(u) is exactly 1.0 or -1.0 in f32 once |u| > 9.01, and some drivers compute tanh as
    // (e^2u - 1) / (e^2u + 1), which is inf / inf = NaN for large u. Clamping is exact and safe.
    return 0.5 * v * (1.0 + tanh(clamp(u, -10.0, 10.0)));
}

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < params.n; i += stride) {
        out[i] = gelu(x[i]);
    }
}
