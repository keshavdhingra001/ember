// SwiGLU's gate (D73): gu: [T, 2F] holds the gate projection in columns 0..F and the up
// projection in F..2F of each row; out[t, j] = silu(gu[t, j]) * gu[t, F + j], out: [T, F],
// silu(x) = x / (1 + exp(-x)). Grid-stride loop as in add.wgsl (D8).

struct Params {
    n: u32,  // T * F
    f: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let f = params.f;
    for (var i = gid.x; i < params.n; i += stride) {
        let t = i / f;
        let j = i % f;
        let g = gu[t * 2u * f + j];
        out[i] = g / (1.0 + exp(-g)) * gu[t * 2u * f + f + j];
    }
}
