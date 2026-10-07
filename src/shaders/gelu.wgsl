// out[i] = gelu(x[i]) for i in 0..n, GPT-2's tanh approximation (gelu_fn.wgsl).
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

// gelu() comes from gelu_fn.wgsl, prepended.

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
