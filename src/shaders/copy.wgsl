// out[i] = x[i] for i in 0..n vec4s: the bandwidth probe (D41). One 16-byte load and one 16-byte
// store per element, consecutive invocations on consecutive addresses, grid-stride (D8). Nothing
// is computed, so its time is the memory system's.

struct Params {
    n: u32,  // vec4 count
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < params.n; i += stride) {
        out[i] = x[i];
    }
}
