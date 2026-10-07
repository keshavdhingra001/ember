// Sum x's n vec4s, grid-stride (D8): the read-only bandwidth probe (D47). A decode step only
// reads its weights, and on this GPU a stream of reads runs faster than copy's reads + writes
// (D41), so the matvec is measured against this one. Few invocations (the host launches 1024
// workgroups), each reading many vec4s: the one store per invocation is ~1.6% of the bytes read
// for 256 MiB.

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
    var acc = vec4(0.0);
    for (var i = gid.x; i < params.n; i += stride) {
        acc += x[i];
    }
    out[gid.x] = acc;
}
