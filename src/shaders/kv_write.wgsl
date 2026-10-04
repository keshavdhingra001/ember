// Append K and V rows to a layer's cache (D32): for t in 0..T and c in 0..E,
//   k_cache[start + t, c] = qkv[t, E + c]      (the K third of the qkv row)
//   v_cache[start + t, c] = qkv[t, 2E + c]     (the V third)
// One invocation per (t, c), grid-stride (D8). The host checks start + T <= n_ctx.

struct Params {
    n: u32,      // T * E
    e: u32,
    start: u32,  // absolute position of qkv row 0
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let e = params.e;
    for (var i = gid.x; i < params.n; i += stride) {
        let t = i / e;
        let c = i % e;
        let dst = (params.start + t) * e + c;
        k_cache[dst] = qkv[t * 3u * e + e + c];
        v_cache[dst] = qkv[t * 3u * e + 2u * e + c];
    }
}
