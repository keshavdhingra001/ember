// Append K and V rows to a layer's cache (D32). A qkv row is [q (Q) | k (KV) | v (KV)] with
// Q = n_head d and KV = n_kv_head d (D71; GPT-2 has KV = Q = E). For t in 0..T and c in 0..KV,
//   k_cache[start + t, c] = qkv[t, Q + c]
//   v_cache[start + t, c] = qkv[t, Q + KV + c]
// One invocation per (t, c), grid-stride (D8). The host checks start + T <= n_ctx.

struct Params {
    n: u32,      // T * KV
    kv: u32,     // cache row width, n_kv_head * d
    start: u32,  // absolute position of qkv row 0
    q: u32,      // query width, n_head * d
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
    let kv = params.kv;
    let row = params.q + 2u * kv;
    for (var i = gid.x; i < params.n; i += stride) {
        let t = i / kv;
        let c = i % kv;
        let dst = (params.start + t) * kv + c;
        k_cache[dst] = qkv[t * row + params.q + c];
        v_cache[dst] = qkv[t * row + params.q + kv + c];
    }
}
