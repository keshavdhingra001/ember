// out[t, c] = wte[ids[t], c] + wpe[start + t, c]: a gather from the token table plus the row of
// the token's absolute position (start > 0 when decoding after a KV-cached prefix, D32).
// The GPU keeps the token table transposed, wte_t: [E, V] (D45), the layout the tied LM head
// reads fast. A token's embedding is column ids[t]: E reads V floats apart, negligible next to
// the LM head's 38.6M.
// One invocation per output element, grid-stride (D8). The host has already checked every id
// is < V and start + T <= n_ctx: a shader can't report an error, only produce wrong numbers.

struct Params {
    n: u32,     // T * E
    e: u32,
    start: u32,
    v: u32,
}

@group(0) @binding(0) var<storage, read> wte_t: array<f32>;
@group(0) @binding(1) var<storage, read> wpe: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < params.n; i += stride) {
        let t = i / params.e;
        let c = i % params.e;
        out[i] = wte_t[c * params.v + ids[t]] + wpe[params.start * params.e + i];  // row start + t, column c
    }
}
