// out[t, c] = wte[ids[t], c] + wpe[t, c]: a gather from the token table plus the position row.
// One invocation per output element, grid-stride (D8). The host has already checked every id
// is < V and T <= n_ctx: a shader can't report an error, only produce wrong numbers.

struct Params {
    n: u32,     // T * E
    e: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> wte: array<f32>;
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
        out[i] = wte[ids[t] * params.e + c] + wpe[i];  // wpe row t, column c is index i
    }
}
