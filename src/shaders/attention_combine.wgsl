// Causal multi-head attention, pass 2 of 2 (D63): merge the key chunks of attention.wgsl. One
// workgroup per (query row i, head h); thread l produces output dims l, l + 64, ....
//
// For chunks c = 0 .. pos / KC with (m_c, l_c, o_c):
//   M = max_c m_c,  L = sum_c l_c exp(m_c - M),  out = (sum_c o_c exp(m_c - M)) / L
// summed in chunk order, serially (every thread computes the same M and L). That is softmax(s) V
// over all keys 0..=pos: rescaling each chunk to the global max is the online softmax. The
// order depends only on pos, so the bits don't depend on the batch (D33).

struct Params {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,  // unused here; the layout matches pass 1's
    start: u32,
    n_chunks: u32,
    kv: u32,     // unused here
    group: u32,  // unused here
}

@group(0) @binding(0) var<storage, read> parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 64u;
const KC: u32 = 64u;  // keys per chunk, as in attention.wgsl
const LOWEST: f32 = -3.40282347e38;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = wid.x;
    let h = wid.y;
    let l = lid.x;
    let d = params.d;
    let pos = params.start + i;
    let nc = pos / KC + 1u;  // chunks with keys for this row
    let row = (i * nwg.y + h) * params.n_chunks;  // this (row, head)'s first chunk
    let size = d + 2u;

    var m = LOWEST;
    for (var c = 0u; c < nc; c++) {
        m = max(m, parts[(row + c) * size]);
    }
    var total = 0.0;
    for (var c = 0u; c < nc; c++) {
        let b = (row + c) * size;
        total += parts[b + 1u] * exp(parts[b] - m);
    }
    for (var cc = l; cc < d; cc += WG) {
        var acc = 0.0;
        for (var c = 0u; c < nc; c++) {
            let b = (row + c) * size;
            acc += parts[b + 2u + cc] * exp(parts[b] - m);
        }
        out[i * params.e + h * d + cc] = acc / total;
    }
}
