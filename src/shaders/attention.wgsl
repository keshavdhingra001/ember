// Causal multi-head attention with a KV cache, pass 1 of 2 (D63): one key chunk. Queries come
// from qkv: [T, 3E] (the Q third of each row); keys and values from the layer's cache
// k_cache, v_cache: [n_ctx, E], which already holds rows 0..start+T (kv_write ran first). Query
// row i sits at absolute position pos = start + i and attends cache rows 0..=pos.
//
// Keys are cut into chunks of KC = 64 by absolute index: chunk c is keys 64c .. 64c + 63. One
// workgroup per (query row i, head h, chunk c) computes, over the chunk's keys j <= pos,
//   m = max_j s_j,  l = sum_j exp(s_j - m),  o[dim] = sum_j exp(s_j - m) v_j[dim]
// with s_j = (q . k_j) / sqrt(d), and writes (m, l, o) to parts. attention_combine.wgsl merges
// the chunks. What a chunk computes depends only on pos and c, never on T or start, so a row has
// the same bits in a prefill batch and in a decode step (D33).
//
// Each workgroup reads its chunk's keys once, coalesced: 32 keys at a time are staged in shared
// memory (rows padded to MAX_D + 1 floats, so 32 threads reading 32 different rows at the same
// dim hit 32 different banks), and thread l scores key l. Staging all 64 would need 16.6 KiB,
// past WebGPU's default 16 KiB of workgroup memory. Values are read straight from global
// memory: at each key the 64 threads read 64 consecutive dims.

struct Params {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,  // 1 / sqrt(d), computed on the host so both sides use the same f32
    start: u32,
    n_chunks: u32,  // chunks per (row, head) in parts: ceil((start + t) / KC)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
// parts[((i * heads + h) * n_chunks + c) * (d + 2) + ...]: m, l, then o[0..d].
@group(0) @binding(3) var<storage, read_write> parts: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 64u;
const KC: u32 = 64u;        // keys per chunk; ops::ATTENTION_CHUNK and the combine must match
const HALF: u32 = 32u;      // keys staged at a time
const MAX_D: u32 = 64u;     // largest head dimension; ops::ATTENTION_MAX_D must match
const STRIDE: u32 = MAX_D + 1u;
const LOWEST: f32 = -3.40282347e38;

var<workgroup> ks: array<f32, HALF * STRIDE>;
var<workgroup> qs: array<f32, MAX_D>;
var<workgroup> ps: array<f32, KC>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = wid.x;
    let h = wid.y;
    let c = wid.z;
    let l = lid.x;
    let e = params.e;
    let d = params.d;
    let hd = h * d;               // this head's column offset within Q, K and V
    let pos = params.start + i;   // absolute position of this query
    let j0 = c * KC;              // first key of the chunk
    // Chunks past pos have no keys for this row. The whole workgroup leaves together, so no
    // barrier below is skipped by some threads only.
    if (j0 > pos) {
        return;
    }
    let n = min(KC, pos + 1u - j0);  // keys of this chunk that this row attends

    for (var cc = l; cc < d; cc += WG) {
        qs[cc] = qkv[i * 3u * e + hd + cc];
    }

    // 1. Scores, 32 keys at a time; thread l scores key l (in half l / 32). Only the halves
    // that hold keys are staged: a row near the start of its chunk (a short prompt, a decode
    // step just past a chunk edge) has fewer than 32. n is the same for the whole workgroup, so
    // every thread runs the same iterations and reaches the same barriers.
    var s = LOWEST;
    for (var h0 = 0u; h0 < n; h0 += HALF) {
        // Stage keys h0 .. h0 + 31 of the chunk, one key row per step: thread l loads dim l
        // (d <= MAX_D = WG), so each step is one coalesced read of d floats.
        let rows = min(HALF, n - h0);
        for (var r = 0u; r < rows; r++) {
            if (l < d) {
                ks[r * STRIDE + l] = k_cache[(j0 + h0 + r) * e + hd + l];
            }
        }
        workgroupBarrier();  // the half is staged (and q, the first time)
        if (l >= h0 && l < h0 + rows) {
            var dot = 0.0;
            for (var cc = 0u; cc < d; cc++) {
                dot += qs[cc] * ks[(l - h0) * STRIDE + cc];
            }
            s = dot * params.scale;
        }
        workgroupBarrier();  // every score of this half is done before the next half restages
    }

    // 2. This chunk's max and sum of exponentials, by the fixed trees (D18).
    let m = tree_max(l, s);
    var p = 0.0;
    if (l < n) {
        p = exp(s - m);
    }
    ps[l] = p;
    let total = tree_sum(l, p);  // its barriers also make every ps[] visible

    // 3. Unnormalized output: thread l takes dims l, l + 64, ...; keys in order.
    let base = ((i * nwg.y + h) * params.n_chunks + c) * (d + 2u);
    for (var cc = l; cc < d; cc += WG) {
        var acc = 0.0;
        for (var r = 0u; r < n; r++) {
            acc += ps[r] * v_cache[(j0 + r) * e + hd + cc];
        }
        parts[base + 2u + cc] = acc;
    }
    if (l == 0u) {
        parts[base] = m;
        parts[base + 1u] = total;
    }
}
