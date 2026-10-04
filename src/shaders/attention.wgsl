// Causal multi-head attention with a KV cache (D31). Queries come from qkv: [T, 3E] (the Q
// third of each row); keys and values from the layer's cache k_cache, v_cache: [n_ctx, E],
// which already holds rows 0..start+T (kv_write ran first). Query row i sits at absolute
// position pos = start + i and attends cache rows 0..=pos. Prefill: start = 0, T rows.
// Decode: start = t, 1 row. Output: [T, E].
//
// Naive (D20): one workgroup per (query row i, head h).
// 1. Score keys j = 0..=pos: thread l takes j = l, l + 64, ...; each score is a serial dot
//    product of length d. Scores live in shared memory (at most MAX_CTX of them).
// 2. Softmax over those scores: max and sum by the fixed tree (D18), then normalize.
// 3. Thread l produces output dims c = l, l + 64, ... as a serial weighted sum of v_j over j.
// Keys after pos are never scored, so the causal mask costs nothing.

struct Params {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,  // 1 / sqrt(d), computed on the host so both sides use the same f32
    start: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 64u;
const MAX_CTX: u32 = 1024u;  // must match ops::ATTENTION_MAX_CTX
const LOWEST: f32 = -3.40282347e38;

var<workgroup> scores: array<f32, MAX_CTX>;
var<workgroup> partial: array<f32, WG>;

fn tree_max(l: u32, v: f32) -> f32 {
    partial[l] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] = max(partial[l], partial[l + s]);
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();  // everyone has read partial[0] before the next tree overwrites it
    return r;
}

fn tree_sum(l: u32, v: f32) -> f32 {
    partial[l] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] += partial[l + s];
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wid.x;
    let h = wid.y;
    let l = lid.x;
    let e = params.e;
    let hd = h * params.d;        // this head's column offset within Q, K and V
    let q = i * 3u * e + hd;      // Q is the first third of the qkv row
    let pos = params.start + i;   // absolute position of this query

    // 1. scores
    var m = LOWEST;
    for (var j = l; j <= pos; j += WG) {
        let k = j * e + hd;
        var dot = 0.0;
        for (var c = 0u; c < params.d; c++) {
            dot += qkv[q + c] * k_cache[k + c];
        }
        let s = dot * params.scale;
        scores[j] = s;
        m = max(m, s);
    }

    // 2. softmax over scores[0..=pos]
    let row_max = tree_max(l, m);
    var sum = 0.0;
    for (var j = l; j <= pos; j += WG) {
        let p = exp(scores[j] - row_max);
        scores[j] = p;
        sum += p;
    }
    let total = tree_sum(l, sum);
    for (var j = l; j <= pos; j += WG) {
        scores[j] = scores[j] / total;
    }
    workgroupBarrier();  // every probability is written before anyone reads all of them

    // 3. weighted sum of V
    for (var c = l; c < params.d; c += WG) {
        var acc = 0.0;
        for (var j = 0u; j <= pos; j++) {
            acc += scores[j] * v_cache[j * e + hd + c];
        }
        out[i * e + hd + c] = acc;
    }
}
