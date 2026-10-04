// Causal multi-head attention from the fused qkv: [t, 3e] -> out: [t, e] (see
// cpu::causal_attention). Naive (D20): one workgroup per (query row i, head h).
//
// 1. Score keys j = 0..=i: thread l takes j = l, l + 64, ...; each score is a serial dot
//    product of length d. Scores live in shared memory (at most MAX_CTX of them).
// 2. Softmax over those scores: max and sum by the fixed tree (D18), then normalize.
// 3. Thread l produces output dims c = l, l + 64, ... as a serial weighted sum of v_j over j.
// Keys after i are never scored, so the causal mask costs nothing.

struct Params {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,  // 1 / sqrt(d), computed on the host so both sides use the same f32
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

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
    workgroupBarrier();
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
    let row = 3u * params.e;      // floats per qkv row
    let hd = h * params.d;        // this head's column offset within Q, K and V
    let q = i * row + hd;

    // 1. scores
    var m = LOWEST;
    for (var j = l; j <= i; j += WG) {
        let k = j * row + params.e + hd;
        var dot = 0.0;
        for (var c = 0u; c < params.d; c++) {
            dot += qkv[q + c] * qkv[k + c];
        }
        let s = dot * params.scale;
        scores[j] = s;
        m = max(m, s);
    }

    // 2. softmax over scores[0..=i]
    let row_max = tree_max(l, m);
    var sum = 0.0;
    for (var j = l; j <= i; j += WG) {
        let p = exp(scores[j] - row_max);
        scores[j] = p;
        sum += p;
    }
    let total = tree_sum(l, sum);
    for (var j = l; j <= i; j += WG) {
        scores[j] = scores[j] / total;
    }
    workgroupBarrier();  // every probability is written before anyone reads all of them

    // 3. weighted sum of V
    for (var c = l; c < params.d; c += WG) {
        var acc = 0.0;
        for (var j = 0u; j <= i; j++) {
            acc += scores[j] * qkv[j * row + 2u * params.e + hd + c];
        }
        out[i * params.e + hd + c] = acc;
    }
}
