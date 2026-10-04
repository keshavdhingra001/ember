// Softmax of each row of x: [rows, cols]. One workgroup per row (D18).
//
// Two reductions per row, max then sum. Each uses the same pattern: thread `l` folds elements
// l, l + 256, l + 512, ... serially, writes its partial result to shared memory, then a fixed
// halving tree (128, 64, ..., 1) combines the 256 partials. Fixed shapes, fixed order: the
// same bits on every run (D4), though not the CPU's serial order, hence a tolerance (D22).

struct Params {
    cols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const LOWEST: f32 = -3.40282347e38;  // WGSL has no infinity literal

var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    let base = wid.x * params.cols;

    var m = LOWEST;
    for (var i = l; i < params.cols; i += WG) {
        m = max(m, x[base + i]);
    }
    partial[l] = m;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] = max(partial[l], partial[l + s]);
        }
        workgroupBarrier();
    }
    let row_max = partial[0];
    workgroupBarrier();  // everyone has read partial[0] before it is overwritten below

    var sum = 0.0;
    for (var i = l; i < params.cols; i += WG) {
        let e = exp(x[base + i] - row_max);
        out[base + i] = e;  // each thread rereads only its own elements: no barrier needed
        sum += e;
    }
    partial[l] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] += partial[l + s];
        }
        workgroupBarrier();
    }
    let total = partial[0];

    for (var i = l; i < params.cols; i += WG) {
        out[base + i] = out[base + i] / total;
    }
}
