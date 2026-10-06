// LayerNorm of each row of x: [rows, cols] (see cpu::layer_norm). One workgroup per row, two
// sum reductions (mean, then the mean of squared deviations: two passes, as on the CPU), each
// a serial fold per thread followed by the fixed tree in reduce.wgsl (D18).

struct Params {
    cols: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gain: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    let n = params.cols;
    let base = wid.x * n;

    var s = 0.0;
    for (var i = l; i < n; i += WG) {
        s += x[base + i];
    }
    let mean = tree_sum(l, s) / f32(n);

    var sq = 0.0;
    for (var i = l; i < n; i += WG) {
        let d = x[base + i] - mean;
        sq += d * d;
    }
    let var_ = tree_sum(l, sq) / f32(n);
    let inv_std = 1.0 / sqrt(var_ + params.eps);

    for (var i = l; i < n; i += WG) {
        out[base + i] = (x[base + i] - mean) * inv_std * gain[i] + bias[i];
    }
}
