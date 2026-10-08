// RMSNorm of each row of x: [rows, cols] (see cpu::rms_norm): out = x / sqrt(mean(x^2) + eps)
// * gain. One workgroup per row; the sum of squares is a serial fold per thread followed by the
// fixed tree in reduce.wgsl (D18), so the bits never depend on timing.

struct Params {
    cols: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gain: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let l = lid.x;
    let n = params.cols;
    let base = wid.x * n;

    var sq = 0.0;
    for (var i = l; i < n; i += WG) {
        let v = x[base + i];
        sq += v * v;
    }
    let inv = 1.0 / sqrt(tree_sum(l, sq) / f32(n) + params.eps);

    for (var i = l; i < n; i += WG) {
        out[base + i] = x[base + i] * inv * gain[i];
    }
}
