// out[i, o] = x[i, :] . w[o, :] + b[o]: x is [t, n_in], w is [n_out, n_in] (D7).
// Naive (D19): one invocation per output element, a serial dot product in the CPU's order.
// 16 x 16 workgroups: gid.x walks output features, gid.y rows.
//
// Known weakness, kept on purpose as M6's baseline: threads with neighbouring `o` read rows of
// w that are n_in floats apart, so their loads don't coalesce, and nothing is reused through
// shared memory.

struct Params {
    t: u32,
    n_in: u32,
    n_out: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let i = gid.y;
    // The grid is rounded up to whole workgroups; the extra invocations do nothing.
    if (o >= params.n_out || i >= params.t) {
        return;
    }
    let xr = i * params.n_in;
    let wr = o * params.n_in;
    var acc = 0.0;
    for (var k = 0u; k < params.n_in; k++) {
        acc += x[xr + k] * w[wr + k];
    }
    if (params.has_bias != 0u) {
        acc += b[o];
    }
    out[i * params.n_out + o] = acc;
}
