// The compute roof (D47): every invocation runs CHAINS independent fma chains for `iters`
// steps, 2 * CHAINS * iters flops, with no memory traffic in the loop. Independent chains let
// the hardware overlap fma latency, as a good matmul's register block does. The result is
// stored so the compiler can't drop the loop; fma isn't reassociable, so it can't shorten it.

struct Params {
    iters: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const CHAINS: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: array<f32, CHAINS>;
    for (var c = 0u; c < CHAINS; c++) {
        a[c] = f32(gid.x + c) * 1e-6;
    }
    for (var i = 0u; i < params.iters; i++) {
        for (var c = 0u; c < CHAINS; c++) {
            a[c] = fma(a[c], 0.999999, 1e-7);
        }
    }
    var s = 0.0;
    for (var c = 0u; c < CHAINS; c++) {
        s += a[c];
    }
    out[gid.x] = s;
}
