// The compute roof (D47): every invocation runs 32 independent fma chains (8 vec4s) for `iters`
// steps, 2 * 32 * iters flops, with no memory traffic in the loop. Independent chains let the
// hardware overlap fma latency, as a good matmul's register block does. The result is stored
// so the compiler can't drop the loop; fma isn't reassociable, so it can't shorten it.
//
// Plain variables, not an array: Mesa keeps a private array indexed in a loop out of registers,
// and the same work then runs 12x slower (D47). Measured on the Iris Xe: 8 chains in an array
// 116 GFLOP/s, 8 scalars 900, 4 vec4s 1177, 8 vec4s 1431, 16 vec4s 489 (out of registers).

struct Params {
    iters: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = f32(gid.x) * 1e-6;
    var a0 = vec4(f, f + 1e-6, f + 2e-6, f + 3e-6);
    var a1 = a0 + 4e-6;
    var a2 = a0 + 8e-6;
    var a3 = a0 + 12e-6;
    var a4 = a0 + 16e-6;
    var a5 = a0 + 20e-6;
    var a6 = a0 + 24e-6;
    var a7 = a0 + 28e-6;
    let m = vec4(0.999999);
    let c = vec4(1e-7);
    for (var i = 0u; i < params.iters; i++) {
        a0 = fma(a0, m, c);
        a1 = fma(a1, m, c);
        a2 = fma(a2, m, c);
        a3 = fma(a3, m, c);
        a4 = fma(a4, m, c);
        a5 = fma(a5, m, c);
        a6 = fma(a6, m, c);
        a7 = fma(a7, m, c);
    }
    let s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[gid.x] = s.x + s.y + s.z + s.w;
}
