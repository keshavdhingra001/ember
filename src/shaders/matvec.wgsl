// out[o] = x . w[:, o] + b[o] for one row x: [n_in], w: [n_in, n_out] (D45). The decode step
// (T = 1, D44): a 64 x 64 matmul tile would leave 63 of its 64 rows empty.
// One invocation per output. At every k the workgroup's 256 threads read 256 consecutive floats
// of row k of w (one coalesced 1 KiB run), and all of them need the same x[k], so x is staged
// through shared memory 256 values at a time.
//
// Each output is the serial sum fma(x[k], w[k, o], acc) for k = 0, 1, ..., n_in - 1: the same
// sequence as matmul.wgsl, so decode equals the prefill row bit for bit (D46).
//
// Compiled twice (D48). With LOOKAHEAD, a thread issues 8 loads before the 8 fmas that use
// them, so 8 loads are in flight per thread: that pays when n_out is small (768 outputs = 768
// threads, too few to hide memory latency otherwise). Without it, for the LM head's 50257
// outputs: enough threads already, and the lookahead's extra registers cost more than they save.

struct Params {
    t: u32,  // always 1; the layout matches matmul's Params
    n_in: u32,
    n_out: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

override LOOKAHEAD: bool = true;

var<workgroup> xs: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let o = gid.x;
    // Threads past n_out can't return early: they still stage x and reach every barrier.
    let live = o < params.n_out;
    var acc = 0.0;
    for (var k0 = 0u; k0 < params.n_in; k0 += WG) {
        if (k0 + lid.x < params.n_in) {
            xs[lid.x] = x[k0 + lid.x];
        }
        // The x chunk must be staged before anyone reads it.
        workgroupBarrier();
        let steps = min(WG, params.n_in - k0);
        if (live) {
            // Eight independent loads first, then their fmas in k order: the loads overlap, the
            // summation order doesn't change (D46). The loop below finishes the chunk either way.
            let n = params.n_out;
            var k = 0u;
            for (; LOOKAHEAD && k + 8u <= steps; k += 8u) {
                let base = (k0 + k) * n + o;
                let w0 = w[base];
                let w1 = w[base + n];
                let w2 = w[base + 2u * n];
                let w3 = w[base + 3u * n];
                let w4 = w[base + 4u * n];
                let w5 = w[base + 5u * n];
                let w6 = w[base + 6u * n];
                let w7 = w[base + 7u * n];
                acc = fma(xs[k], w0, acc);
                acc = fma(xs[k + 1u], w1, acc);
                acc = fma(xs[k + 2u], w2, acc);
                acc = fma(xs[k + 3u], w3, acc);
                acc = fma(xs[k + 4u], w4, acc);
                acc = fma(xs[k + 5u], w5, acc);
                acc = fma(xs[k + 6u], w6, acc);
                acc = fma(xs[k + 7u], w7, acc);
            }
            for (; k < steps; k++) {
                acc = fma(xs[k], w[(k0 + k) * n + o], acc);
            }
        }
        // Everyone must be done with this chunk before the next one overwrites it.
        workgroupBarrier();
    }
    if (live) {
        if (params.has_bias != 0u) {
            acc += b[o];
        }
        out[o] = acc;
    }
}
