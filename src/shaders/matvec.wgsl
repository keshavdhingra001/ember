// out[o] = x . w[:, o] + b[o] for one row x: [n_in], w: [n_in, n_out] (D45), then the epilogue
// (D62). The decode step
// (T = 1, D44): a 64 x 64 matmul tile would leave 63 of its 64 rows empty.
//
// Chunked sum (D51): K is cut into chunks of CHUNK = 256. Each chunk's partial is a serial chain
// fma(x[k], w[k, o], p) from p = 0, and the output is 0 + p0 + p1 + p2 + ..., added in chunk
// order, then the bias. matmul.wgsl computes exactly the same operations in the same order, so
// decode equals the prefill row bit for bit (D46, D33).
//
// Layout (D52, D56): a workgroup of 256 threads owns 256 / SLICES consecutive outputs and
// splits K SLICES ways. Thread l computes output o = outs * wg + l % outs for slice
// s = l / outs. K is walked in rounds of SLICES chunks: in each round slice s computes chunk
// SLICES * round + s. With SLICES = 4 that gives n_out * 4 threads instead of n_out (768 -> 3072
// for attn_out and fc_out), so more loads are in flight. At every k, the outs threads of a slice
// read outs consecutive floats of row k of w: a 1 KiB run with one slice, 256 bytes with four,
// which is why wide matrices, which have enough threads already, run with one. All threads need
// the same x values, so a round's SLICES * 256 values of x are staged in shared memory.
// The slice count changes which thread computes a partial, never the order they're added in,
// so every configuration gives the same bits.
//
// Compiled three times from this source (D48, D56): (SLICES, LOOKAHEAD) = (4, on) up to 1024
// outputs, (1, on) below 16384, (1, off) above. With LOOKAHEAD, a thread issues 8 loads before
// the 8 fmas that use them, so 8 loads are in flight per thread: that pays when there are few
// threads. Without it, for the LM head: enough threads already, and the lookahead's extra
// registers cost more than they save.

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
// binding 4: `res`, in epilogue.wgsl (prepended), which also has finish()
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const CHUNK: u32 = 256u;            // k values per chunk; matmul.wgsl must use the same (D51)
const MAX_SLICES: u32 = 4u;

override LOOKAHEAD: bool = true;
// Chunks computed side by side, one per slice: 1, 2 or 4 (at most MAX_SLICES, so xs fits).
override SLICES: u32 = 4u;

var<workgroup> xs: array<f32, MAX_SLICES * CHUNK>;
// parts[s * outs + j]: this round's partial of slice s for the workgroup's output j.
var<workgroup> parts: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let outs = WG / SLICES;  // outputs per workgroup
    let round = SLICES * CHUNK;  // k values per round
    let j = lid.x % outs;
    let s = lid.x / outs;
    let o = wg.x * outs + j;
    let n = params.n_out;
    // Threads past n_out can't return early: they still stage x and reach every barrier.
    let live = o < n;
    // The running total, kept by slice 0's thread for each output.
    var total = 0.0;
    for (var r0 = 0u; r0 < params.n_in; r0 += round) {
        // Stage this round's x, 4 values per thread. The barrier at the end of the previous
        // round guarantees nobody still reads the previous round's values.
        for (var q = 0u; q < round / WG; q++) {
            let i = lid.x + q * WG;
            if (r0 + i < params.n_in) {
                xs[i] = x[r0 + i];
            }
        }
        // The x values must be staged before anyone reads them.
        workgroupBarrier();

        // This slice's chunk: k from c0 to c0 + steps, x at xs[s * CHUNK ...]. A chunk past
        // n_in (the last round may have fewer than 4) is empty: steps = 0 and p stays 0, and
        // slice 0 doesn't add it below.
        let c0 = r0 + s * CHUNK;
        var steps = 0u;
        if (c0 < params.n_in) {
            steps = min(CHUNK, params.n_in - c0);
        }
        var p = 0.0;
        if (live) {
            let xb = s * CHUNK;
            var k = 0u;
            // Eight independent loads first, then their fmas in k order: the loads overlap, the
            // summation order doesn't change (D46). The loop below finishes the chunk either way.
            for (; LOOKAHEAD && k + 8u <= steps; k += 8u) {
                let base = (c0 + k) * n + o;
                let w0 = w[base];
                let w1 = w[base + n];
                let w2 = w[base + 2u * n];
                let w3 = w[base + 3u * n];
                let w4 = w[base + 4u * n];
                let w5 = w[base + 5u * n];
                let w6 = w[base + 6u * n];
                let w7 = w[base + 7u * n];
                p = fma(xs[xb + k], w0, p);
                p = fma(xs[xb + k + 1u], w1, p);
                p = fma(xs[xb + k + 2u], w2, p);
                p = fma(xs[xb + k + 3u], w3, p);
                p = fma(xs[xb + k + 4u], w4, p);
                p = fma(xs[xb + k + 5u], w5, p);
                p = fma(xs[xb + k + 6u], w6, p);
                p = fma(xs[xb + k + 7u], w7, p);
            }
            for (; k < steps; k++) {
                p = fma(xs[xb + k], w[(c0 + k) * n + o], p);
            }
        }
        parts[s * outs + j] = p;
        // Every partial must be written before slice 0 adds them. This barrier also ends every
        // read of xs in this round, so the next round may restage it. Slice 0 reads parts below
        // before it reaches the next round's first barrier, and nobody writes parts before that
        // barrier, so the reads are safe too.
        workgroupBarrier();
        if (s == 0u) {
            // In chunk order (D51). Only chunks that exist: adding an empty chunk's +0 would not
            // change the bits (total is never -0), but matmul doesn't add it either.
            for (var c = 0u; c < SLICES && r0 + c * CHUNK < params.n_in; c++) {
                total += parts[c * outs + j];
            }
        }
    }
    if (live && s == 0u) {
        out[o] = finish(o, o, total);
    }
}
