// out[i, o] = x[i, :] . w[:, o] + b[o] for 2 to 8 rows: x is [t, n_in], w is [n_in, n_out] (D45).
// A short prefill (D57): a 64 x 64 matmul tile would leave most of its rows empty and run too few
// workgroups (n_out / 64). Like the matvec, it is memory-bound, so this is matvec.wgsl with up
// to ROWS rows per thread: each weight is read once and used for every row.
//
// Same layout as matvec.wgsl (D52, D56): 256 / SLICES consecutive outputs per workgroup, K split
// SLICES ways in rounds of SLICES chunks of 256. Each thread keeps one partial per row, 8 rows
// in two vec4s (named vectors, not an array: D47). The lane-wise vec4 fma is the same fma as the
// matvec's, so every row is the chunked sum of D51 and has the bits it has alone or in a big
// batch (tested).
//
// x has 8 values per k here, too many to stage a whole round (4 chunks x 256 k x 8 rows = 32 KiB)
// in shared memory, so each round is staged in windows of SUB k values per chunk.

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
// binding 4: `res`, in epilogue.wgsl (prepended), which also has finish()
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const CHUNK: u32 = 256u;            // k values per chunk; matmul.wgsl must use the same (D51)
const MAX_SLICES: u32 = 4u;
const ROWS: u32 = 8u;               // most rows per dispatch; ops.rs must use the same
const SUB: u32 = 64u;               // k values per chunk staged at a time

override LOOKAHEAD: bool = true;
// Chunks computed side by side, one per slice: 1, 2 or 4 (at most MAX_SLICES, so xs fits).
override SLICES: u32 = 4u;

// xs[s * SUB + k][h]: rows 4h .. 4h + 3 of x at the window's k-th value of slice s's chunk.
var<workgroup> xs: array<array<vec4<f32>, 2>, MAX_SLICES * SUB>;
// parts[s * outs + j][h]: this round's partials of slice s for output j, rows 4h .. 4h + 3.
var<workgroup> parts: array<array<vec4<f32>, 2>, WG>;

// Row `i` of out at column `o`, from the total (bias and epilogue in finish(), D62); rows past
// t aren't written.
fn store(i: u32, o: u32, total: f32) {
    if (i < params.t) {
        let idx = i * params.n_out + o;
        out[idx] = finish(idx, o, total);
    }
}

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
    // Running totals of rows 0-3 and 4-7, kept by slice 0's thread for each output.
    var tot_a = vec4(0.0);
    var tot_b = vec4(0.0);
    for (var r0 = 0u; r0 < params.n_in; r0 += round) {
        // This slice's chunk: k from c0 to c0 + steps (0 steps past n_in, as in matvec.wgsl).
        let c0 = r0 + s * CHUNK;
        var steps = 0u;
        if (c0 < params.n_in) {
            steps = min(CHUNK, params.n_in - c0);
        }
        var pa = vec4(0.0);
        var pb = vec4(0.0);
        for (var w0 = 0u; w0 < CHUNK; w0 += SUB) {
            // Stage window w0 of every slice's chunk: SLICES * SUB k values x 8 rows. Consecutive
            // threads load consecutive k of one row. Rows past t and k past n_in are staged as 0
            // and never used.
            for (var q = 0u; q < SLICES * SUB * ROWS / WG; q++) {
                let e = lid.x + q * WG;
                let r = e / (SLICES * SUB);
                let kk = e % (SLICES * SUB);
                let gk = r0 + (kk / SUB) * CHUNK + w0 + kk % SUB;
                var v = 0.0;
                if (r < params.t && gk < params.n_in) {
                    v = x[r * params.n_in + gk];
                }
                xs[kk][r / 4u][r % 4u] = v;
            }
            // The window must be staged before anyone reads it.
            workgroupBarrier();
            // This chunk's k values in this window: none once the chunk has ended.
            let wsteps = min(SUB, steps - min(steps, w0));
            if (live) {
                let xb = s * SUB;
                let kb = c0 + w0;
                var k = 0u;
                // Eight loads in flight, then their fmas in k order (D48); the order per row is
                // the serial chain of matvec.wgsl.
                for (; LOOKAHEAD && k + 8u <= wsteps; k += 8u) {
                    let base = (kb + k) * n + o;
                    let w0v = w[base];
                    let w1v = w[base + n];
                    let w2v = w[base + 2u * n];
                    let w3v = w[base + 3u * n];
                    let w4v = w[base + 4u * n];
                    let w5v = w[base + 5u * n];
                    let w6v = w[base + 6u * n];
                    let w7v = w[base + 7u * n];
                    pa = fma(xs[xb + k][0], vec4(w0v), pa);
                    pb = fma(xs[xb + k][1], vec4(w0v), pb);
                    pa = fma(xs[xb + k + 1u][0], vec4(w1v), pa);
                    pb = fma(xs[xb + k + 1u][1], vec4(w1v), pb);
                    pa = fma(xs[xb + k + 2u][0], vec4(w2v), pa);
                    pb = fma(xs[xb + k + 2u][1], vec4(w2v), pb);
                    pa = fma(xs[xb + k + 3u][0], vec4(w3v), pa);
                    pb = fma(xs[xb + k + 3u][1], vec4(w3v), pb);
                    pa = fma(xs[xb + k + 4u][0], vec4(w4v), pa);
                    pb = fma(xs[xb + k + 4u][1], vec4(w4v), pb);
                    pa = fma(xs[xb + k + 5u][0], vec4(w5v), pa);
                    pb = fma(xs[xb + k + 5u][1], vec4(w5v), pb);
                    pa = fma(xs[xb + k + 6u][0], vec4(w6v), pa);
                    pb = fma(xs[xb + k + 6u][1], vec4(w6v), pb);
                    pa = fma(xs[xb + k + 7u][0], vec4(w7v), pa);
                    pb = fma(xs[xb + k + 7u][1], vec4(w7v), pb);
                }
                for (; k < wsteps; k++) {
                    let wv = vec4(w[(kb + k) * n + o]);
                    pa = fma(xs[xb + k][0], wv, pa);
                    pb = fma(xs[xb + k][1], wv, pb);
                }
            }
            // Everyone must be done with this window before the next one overwrites it.
            workgroupBarrier();
        }
        parts[s * outs + j][0] = pa;
        parts[s * outs + j][1] = pb;
        // Every partial must be written before slice 0 adds them. Slice 0 reads them before it
        // reaches the next round's first barrier, and nobody writes parts before that barrier.
        workgroupBarrier();
        if (s == 0u) {
            // In chunk order (D51), only chunks that exist, as in matvec.wgsl.
            for (var c = 0u; c < SLICES && r0 + c * CHUNK < params.n_in; c++) {
                tot_a += parts[c * outs + j][0];
                tot_b += parts[c * outs + j][1];
            }
        }
    }
    if (live && s == 0u) {
        for (var r = 0u; r < 4u; r++) {
            store(r, o, tot_a[r]);
            store(r + 4u, o, tot_b[r]);
        }
    }
}
