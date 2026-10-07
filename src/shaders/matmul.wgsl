// out[i, o] = x[i, :] . w[:, o] + b[o]: x is [t, n_in], w is [n_in, n_out] (the GPU layout, D45).
// Tiled (D43): a 16 x 16 workgroup computes a BM x BN = 64 x 64 block of out. Each thread owns a
// 4 x 4 block of it in registers (four vec4 accumulators): rows ty + 16 * r and columns tx + 16 * c, so neighbouring
// threads touch neighbouring columns (coalesced stores, no shared-memory bank conflicts). K is
// walked BK = 16 at a time: the workgroup stages a 64 x 16 slice of x and a 16 x 64 slice of w in
// shared memory, then every thread does its 16 x 4 x 4 multiply-adds from there. Each staged
// value is read by 16 threads, so global traffic drops 16x against the naive kernel (D19).
//
// Every output is the serial sum fma(x[i, k], w[k, o], acc) for k = 0, 1, ..., n_in - 1, the
// same sequence matvec.wgsl computes, so a row of a batch and the same row alone have
// identical bits (D46).

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

const TS: u32 = 16u;          // threads per side of the workgroup
const R: u32 = 4u;            // rows and columns of out per thread
const BM: u32 = TS * R;       // rows of out per workgroup
const BN: u32 = TS * R;       // columns of out per workgroup
const BK: u32 = 16u;          // k values staged per step
const THREADS: u32 = TS * TS;

// The staged slices, packed so one vec4 load gives a thread its 4 values for one k:
// xs[k][j] = x rows j, j + 16, j + 32, j + 48 (of this tile) at k, and
// ws[k][j] = w columns j, j + 16, j + 32, j + 48 at k.
var<workgroup> xs: array<array<vec4<f32>, TS>, BK>;
var<workgroup> ws: array<array<vec4<f32>, TS>, BK>;

// Row `i` of out, columns o, o + 16, o + 32, o + 48, from one accumulator. The bounds checks
// must stay even though dropping the row check passes every test here (D50): WebGPU lets an
// out-of-bounds write land anywhere in the same buffer, and this driver happens to drop it.
fn store_row(i: u32, o: u32, acc: vec4<f32>) {
    if (i >= params.t) {
        return;
    }
    for (var c = 0u; c < R; c++) {
        let col = o + TS * c;
        if (col < params.n_out) {
            var v = acc[c];
            if (params.has_bias != 0u) {
                v += b[col];
            }
            out[i * params.n_out + col] = v;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * TS + tx;
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;

    // acc_r holds row ty + 16 r, columns tx + 16 c in lane c. Four named vectors, not an array:
    // Mesa keeps a loop-indexed private array out of registers, which costs ~12x (D47).
    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    for (var k0 = 0u; k0 < params.n_in; k0 += BK) {
        // Stage the slices: BM * BK = BK * BN = 1024 values each, 4 per thread. Consecutive
        // threads load consecutive addresses (a 16-float run of an x row, a 64-float run of a w
        // row). Out-of-range values are staged as 0 but never used: the inner loop below stops
        // at n_in, and rows / columns past the edge are never written.
        for (var q = 0u; q < BM * BK / THREADS; q++) {
            let e = l + q * THREADS;
            let r = e / BK;
            let k = e % BK;
            let gr = row0 + r;
            let gk = k0 + k;
            var v = 0.0;
            if (gr < params.t && gk < params.n_in) {
                v = x[gr * params.n_in + gk];
            }
            xs[k][r % TS][r / TS] = v;
        }
        for (var q = 0u; q < BK * BN / THREADS; q++) {
            let e = l + q * THREADS;
            let k = e / BN;
            let c = e % BN;
            let gk = k0 + k;
            let gc = col0 + c;
            var v = 0.0;
            if (gk < params.n_in && gc < params.n_out) {
                v = w[gk * params.n_out + gc];
            }
            ws[k][c % TS][c / TS] = v;
        }
        // The stores above must land before any thread reads the slices.
        workgroupBarrier();

        // Only the k values that exist. A padded step would add fma(0, 0, acc) = acc, so the bits
        // wouldn't change (D46); stopping here just skips the work.
        let steps = min(BK, params.n_in - k0);
        for (var k = 0u; k < steps; k++) {
            let a = xs[k][ty];
            let bv = ws[k][tx];
            acc0 = fma(vec4(a.x), bv, acc0);
            acc1 = fma(vec4(a.y), bv, acc1);
            acc2 = fma(vec4(a.z), bv, acc2);
            acc3 = fma(vec4(a.w), bv, acc3);
        }
        // Every thread must finish reading this step's slices before the next step overwrites
        // them.
        workgroupBarrier();
    }

    let o = col0 + tx;
    store_row(row0 + ty, o, acc0);
    store_row(row0 + ty + TS, o, acc1);
    store_row(row0 + ty + 2u * TS, o, acc2);
    store_row(row0 + ty + 3u * TS, o, acc3);
}
