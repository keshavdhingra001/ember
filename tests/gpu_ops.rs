//! Every M2 kernel against its CPU twin in `cpu.rs` (D3), on edge shapes and random data, with
//! a per-op tolerance (D22), and run twice to check the bits repeat (D4).
//!
//! `cargo test --test gpu_ops -- --nocapture` prints the worst error per case; the tolerances
//! below have headroom over those numbers (recorded in DESIGN.md, D22).

mod common;

use common::gpu;
use ember::compare::{Stats, Tol, check};
use ember::rng::Rng;
use ember::{Tensor, cpu, ops};

fn random(shape: &[usize], lo: f32, hi: f32, seed: u64) -> Tensor {
    let n = shape.iter().product();
    Tensor::new(shape, Rng::new(seed).vec(n, lo, hi)).unwrap()
}

/// Compare a GPU result with the CPU's, print the error, and rerun the GPU op to check the
/// bits are identical. `run` performs the GPU op and reads the result back.
fn compare(label: &str, want: &Tensor, tol: Tol, run: impl Fn() -> Tensor) -> Stats {
    let got = run();
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let stats = check(got.data(), want.data(), tol).unwrap_or_else(|m| panic!("{label}: {m}"));
    eprintln!(
        "{label:<40} max_abs {:.2e}  max_rel {:.2e}",
        stats.max_abs, stats.max_rel
    );
    let again = run();
    let bits = |t: &Tensor| t.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&again), bits(&got), "{label}: not deterministic");
    stats
}

// ------------------------------------------------------------------ gelu

/// tanh differs between the CPU's libm and the GPU driver by a few ulp.
const GELU_TOL: Tol = Tol {
    abs: 1e-6,
    rel: 1e-5,
};

#[test]
fn gelu_matches_cpu() {
    let g = gpu();
    for (seed, n) in [1usize, 255, 256, 257, 100_003].into_iter().enumerate() {
        let x = random(&[n], -8.0, 8.0, seed as u64);
        let gx = g.upload(&x);
        compare(&format!("gelu n={n}"), &cpu::gelu(&x), GELU_TOL, || {
            g.read(&ops::gelu(g, &gx).unwrap()).unwrap()
        });
    }
}

#[test]
fn gelu_extremes_stay_finite() {
    // Large |x| is where a naive tanh returns NaN (inf / inf); see gelu.wgsl.
    let g = gpu();
    let x = Tensor::new(
        &[10],
        vec![0.0, -0.0, 9.0, -9.0, 20.0, -20.0, 100.0, -100.0, 1e4, -1e4],
    )
    .unwrap();
    let gx = g.upload(&x);
    compare("gelu extremes", &cpu::gelu(&x), GELU_TOL, || {
        g.read(&ops::gelu(g, &gx).unwrap()).unwrap()
    });
}

// ------------------------------------------------------------------ embed

#[test]
fn embed_matches_cpu_exactly() {
    // One add per element: must be bit-exact (D5).
    let g = gpu();
    let (v, e, n_ctx) = (37, 12, 16);
    let wte = random(&[v, e], -1.0, 1.0, 1);
    let wpe = random(&[n_ctx, e], -1.0, 1.0, 2);
    let (gwte, gwpe) = (g.upload(&cpu::transpose(&wte).unwrap()), g.upload(&wpe));
    let cases: [&[u32]; 4] = [&[0], &[36, 0, 5, 5], &[7; 16], &[]];
    for ids in cases {
        compare(
            &format!("embed T={}", ids.len()),
            &cpu::embed(&wte, &wpe, ids).unwrap(),
            Tol::EXACT,
            || {
                g.read(&ops::embed(g, &gwte, &gwpe, ids, 0).unwrap())
                    .unwrap()
            },
        );
    }
    assert!(ops::embed(g, &gwte, &gwpe, &[37], 0).is_err());
    assert!(ops::embed(g, &gwte, &gwpe, &[0; 17], 0).is_err());
}

// ------------------------------------------------------------------ softmax

/// exp differs by a few ulp; the sum is a tree on the GPU and serial on the CPU.
const SOFTMAX_TOL: Tol = Tol {
    abs: 1e-7,
    rel: 1e-5,
};

#[test]
fn softmax_matches_cpu() {
    let g = gpu();
    let shapes: [[usize; 2]; 8] = [
        [1, 1],
        [3, 255],
        [3, 256],
        [3, 257],
        [2, 1000],
        [16, 768],
        [1, 50257],
        [4, 3],
    ];
    for (seed, shape) in shapes.into_iter().enumerate() {
        let x = random(&shape, -10.0, 10.0, seed as u64);
        let gx = g.upload(&x);
        compare(
            &format!("softmax {shape:?}"),
            &cpu::softmax_rows(&x).unwrap(),
            SOFTMAX_TOL,
            || g.read(&ops::softmax_rows(g, &gx).unwrap()).unwrap(),
        );
    }
}

#[test]
fn softmax_large_logits_dont_overflow() {
    let g = gpu();
    let x = random(&[4, 300], 900.0, 1000.0, 9);
    let gx = g.upload(&x);
    compare(
        "softmax huge",
        &cpu::softmax_rows(&x).unwrap(),
        SOFTMAX_TOL,
        || g.read(&ops::softmax_rows(g, &gx).unwrap()).unwrap(),
    );
}

#[test]
fn softmax_uses_the_true_row_max() {
    // Softmax is invariant to the value subtracted, so a slightly wrong max is invisible. It
    // only shows when exp overflows: one logit of 100 among values in +-5 needs the max to be
    // exactly that logit, or exp(~95) = inf and the row turns to NaN. The spike positions are
    // chosen so each way of getting the max wrong misses them: index 301 belongs to thread 45
    // (odd, so not in the even half of the tree) and is not the last element that thread folds.
    let g = gpu();
    let cols = 1000;
    let mut x = random(&[3, cols], -5.0, 5.0, 31).data().to_vec();
    for (row, spike) in [301, 2, 640].into_iter().enumerate() {
        x[row * cols + spike] = 100.0;
    }
    let x = Tensor::new(&[3, cols], x).unwrap();
    let gx = g.upload(&x);
    compare(
        "softmax one dominant logit",
        &cpu::softmax_rows(&x).unwrap(),
        SOFTMAX_TOL,
        || g.read(&ops::softmax_rows(g, &gx).unwrap()).unwrap(),
    );
}

// ------------------------------------------------------------------ layer_norm

/// Two tree sums vs the oracle's f64 sums (D23), and Vulkan's sqrt isn't correctly rounded.
/// Measured worst 5.7e-6 (3072 columns, inputs around 100 with a spread of ~3). With the oracle
/// in f32 it was 8.8e-5, mostly the oracle's own error (D22).
const LAYER_NORM_TOL: Tol = Tol {
    abs: 2e-5,
    rel: 1e-5,
};

#[test]
fn layer_norm_matches_cpu() {
    let g = gpu();
    let shapes: [[usize; 2]; 6] = [[1, 1], [2, 255], [2, 256], [3, 257], [5, 768], [2, 3072]];
    for (seed, shape) in shapes.into_iter().enumerate() {
        let cols = shape[1];
        // An offset like GPT-2's residual stream: mean far from 0 relative to the spread.
        let x = random(&shape, 95.0, 105.0, seed as u64);
        let gain = random(&[cols], 0.5, 1.5, 100 + seed as u64);
        let bias = random(&[cols], -0.5, 0.5, 200 + seed as u64);
        let (gx, gg, gb) = (g.upload(&x), g.upload(&gain), g.upload(&bias));
        compare(
            &format!("layer_norm {shape:?}"),
            &cpu::layer_norm(&x, &gain, &bias, 1e-5).unwrap(),
            LAYER_NORM_TOL,
            || {
                let y = ops::layer_norm(g, &gx, &gg, &gb, 1e-5).unwrap();
                g.read(&y).unwrap()
            },
        );
    }
}

// ------------------------------------------------------------------ linear

/// The GPU rounds `acc + x * w` once (fused), the CPU twice, and `linear` sums in chunks of 256
/// (D51) while the CPU and `linear_naive` sum serially (D53, D54). Measured worst: see D54.
const LINEAR_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};

/// `(x, w, b)` for a `(T, in, out)` case, `w` in the CPU's `[out, in]`.
fn linear_case(t: usize, n_in: usize, n_out: usize, seed: u64) -> (Tensor, Tensor, Tensor) {
    (
        random(&[t, n_in], -1.0, 1.0, seed),
        random(&[n_out, n_in], -0.1, 0.1, 100 + seed),
        random(&[n_out], -1.0, 1.0, 200 + seed),
    )
}

#[test]
fn linear_matches_cpu() {
    let g = gpu();
    // (T, in, out). T = 1 runs the matvec, T > 1 the 64 x 64 tiled matmul. Edges: one element,
    // tile edges (64 rows and columns, 16-deep k steps, 256-wide matvec chunks) and one past
    // them, odd sizes, and GPT-2's shapes, including the LM head's 50257 columns.
    let shapes = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 255, 257),
        (1, 257, 255),
        (64, 16, 64),
        (65, 17, 65),
        (63, 15, 63),
        (17, 33, 15),
        (1, 768, 2304),
        (3, 768, 2304),
        (130, 768, 2304),
        (1, 3072, 768),
        (5, 3072, 768),
        (1, 768, 50257),
        (2, 768, 50257),
    ];
    for (seed, (t, n_in, n_out)) in shapes.into_iter().enumerate() {
        let (x, w, b) = linear_case(t, n_in, n_out, seed as u64);
        let (gx, gw, gb) = (
            g.upload(&x),
            g.upload(&cpu::transpose(&w).unwrap()),
            g.upload(&b),
        );
        compare(
            &format!("linear ({t}, {n_in}, {n_out})"),
            &cpu::linear(&x, &w, Some(&b)).unwrap(),
            LINEAR_TOL,
            || {
                g.read(&ops::linear(g, &gx, &gw, Some(&gb)).unwrap())
                    .unwrap()
            },
        );
        compare(
            &format!("linear ({t}, {n_in}, {n_out}) no bias"),
            &cpu::linear(&x, &w, None).unwrap(),
            LINEAR_TOL,
            || g.read(&ops::linear(g, &gx, &gw, None).unwrap()).unwrap(),
        );
    }
}

#[test]
fn linear_naive_matches_cpu() {
    let g = gpu();
    for (seed, (t, n_in, n_out)) in [(1, 1, 1), (17, 33, 15), (16, 16, 17), (5, 3072, 768)]
        .into_iter()
        .enumerate()
    {
        let (x, w, b) = linear_case(t, n_in, n_out, seed as u64);
        let (gx, gw, gb) = (g.upload(&x), g.upload(&w), g.upload(&b));
        compare(
            &format!("linear_naive ({t}, {n_in}, {n_out})"),
            &cpu::linear(&x, &w, Some(&b)).unwrap(),
            LINEAR_TOL,
            || {
                g.read(&ops::linear_naive(g, &gx, &gw, Some(&gb)).unwrap())
                    .unwrap()
            },
        );
    }
}

/// D46: a row computed alone (matvec) has the same bits as that row inside a batch (tiled
/// matmul): both run the same chunked sum (D51). The decode-equals-recompute check (D33) depends
/// on it. The naive kernel sums serially, so it isn't part of this (D53).
#[test]
fn matmul_and_matvec_give_a_row_the_same_bits() {
    let g = gpu();
    // 777 = 3 chunks of 256 and 9 left over (one past the 8-load lookahead); 1300 = 5 chunks and
    // a partial one, so two split-matvec rounds, the second with 2 of its 4 slices empty; 3072 =
    // 3 full rounds. Up to 1024 outputs the matvec splits K, 2304 don't, 16384 run without
    // lookahead (D48, D56). 70 rows run the tiled matmul, 2-8 matvec_rows (D57).
    for (seed, (t, n_in, n_out)) in [
        (70, 777, 130),
        (70, 1300, 2304),
        (3, 1300, 70),
        (5, 3072, 768),
        (3, 768, 2304),
        (2, 100, 16384),
    ]
    .into_iter()
    .enumerate()
    {
        let (x, w, b) = linear_case(t, n_in, n_out, 50 + seed as u64);
        let (gx, gb) = (g.upload(&x), g.upload(&b));
        let gw = g.upload(&cpu::transpose(&w).unwrap());
        let batch = g
            .read(&ops::linear(g, &gx, &gw, Some(&gb)).unwrap())
            .unwrap();
        // Row i alone (matvec) and rows i.. in sub-batches of 2 and 8 (matvec_rows, D57), or
        // fewer at the end, all against the whole batch.
        for i in [0, t / 2, t - 1] {
            for len in [1, 2, 8] {
                let end = (i + len).min(t);
                let part = g.upload(&rows(&x, i, end));
                let got = g
                    .read(&ops::linear(g, &part, &gw, Some(&gb)).unwrap())
                    .unwrap();
                assert_eq!(
                    bits(&got),
                    bits(&rows(&batch, i, end)),
                    "({t}, {n_in}, {n_out}) rows {i}..{end}"
                );
            }
        }
    }
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}

/// D51 written out on the CPU with `f32::mul_add` (one rounding, like a fused `fma`): chunks of
/// 256 summed serially from 0, chunk partials added in order from 0, then the bias.
fn chunked_linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let (&[t, n_in], &[n_out, _]) = (x.shape(), w.shape()) else {
        panic!("2-D");
    };
    let mut out = Vec::with_capacity(t * n_out);
    for i in 0..t {
        let xr = &x.data()[i * n_in..(i + 1) * n_in];
        for o in 0..n_out {
            let wr = &w.data()[o * n_in..(o + 1) * n_in];
            let mut total = 0.0f32;
            for (xc, wc) in xr.chunks(256).zip(wr.chunks(256)) {
                total += xc.iter().zip(wc).fold(0.0f32, |p, (a, b)| a.mul_add(*b, p));
            }
            out.push(total + b.data()[o]);
        }
    }
    Tensor::new(&[t, n_out], out).unwrap()
}

/// Both kernels compute exactly D51's order, not just some order they share: a wrong chunk size
/// in both would pass the test above and the CPU tolerance. Like D46's check this holds because
/// this driver fuses `fma()`, which WGSL allows but doesn't require; if a driver doesn't, this
/// test says so.
#[test]
fn linear_is_bitwise_the_chunked_sum() {
    let g = gpu();
    // Row 1 runs the matvec, 2-8 rows matvec_rows (D57): 70 and 130 outputs split K four ways,
    // 2000 outputs don't, 16384 run without lookahead (D56). More rows run the tiled matmul.
    for (seed, (t, n_in, n_out)) in [
        (1, 1300, 70),
        (1, 3072, 130),
        (1, 1300, 2000),
        (1, 777, 16384),
        (3, 1300, 70),
        (8, 1300, 2000),
        (7, 777, 16384),
        (2, 3072, 768),
        (70, 255, 65),
    ]
    .into_iter()
    .enumerate()
    {
        let (x, w, b) = linear_case(t, n_in, n_out, 80 + seed as u64);
        let got = g
            .read(
                &ops::linear(
                    g,
                    &g.upload(&x),
                    &g.upload(&cpu::transpose(&w).unwrap()),
                    Some(&g.upload(&b)),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            bits(&got),
            bits(&chunked_linear(&x, &w, &b)),
            "({t}, {n_in}, {n_out})"
        );
    }
}

#[test]
fn linear_rejects_bad_shapes() {
    let g = gpu();
    // x: [2, 3] needs w: [3, out] (GPU layout) or [out, 3] (naive, CPU layout).
    let x = g.upload(&Tensor::zeros(&[2, 3]));
    let w = g.upload(&Tensor::zeros(&[2, 4]));
    assert!(ops::linear(g, &x, &w, None).is_err());
    assert!(ops::linear_naive(g, &x, &w, None).is_err());
    let w = g.upload(&Tensor::zeros(&[3, 4]));
    assert!(ops::linear(g, &x, &w, None).is_ok());
    let b = g.upload(&Tensor::zeros(&[3]));
    assert!(ops::linear(g, &x, &w, Some(&b)).is_err());
    let w = g.upload(&Tensor::zeros(&[4, 3]));
    assert!(ops::linear(g, &x, &w, None).is_err());
    assert!(ops::linear_naive(g, &x, &w, None).is_ok());
}

// ------------------------------------------------------------------ attention

/// Dot products, exp, a tree softmax and a weighted sum. Measured worst 7.8e-7 (T=200, GPT-2's
/// 12 x 64 heads); outputs near 0 make the relative error meaningless, so `abs` carries it.
const ATTENTION_TOL: Tol = Tol {
    abs: 5e-6,
    rel: 1e-4,
};

#[test]
fn attention_matches_cpu() {
    let g = gpu();
    // (T, n_head, E): T around the 64-thread workgroup and the 64-key chunk, odd head widths,
    // GPT-2's 12 x 64, and the widest head the kernel takes (64).
    let cases = [
        (1, 1, 4),
        (2, 2, 4),
        (63, 3, 12),
        (64, 3, 12),
        (65, 3, 12),
        (200, 12, 768),
        (130, 2, 128),
    ];
    for (seed, (t, h, e)) in cases.into_iter().enumerate() {
        let qkv = random(&[t, 3 * e], -2.0, 2.0, seed as u64);
        let gq = g.upload(&qkv);
        compare(
            &format!("attention T={t} H={h} E={e}"),
            &cpu::causal_attention(&qkv, h).unwrap(),
            ATTENTION_TOL,
            || g.read(&ops::causal_attention(g, &gq, h).unwrap()).unwrap(),
        );
    }
}

#[test]
fn attention_uses_the_true_score_max() {
    // As for softmax: key 5 scores ~113 for every query of head 0 (q = 4, k = 10 over d = 8
    // dims, / sqrt(8)) while the others score at most ~11. A max that misses key 5 (thread 5)
    // makes exp overflow to inf and the output NaN.
    let g = gpu();
    let (t, h, e) = (100, 2, 16);
    let mut qkv = random(&[t, 3 * e], -1.0, 1.0, 41).data().to_vec();
    for i in 0..t {
        qkv[i * 3 * e..][..8].fill(4.0); // head 0 of Q
    }
    qkv[5 * 3 * e + e..][..8].fill(10.0); // head 0 of K at position 5
    let qkv = Tensor::new(&[t, 3 * e], qkv).unwrap();
    let gq = g.upload(&qkv);
    compare(
        "attention one dominant key",
        &cpu::causal_attention(&qkv, h).unwrap(),
        ATTENTION_TOL,
        || g.read(&ops::causal_attention(g, &gq, h).unwrap()).unwrap(),
    );
}

#[test]
fn attention_uses_the_true_max_across_chunks() {
    // Key 130 (chunk 2) scores ~113 for head 0 while chunks 0 and 1 score at most ~11. Rows
    // from 130 on must rescale chunks 0 and 1 to chunk 2's max (D63); a combine that kept the
    // first chunk's max overflows exp and gives inf or NaN, one that ignored a chunk's max
    // gives wrong weights.
    let g = gpu();
    let (t, h, e) = (200, 2, 16);
    let mut qkv = random(&[t, 3 * e], -1.0, 1.0, 42).data().to_vec();
    for i in 0..t {
        qkv[i * 3 * e..][..8].fill(4.0); // head 0 of Q
    }
    qkv[130 * 3 * e + e..][..8].fill(10.0); // head 0 of K at position 130
    let qkv = Tensor::new(&[t, 3 * e], qkv).unwrap();
    let gq = g.upload(&qkv);
    compare(
        "attention dominant key in chunk 2",
        &cpu::causal_attention(&qkv, h).unwrap(),
        ATTENTION_TOL,
        || g.read(&ops::causal_attention(g, &gq, h).unwrap()).unwrap(),
    );
}

#[test]
fn attention_with_every_score_far_below_zero() {
    // Head 0: q = 8, k = -8 on its 8 dims scores exactly -512 / sqrt(8) = -181 for every key,
    // so the right output is the mean of the values (the CPU subtracts the max). A thread that
    // scores a key slot nobody staged (a row of shared memory past a short half, still zero)
    // puts 0 into the chunk's max; every real exp(s - m) then underflows, the chunk's sum is 0
    // and the output NaN. T = 100 gives rows with 1..32 keys in chunk 0 and short first halves
    // in chunk 1. The scores are equal on purpose: random ones near -181 round by ~4e-6 (one
    // f32 ulp there is 1.5e-5), which the output inherits and ATTENTION_TOL doesn't allow.
    let g = gpu();
    let (t, h, e) = (100, 2, 16);
    let mut qkv = random(&[t, 3 * e], -1.0, 1.0, 43).data().to_vec();
    for i in 0..t {
        qkv[i * 3 * e..][..8].fill(8.0); // head 0 of Q
        qkv[i * 3 * e + e..][..8].fill(-8.0); // head 0 of K
    }
    let qkv = Tensor::new(&[t, 3 * e], qkv).unwrap();
    let gq = g.upload(&qkv);
    compare(
        "attention scores far below zero",
        &cpu::causal_attention(&qkv, h).unwrap(),
        ATTENTION_TOL,
        || g.read(&ops::causal_attention(g, &gq, h).unwrap()).unwrap(),
    );
}

#[test]
fn attention_at_chunk_boundaries_and_past_1024() {
    // D64: T at the chunk edges (63, 64, 65 keys for the last row; 16 vs 17 chunks at
    // 1024/1025) and past the old 1024-position limit of D20. Every row i of a prefill sees
    // i + 1 keys, so the long cases also cover every key count below them.
    let g = gpu();
    for (seed, t) in [63, 64, 65, 1023, 1024, 1025, 1300].into_iter().enumerate() {
        let qkv = random(&[t, 3 * 16], -2.0, 2.0, 77 + seed as u64);
        let gq = g.upload(&qkv);
        compare(
            &format!("attention T={t} H=2 E=16"),
            &cpu::causal_attention(&qkv, 2).unwrap(),
            ATTENTION_TOL,
            || g.read(&ops::causal_attention(g, &gq, 2).unwrap()).unwrap(),
        );
    }
}

#[test]
fn attention_rejects_wide_heads_and_small_scratch() {
    let g = gpu();
    // One head of 65 dims: past ATTENTION_MAX_D.
    let d = ops::ATTENTION_MAX_D + 1;
    let wide = g.upload(&Tensor::zeros(&[2, 3 * d]));
    assert!(ops::causal_attention(g, &wide, 1).is_err());

    // A parts scratch one float short.
    let (t, h, e) = (65, 2, 16);
    let qkv = g.upload(&random(&[t, 3 * e], -1.0, 1.0, 5));
    let (k, v) = (g.alloc(&[t, e]), g.alloc(&[t, e]));
    ops::kv_write(g, &qkv, &k, &v, 0).unwrap();
    let shape = ops::attention_parts_shape(t, 0, e, h);
    assert_eq!(shape, [65, 2, 2, 10]);
    let n: usize = shape.iter().product();
    let short = g.alloc(&[n - 1]);
    let out = g.alloc(&[t, e]);
    let mut rec = g.rec();
    assert!(
        ops::attention_into(&mut rec, &qkv, &k, &v, 0, ops::Heads::mha(h), &short, &out).is_err()
    );
}

// ------------------------------------------------------------------ row

#[test]
fn row_copies_exactly() {
    let g = gpu();
    let x = random(&[4, 7], -1.0, 1.0, 51);
    let gx = g.upload(&x);
    for i in 0..4 {
        let r = g.read(&ops::row(g, &gx, i).unwrap()).unwrap();
        assert_eq!(r.shape(), &[1, 7]);
        assert_eq!(r.data(), &x.data()[i * 7..(i + 1) * 7]);
    }
    assert!(ops::row(g, &gx, 4).is_err());
}

#[test]
fn copy_kernel_is_exact() {
    // Sizes around whole workgroups (256 vec4s each), more than 1024 vec4s, and a cap of 2
    // workgroups so the grid-stride loop wraps several times; plus special values: a copy must
    // move bits, not numbers.
    let g = gpu();
    for n in [4, 1020, 1024, 1028, 4 * 256 * 3 + 8, 4 * 5000] {
        let x = random(&[n], -1.0, 1.0, n as u64);
        let gx = g.upload(&x);
        assert_eq!(g.read(&ops::copy(g, &gx).unwrap()).unwrap(), x, "n = {n}");
        let capped = ops::copy_with_max_groups(g, &gx, 2).unwrap();
        assert_eq!(g.read(&capped).unwrap(), x, "n = {n}, 2 workgroups");
    }
    let special = Tensor::new(
        &[2, 4],
        vec![
            f32::NAN,
            -0.0,
            f32::INFINITY,
            1e-40,
            1.0,
            -1.0,
            f32::MIN,
            f32::MAX,
        ],
    )
    .unwrap();
    let got = g.read(&ops::copy(g, &g.upload(&special)).unwrap()).unwrap();
    let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&special));
    assert!(ops::copy(g, &g.upload(&Tensor::zeros(&[6]))).is_err());
    assert!(
        g.read(&ops::copy(g, &g.upload(&Tensor::zeros(&[0]))).unwrap())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn read_peak_reads_every_element_once() {
    // Small integers, so every partial sum is exact and only a skipped or doubled element can
    // change the total. 1024 workgroups x 256 invocations = 262144 vec4s per sweep: one size
    // below that, two that wrap the grid-stride loop.
    let g = gpu();
    for n4 in [5, 262_144 + 3, 3 * 262_144] {
        let x: Vec<f32> = (0..4 * n4).map(|i| (i % 7) as f32).collect();
        let want: f64 = x.iter().map(|&v| v as f64).sum();
        let gx = g.upload(&Tensor::new(&[4 * n4], x).unwrap());
        let sums = g.read(&ops::read_peak(g, &gx).unwrap()).unwrap();
        let got: f64 = sums.data().iter().map(|&v| v as f64).sum();
        assert_eq!(got, want, "{n4} vec4s");
    }
    assert!(ops::read_peak(g, &g.upload(&Tensor::zeros(&[6]))).is_err());
    assert!(ops::read_peak(g, &g.upload(&Tensor::zeros(&[0]))).is_err());
}

#[test]
fn fma_peak_runs() {
    let g = gpu();
    let out = g.read(&ops::fma_peak(g, 2, 10).unwrap()).unwrap();
    assert_eq!(out.len(), 2 * 256);
    assert!(out.data().iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(ops::fma_peak(g, 0, 10).is_err());
}

// ------------------------------------------------------------------ KV cache (M4)

/// Rows `lo..hi` of a 2-D host tensor.
fn rows(x: &Tensor, lo: usize, hi: usize) -> Tensor {
    let c = x.shape()[1];
    Tensor::new(&[hi - lo, c], x.data()[lo * c..hi * c].to_vec()).unwrap()
}

#[test]
fn kv_write_fills_exactly_its_rows() {
    // The cache starts full of a sentinel; rows outside start..start+T must keep it (D9: a
    // stray write past the intended rows would otherwise go unseen).
    let g = gpu();
    let (t, e, n_ctx, start) = (5, 12, 9, 2);
    let qkv = random(&[t, 3 * e], -1.0, 1.0, 61);
    let sentinel = Tensor::new(&[n_ctx, e], vec![7.5; n_ctx * e]).unwrap();
    let (k, v) = (g.upload(&sentinel), g.upload(&sentinel));
    ops::kv_write(g, &g.upload(&qkv), &k, &v, start).unwrap();
    let (k, v) = (g.read(&k).unwrap(), g.read(&v).unwrap());
    for r in 0..n_ctx {
        for c in 0..e {
            let (want_k, want_v) = if (start..start + t).contains(&r) {
                let q = qkv.data();
                let base = (r - start) * 3 * e;
                (q[base + e + c], q[base + 2 * e + c])
            } else {
                (7.5, 7.5)
            };
            assert_eq!(k.data()[r * e + c], want_k, "k[{r}, {c}]");
            assert_eq!(v.data()[r * e + c], want_v, "v[{r}, {c}]");
        }
    }
    // Past the end of the cache.
    let (k, v) = (g.upload(&sentinel), g.upload(&sentinel));
    assert!(ops::kv_write(g, &g.upload(&qkv), &k, &v, 5).is_err());
}

#[test]
fn cached_attention_continues_exactly() {
    // Cache a prefix, then attend the rest at an offset: the rows must match the CPU and be
    // bitwise equal to attending the whole sequence at once (D31, D33).
    let g = gpu();
    let (t, h, e) = (70, 3, 12);
    let qkv = random(&[t, 3 * e], -2.0, 2.0, 62);
    let want = cpu::causal_attention(&qkv, h).unwrap();
    let whole = g
        .read(&ops::causal_attention(g, &g.upload(&qkv), h).unwrap())
        .unwrap();
    for split in [1, 33, 64, 69] {
        let cache = Tensor::zeros(&[t + 5, e]); // longer than needed, like a real cache
        let (k, v) = (g.upload(&cache), g.upload(&cache));
        let first = g.upload(&rows(&qkv, 0, split));
        let rest = g.upload(&rows(&qkv, split, t));
        ops::kv_write(g, &first, &k, &v, 0).unwrap();
        ops::kv_write(g, &rest, &k, &v, split).unwrap();
        let got = g
            .read(&ops::attention_cached(g, &rest, &k, &v, split, h).unwrap())
            .unwrap();
        let tail = rows(&want, split, t);
        check(got.data(), tail.data(), ATTENTION_TOL)
            .unwrap_or_else(|m| panic!("split {split}: {m}"));
        let whole_tail = rows(&whole, split, t);
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got.data()), bits(whole_tail.data()), "split {split}");
    }
    // Asking for positions the cache doesn't have.
    let small = g.upload(&Tensor::zeros(&[10, e]));
    let q = g.upload(&rows(&qkv, 0, 3));
    assert!(ops::attention_cached(g, &q, &small, &small, 8, h).is_err());
}

#[test]
fn embed_at_an_offset() {
    let g = gpu();
    let (v, e, n_ctx) = (37, 12, 16);
    let wte = random(&[v, e], -1.0, 1.0, 63);
    let wpe = random(&[n_ctx, e], -1.0, 1.0, 64);
    let ids = [3, 1, 4, 1, 5, 9, 2, 6];
    let want = cpu::embed(&wte, &wpe, &ids).unwrap();
    let (gwte, gwpe) = (g.upload(&cpu::transpose(&wte).unwrap()), g.upload(&wpe));
    let got = g
        .read(&ops::embed(g, &gwte, &gwpe, &ids[5..], 5).unwrap())
        .unwrap();
    assert_eq!(got.data(), rows(&want, 5, 8).data());
    assert!(ops::embed(g, &gwte, &gwpe, &ids[..2], 15).is_err());
}

// ------------------------------------------------------------------ outputs into a workspace

/// Run `record` (one `_into` op writing `out`) with `out` a view of a buffer `extra` floats
/// larger, pre-filled with a sentinel (D9, D64). The extra floats must keep the sentinel, and
/// the view must hold exactly what `want` (the one-op form) gives.
fn into_oversized(
    label: &str,
    shape: &[usize],
    want: &Tensor,
    record: impl FnOnce(&mut ember::gpu::Rec, &ember::GpuTensor) -> ember::Result<()>,
) {
    const SENTINEL: f32 = -7777.25;
    const EXTRA: usize = 37;
    let g = gpu();
    let n: usize = shape.iter().product();
    let big = g.upload(&Tensor::new(&[n + EXTRA], vec![SENTINEL; n + EXTRA]).unwrap());
    let out = big.view(shape).unwrap();
    let mut rec = g.rec();
    record(&mut rec, &out).unwrap();
    rec.submit();
    let all = g.read(&big).unwrap();
    let view: Vec<u32> = all.data()[..n].iter().map(|x| x.to_bits()).collect();
    assert_eq!(view, bits(want), "{label}: view");
    assert!(
        all.data()[n..].iter().all(|&x| x == SENTINEL),
        "{label}: wrote past the view"
    );
}

#[test]
fn into_ops_write_only_their_view() {
    let g = gpu();
    let (t, e) = (3, 12);
    let x = g.upload(&random(&[t, e], -2.0, 2.0, 70));
    let y = g.upload(&random(&[t, e], -2.0, 2.0, 71));
    let want = g.read(&ops::add(g, &x, &y).unwrap()).unwrap();
    into_oversized("add", &[t, e], &want, |r, o| ops::add_into(r, &x, &y, o));
    let want = g.read(&ops::gelu(g, &x).unwrap()).unwrap();
    into_oversized("gelu", &[t, e], &want, |r, o| ops::gelu_into(r, &x, o));

    let (gain, bias) = (
        g.upload(&random(&[e], 0.5, 1.5, 72)),
        g.upload(&random(&[e], -0.5, 0.5, 73)),
    );
    let want = g
        .read(&ops::layer_norm(g, &x, &gain, &bias, 1e-5).unwrap())
        .unwrap();
    into_oversized("layer_norm", &[t, e], &want, |r, o| {
        ops::layer_norm_into(r, &x, &gain, &bias, 1e-5, o)
    });

    let (v, n_ctx) = (17, 9);
    let wte_t = g.upload(&random(&[e, v], -1.0, 1.0, 74));
    let wpe = g.upload(&random(&[n_ctx, e], -1.0, 1.0, 75));
    let ids = [3u32, 16, 0];
    let want = g
        .read(&ops::embed(g, &wte_t, &wpe, &ids, 2).unwrap())
        .unwrap();
    let ids_buf = g.upload_u32(&ids);
    into_oversized("embed", &[t, e], &want, |r, o| {
        ops::embed_into(r, &wte_t, &wpe, &ids, &ids_buf, 2, o)
    });

    let want = g.read(&ops::row(g, &x, 1).unwrap()).unwrap();
    into_oversized("row", &[1, e], &want, |r, o| ops::row_into(r, &x, 1, o));

    let h = 3;
    let qkv = g.upload(&random(&[t, 3 * e], -1.0, 1.0, 76));
    let (k, vc) = (g.alloc(&[n_ctx, e]), g.alloc(&[n_ctx, e]));
    ops::kv_write(g, &qkv, &k, &vc, 4).unwrap();
    let want = g
        .read(&ops::attention_cached(g, &qkv, &k, &vc, 4, h).unwrap())
        .unwrap();
    let parts = g.alloc(&ops::attention_parts_shape(t, 4, e, h));
    into_oversized("attention", &[t, e], &want, |r, o| {
        ops::attention_into(r, &qkv, &k, &vc, 4, ops::Heads::mha(h), &parts, o)
    });

    // Every linear kernel: matvec split / plain / wide, matvec_rows likewise, matmul; with and
    // without a bias.
    for (seed, (t, n_in, n_out)) in [
        (1, 300, 70),
        (1, 300, 2304),
        (1, 20, 16384),
        (5, 300, 70),
        (5, 300, 2304),
        (2, 20, 16384),
        (70, 300, 130),
    ]
    .into_iter()
    .enumerate()
    {
        let (xc, w, b) = linear_case(t, n_in, n_out, 80 + seed as u64);
        let (gx, gb) = (g.upload(&xc), g.upload(&b));
        let gw = g.upload(&cpu::transpose(&w).unwrap());
        for bias in [Some(&gb), None] {
            let want = g.read(&ops::linear(g, &gx, &gw, bias).unwrap()).unwrap();
            into_oversized(
                &format!("linear ({t}, {n_in}, {n_out}) bias {}", bias.is_some()),
                &[t, n_out],
                &want,
                |r, o| ops::linear_into(r, &gx, &gw, bias, ops::Epilogue::None, o),
            );
        }
    }
}

#[test]
fn epilogues_equal_the_separate_kernels_bitwise() {
    // D62: GELU fused into a linear kernel is gelu.wgsl's function on the same f32, and the
    // fused residual is the add kernel's `res + y`; every kernel configuration, with and
    // without a bias.
    let g = gpu();
    for (seed, (t, n_in, n_out)) in [
        (1, 300, 70),
        (1, 300, 2304),
        (1, 20, 16384),
        (5, 300, 70),
        (5, 300, 2304),
        (2, 20, 16384),
        (70, 300, 130),
    ]
    .into_iter()
    .enumerate()
    {
        let (x, w, b) = linear_case(t, n_in, n_out, 90 + seed as u64);
        let (gx, gb) = (g.upload(&x), g.upload(&b));
        let gw = g.upload(&cpu::transpose(&w).unwrap());
        let res = g.upload(&random(&[t, n_out], -3.0, 3.0, 99));
        for bias in [Some(&gb), None] {
            let label = format!("({t}, {n_in}, {n_out}) bias {}", bias.is_some());
            let y = ops::linear(g, &gx, &gw, bias).unwrap();
            let fused = |ep| {
                let out = g.alloc(&[t, n_out]);
                let mut rec = g.rec();
                ops::linear_into(&mut rec, &gx, &gw, bias, ep, &out).unwrap();
                rec.submit();
                g.read(&out).unwrap()
            };
            let want = g.read(&ops::gelu(g, &y).unwrap()).unwrap();
            assert_eq!(
                bits(&fused(ops::Epilogue::Gelu)),
                bits(&want),
                "gelu {label}"
            );
            let want = g.read(&ops::add(g, &res, &y).unwrap()).unwrap();
            let got = fused(ops::Epilogue::Residual(&res));
            assert_eq!(bits(&got), bits(&want), "residual {label}");
        }
    }
}

#[test]
fn residual_epilogue_rejects_bad_buffers() {
    let g = gpu();
    let x = g.upload(&random(&[2, 8], -1.0, 1.0, 1));
    let w = g.upload(&random(&[8, 4], -1.0, 1.0, 2));
    let out = g.alloc(&[2, 4]);
    let mut rec = g.rec();
    let wrong = g.alloc(&[2, 5]);
    let ep = ops::Epilogue::Residual;
    assert!(ops::linear_into(&mut rec, &x, &w, None, ep(&wrong), &out).is_err());
    // In place would bind one buffer read-only and read-write at once.
    assert!(ops::linear_into(&mut rec, &x, &w, None, ep(&out), &out).is_err());
}

// ------------------------------------------------------------------ M8: Llama-family ops

/// As LayerNorm: the CPU computes in f64, the GPU in f32 with a tree sum.
const RMS_NORM_TOL: Tol = Tol {
    abs: 2e-5,
    rel: 1e-5,
};

#[test]
fn rms_norm_matches_cpu() {
    let g = gpu();
    let shapes: [[usize; 2]; 6] = [[1, 1], [2, 255], [2, 256], [3, 257], [5, 576], [2, 1536]];
    for (seed, shape) in shapes.into_iter().enumerate() {
        let cols = shape[1];
        let x = random(&shape, -20.0, 20.0, 300 + seed as u64);
        let gain = random(&[cols], 0.5, 1.5, 400 + seed as u64);
        let (gx, gg) = (g.upload(&x), g.upload(&gain));
        compare(
            &format!("rms_norm {shape:?}"),
            &cpu::rms_norm(&x, &gain, 1e-5).unwrap(),
            RMS_NORM_TOL,
            || g.read(&ops::rms_norm(g, &gx, &gg, 1e-5).unwrap()).unwrap(),
        );
    }
}

/// exp differs between libm and the driver by a few ulp, as tanh does for GELU.
const SILU_TOL: Tol = Tol {
    abs: 1e-6,
    rel: 1e-5,
};

#[test]
fn silu_mul_matches_cpu() {
    let g = gpu();
    for (seed, (t, f)) in [(1, 1), (1, 255), (3, 257), (7, 1536)]
        .into_iter()
        .enumerate()
    {
        let gu = random(&[t, 2 * f], -12.0, 12.0, 500 + seed as u64);
        let ggu = g.upload(&gu);
        compare(
            &format!("silu_mul T={t} F={f}"),
            &cpu::silu_mul(&gu).unwrap(),
            SILU_TOL,
            || g.read(&ops::silu_mul(g, &ggu).unwrap()).unwrap(),
        );
    }
}

/// Two products and a sum per output from the same f32 inputs: a driver may fuse them into an
/// fma, which rounds once instead of twice.
const ROPE_TOL: Tol = Tol {
    abs: 1e-6,
    rel: 1e-6,
};

#[test]
fn rope_matches_cpu() {
    // SmolLM2's shape (9 query + 3 key heads rotated, 3 value heads not, d = 64, theta 1e5)
    // and a small odd one, at positions from 0 and from 1000 (large angles).
    let g = gpu();
    let cases = [
        (1, 12, 15, 64, 0, 100000.0),
        (7, 12, 15, 64, 0, 100000.0),
        (5, 12, 15, 64, 1000, 100000.0),
        (3, 2, 3, 4, 7, 10000.0),
    ];
    for (seed, (t, n_rot, heads, d, start, theta)) in cases.into_iter().enumerate() {
        let (cos, sin) = cpu::rope_tables(start + t, d, theta);
        let (gc, gs) = (g.upload(&cos), g.upload(&sin));
        let x = random(&[t, heads * d], -3.0, 3.0, 600 + seed as u64);
        let gx = g.upload(&x);
        let run = || {
            g.read(&ops::rope(g, &gx, n_rot, d, start, &gc, &gs).unwrap())
                .unwrap()
        };
        compare(
            &format!("rope T={t} rot={n_rot}/{heads} d={d} from {start}"),
            &cpu::rope(&x, n_rot, d, start, &cos, &sin).unwrap(),
            ROPE_TOL,
            run,
        );
        // The heads past n_rot (the values) are copied bit for bit.
        let out = run();
        for r in 0..t {
            let tail = |v: &[f32]| v[r * heads * d + n_rot * d..(r + 1) * heads * d].to_vec();
            assert_eq!(tail(out.data()), tail(x.data()), "values rotated");
        }
        // One position past the table is refused (the kernel would read past it).
        assert!(ops::rope(g, &gx, n_rot, d, start + 1, &gc, &gs).is_err());
    }
}

#[test]
fn gqa_attention_matches_cpu() {
    // Grouped heads at chunk boundaries: SmolLM2's 9 over 3 (d = 64), 6 over 2, 4 over 1, and
    // 4 over 4 (plain multi-head through the same path).
    let g = gpu();
    let cases = [
        (1, 9, 3, 64),
        (63, 9, 3, 64),
        (65, 9, 3, 64),
        (130, 6, 2, 16),
        (70, 4, 1, 8),
        (20, 4, 4, 8),
    ];
    for (seed, (t, h, kv, d)) in cases.into_iter().enumerate() {
        let qkv = random(&[t, (h + 2 * kv) * d], -2.0, 2.0, 700 + seed as u64);
        let gq = g.upload(&qkv);
        let heads = ops::Heads { q: h, kv };
        compare(
            &format!("gqa attention T={t} {h}/{kv} d={d}"),
            &cpu::causal_attention_gqa(&qkv, h, kv).unwrap(),
            ATTENTION_TOL,
            || {
                g.read(&ops::causal_attention_gqa(g, &gq, heads).unwrap())
                    .unwrap()
            },
        );
    }
}

#[test]
fn gqa_cached_attention_continues_exactly() {
    // D33 with grouped heads: prefill 50 rows, then the other 20 one at a time into the same
    // cache; every row must have the bits of attending over all 70 at once.
    let g = gpu();
    let (t, h, kv, d) = (70, 9, 3, 64);
    let w = (h + 2 * kv) * d;
    let heads = ops::Heads { q: h, kv };
    let qkv = random(&[t, w], -2.0, 2.0, 800);
    let full = g
        .read(&ops::causal_attention_gqa(g, &g.upload(&qkv), heads).unwrap())
        .unwrap();
    let (k, v) = (g.alloc(&[t, kv * d]), g.alloc(&[t, kv * d]));
    let rows = |a: usize, b: usize| {
        g.upload(&Tensor::new(&[b - a, w], qkv.data()[a * w..b * w].to_vec()).unwrap())
    };
    let mut got = Vec::new();
    let mut at = 0;
    for end in std::iter::once(50).chain(51..=t) {
        let part = rows(at, end);
        let mut rec = g.rec();
        ops::kv_write_into(&mut rec, &part, &k, &v, at, heads).unwrap();
        rec.submit();
        let out = ops::attention_cached_gqa(g, &part, &k, &v, at, heads).unwrap();
        got.extend_from_slice(g.read(&out).unwrap().data());
        at = end;
    }
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(full.data()));
}

#[test]
fn gqa_shapes_are_checked() {
    let g = gpu();
    let qkv = g.upload(&Tensor::zeros(&[4, (9 + 6) * 64]));
    // 9 query heads don't group over 2 key/value heads; the row doesn't split into 9 + 2 x 4.
    assert!(ops::causal_attention_gqa(g, &qkv, ops::Heads { q: 9, kv: 2 }).is_err());
    assert!(ops::causal_attention_gqa(g, &qkv, ops::Heads { q: 7, kv: 7 }).is_err());
    // A cache as wide as the queries (the GPT-2 layout) is the wrong width for 3 kv heads.
    let wide = g.alloc(&[4, 9 * 64]);
    let mut rec = g.rec();
    let heads = ops::Heads { q: 9, kv: 3 };
    assert!(ops::kv_write_into(&mut rec, &qkv, &wide, &wide, 0, heads).is_err());
}
