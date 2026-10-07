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
    let (gwte, gwpe) = (g.upload(&wte), g.upload(&wpe));
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

/// Same summation order as the CPU, so the nonzero error (measured worst 2.9e-6 at n_in = 3072)
/// is fused multiply-add: the GPU rounds `acc + x * w` once, the CPU twice.
const LINEAR_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};

#[test]
fn linear_matches_cpu() {
    let g = gpu();
    // (T, in, out): single elements, workgroup edges (16), odd sizes, and GPT-2's shapes.
    let shapes = [
        (1, 1, 1),
        (1, 16, 16),
        (17, 33, 15),
        (16, 16, 17),
        (3, 768, 2304),
        (5, 3072, 768),
        (2, 768, 50257),
    ];
    for (seed, (t, n_in, n_out)) in shapes.into_iter().enumerate() {
        let s = seed as u64;
        let x = random(&[t, n_in], -1.0, 1.0, s);
        let w = random(&[n_out, n_in], -0.1, 0.1, 100 + s);
        let b = random(&[n_out], -1.0, 1.0, 200 + s);
        let (gx, gw, gb) = (g.upload(&x), g.upload(&w), g.upload(&b));
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
fn linear_rejects_bad_shapes() {
    let g = gpu();
    let x = g.upload(&Tensor::zeros(&[2, 3]));
    let w = g.upload(&Tensor::zeros(&[4, 2]));
    assert!(ops::linear(g, &x, &w, None).is_err());
    let w = g.upload(&Tensor::zeros(&[4, 3]));
    let b = g.upload(&Tensor::zeros(&[3]));
    assert!(ops::linear(g, &x, &w, Some(&b)).is_err());
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
    // (T, n_head, E): T around the 64-thread workgroup, odd head widths, GPT-2's 12 x 64.
    let cases = [
        (1, 1, 4),
        (2, 2, 4),
        (63, 3, 12),
        (64, 3, 12),
        (65, 3, 12),
        (200, 12, 768),
        (130, 1, 130),
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
fn attention_at_the_context_limit() {
    let g = gpu();
    let max = ops::ATTENTION_MAX_CTX;
    let qkv = random(&[max, 3 * 16], -2.0, 2.0, 77);
    let gq = g.upload(&qkv);
    compare(
        &format!("attention T={max} H=2 E=16"),
        &cpu::causal_attention(&qkv, 2).unwrap(),
        ATTENTION_TOL,
        || g.read(&ops::causal_attention(g, &gq, 2).unwrap()).unwrap(),
    );
    let too_long = g.upload(&Tensor::zeros(&[max + 1, 48]));
    assert!(ops::causal_attention(g, &too_long, 2).is_err());
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
    let (gwte, gwpe) = (g.upload(&wte), g.upload(&wpe));
    let got = g
        .read(&ops::embed(g, &gwte, &gwpe, &ids[5..], 5).unwrap())
        .unwrap();
    assert_eq!(got.data(), rows(&want, 5, 8).data());
    assert!(ops::embed(g, &gwte, &gwpe, &ids[..2], 15).is_err());
}
