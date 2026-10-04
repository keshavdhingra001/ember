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
            || g.read(&ops::embed(g, &gwte, &gwpe, ids).unwrap()).unwrap(),
        );
    }
    assert!(ops::embed(g, &gwte, &gwpe, &[37]).is_err());
    assert!(ops::embed(g, &gwte, &gwpe, &[0; 17]).is_err());
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

// ------------------------------------------------------------------ layer_norm

/// Dominated by the *CPU's* error, not the GPU's: with inputs around 100 and a spread of ~3,
/// the serial f32 sum gives a mean a few ulp off, and the error is divided by the std. Against a
/// float64 LayerNorm (measured 2026-10-05, 3072 columns) the CPU was 5.2e-5 off and the GPU's
/// tree sum 4.4e-6. See D22.
const LAYER_NORM_TOL: Tol = Tol {
    abs: 1e-4,
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
