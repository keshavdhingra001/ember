//! The CPU reference ops: plain, obviously correct Rust. This is the oracle every GPU kernel
//! is differential-tested against (D3). Clarity beats speed here, always.

use crate::error::{Error, Result};
use crate::shape;
use crate::tensor::Tensor;

/// Elementwise `a + b`. Shapes must match exactly (no broadcasting yet).
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    shape::same("add", a.shape(), b.shape())?;
    let data = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    Tensor::new(a.shape(), data)
}

/// `y = x @ w^T + b`: `x: [T, in]`, `w: [out, in]` (D7), `b: [out]` or none. The naive triple
/// loop (D13); each output is one dot product summed in index order, then the bias is added.
///
/// Big products are split across threads in contiguous chunks of the output. That changes who
/// computes an element, never how: every element is still the same serial sum, so the bits
/// are identical to the single-threaded loop (D4) regardless of the thread count.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let (t, n_in, n_out) = shape::linear(x.shape(), w.shape(), b.map(Tensor::shape))?;
    let (x, w, b) = (x.data(), w.data(), b.map(Tensor::data));
    // Output element `idx` is row i = idx / n_out, feature o = idx % n_out.
    let element = |idx: usize| {
        let (i, o) = (idx / n_out, idx % n_out);
        let row = &x[i * n_in..(i + 1) * n_in];
        let w_row = &w[o * n_in..(o + 1) * n_in];
        let mut acc = 0.0f32;
        for k in 0..n_in {
            acc += row[k] * w_row[k];
        }
        acc + b.map_or(0.0, |b| b[o])
    };
    let mut out = vec![0.0; t * n_out];
    // Below ~1M multiply-adds a thread costs more to start than it saves.
    let threads = if t * n_out * n_in < 1 << 20 {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    };
    let chunk = out.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (j, y) in part.iter_mut().enumerate() {
                    *y = element(c * chunk + j);
                }
            });
        }
    });
    Tensor::new(&[t, n_out], out)
}

/// LayerNorm over the last dimension of `x: [T, E]`: each row is shifted to mean 0, scaled to
/// variance 1, then multiplied by `gain` and shifted by `bias` (both `[E]`). The variance is
/// the biased one (divide by E), as in PyTorch.
///
/// Two passes (mean first, then the mean of squared deviations) instead of `E[x^2] - E[x]^2`:
/// GPT-2's residual stream has a few very large features, and subtracting two nearly equal
/// large numbers would cancel away most of the variance's digits.
///
/// Computed in f64 and rounded to f32 once per output (D23). A serial f32 sum's rounding error
/// grows with the row length, and the mean's error is then divided by the standard deviation;
/// in f32 this oracle was less accurate than the GPU's tree sum (D22). The oracle should be the
/// most accurate thing in the room.
pub fn layer_norm(x: &Tensor, gain: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let (t, e) = shape::layer_norm(x.shape(), gain.shape(), bias.shape())?;
    let (g, b) = (gain.data(), bias.data());
    let mut out = Vec::with_capacity(t * e);
    for row in x.data().chunks_exact(e) {
        let n = e as f64;
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        out.extend(
            row.iter()
                .enumerate()
                .map(|(j, &v)| ((v as f64 - mean) * inv_std * g[j] as f64 + b[j] as f64) as f32),
        );
    }
    Tensor::new(x.shape(), out)
}

/// GPT-2's GELU, the tanh approximation ("gelu_new"):
/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`. It differs from the exact
/// `x Phi(x)` by up to about 5e-4 (near |x| = 2), enough to change logits, so it must be this one.
pub fn gelu_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// Elementwise GELU, any shape.
pub fn gelu(x: &Tensor) -> Tensor {
    Tensor::new(
        x.shape(),
        x.data().iter().map(|&v| gelu_scalar(v)).collect(),
    )
    .unwrap()
}

/// Softmax of one row, in place. Subtracting the row max first changes nothing mathematically
/// (it cancels between numerator and denominator) but keeps `exp` from overflowing: exp(89)
/// is already infinite in f32, and attention scores can be that large.
pub fn softmax_in_place(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

/// Softmax over the last dimension of `x: [R, C]`, each row independently.
pub fn softmax_rows(x: &Tensor) -> Result<Tensor> {
    let &[_, c] = x.shape() else {
        return Err(Error::Shape(format!(
            "softmax_rows: x {:?} must be 2-D",
            x.shape()
        )));
    };
    let mut out = x.data().to_vec();
    if c > 0 {
        out.chunks_exact_mut(c).for_each(softmax_in_place);
    }
    Tensor::new(x.shape(), out)
}

/// Causal multi-head self-attention, from the fused projection `qkv: [T, 3E]` (columns `0..E`
/// are Q, `E..2E` K, `2E..3E` V; head `h` owns columns `h*D..(h+1)*D` of each, D = E / H).
/// Returns the heads' outputs side by side, `[T, E]`, before the output projection.
///
/// For each head and each position t: score every position j <= t by `q_t . k_j / sqrt(D)`,
/// softmax the scores, and average the `v_j` with those weights. Positions after t are not
/// masked with -inf: they are never scored at all, which is the same thing.
pub fn causal_attention(qkv: &Tensor, n_head: usize) -> Result<Tensor> {
    causal_attention_gqa(qkv, n_head, n_head)
}

/// Causal attention with grouped key/value heads (D71): `qkv: [T, (H + 2 KV) d]` holds H query
/// heads, then KV key heads, then KV value heads; query head h attends with key/value head
/// `h / (H / KV)`. Returns `[T, H d]`. With KV = H this is GPT-2's attention, the same loops.
pub fn causal_attention_gqa(qkv: &Tensor, n_head: usize, n_kv_head: usize) -> Result<Tensor> {
    let (t, d) = shape::qkv_gqa("attention", qkv.shape(), n_head, n_kv_head)?;
    let width = (n_head + 2 * n_kv_head) * d;
    let (k_at, v_at) = (n_head * d, (n_head + n_kv_head) * d);
    let group = n_head / n_kv_head;
    let e = n_head * d;
    let scale = 1.0 / (d as f32).sqrt();
    let x = qkv.data();
    // Row i's slice starting at column `col`.
    let at = |i: usize, col: usize| &x[i * width + col..][..d];
    let mut out = vec![0.0; t * e];
    let mut scores = Vec::with_capacity(t);
    for h in 0..n_head {
        let kv = h / group;
        for i in 0..t {
            let q = at(i, h * d);
            scores.clear();
            for j in 0..=i {
                let k = at(j, k_at + kv * d);
                let mut dot = 0.0f32;
                for c in 0..d {
                    dot += q[c] * k[c];
                }
                scores.push(dot * scale);
            }
            softmax_in_place(&mut scores);
            let y = &mut out[i * e + h * d..][..d];
            for (j, &p) in scores.iter().enumerate() {
                let v = at(j, v_at + kv * d);
                for c in 0..d {
                    y[c] += p * v[c];
                }
            }
        }
    }
    Tensor::new(&[t, e], out)
}

/// RMSNorm over the last dimension of `x: [T, E]`: each row divided by its root mean square,
/// `sqrt(mean(x^2) + eps)`, then multiplied by `gain: [E]`. No mean subtraction and no bias.
/// In f64, rounded once per output, for the reason given at `layer_norm` (D23).
pub fn rms_norm(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
    let (t, e) = shape::rms_norm(x.shape(), gain.shape())?;
    let g = gain.data();
    let mut out = Vec::with_capacity(t * e);
    for row in x.data().chunks_exact(e) {
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / e as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        out.extend(
            row.iter()
                .zip(g)
                .map(|(&v, &g)| (v as f64 * inv * g as f64) as f32),
        );
    }
    Tensor::new(x.shape(), out)
}

/// SwiGLU's gate: `gu: [T, 2F]` holds the gate projection in columns `0..F` and the up
/// projection in `F..2F` (D72); returns `silu(gate) * up`, `[T, F]`, with
/// `silu(x) = x / (1 + exp(-x))`.
pub fn silu_mul(gu: &Tensor) -> Result<Tensor> {
    let &[t, two_f] = gu.shape() else {
        return Err(Error::Shape(format!(
            "silu_mul: {:?} must be 2-D",
            gu.shape()
        )));
    };
    if !two_f.is_multiple_of(2) {
        return Err(Error::Shape(format!(
            "silu_mul: {:?} has an odd width",
            gu.shape()
        )));
    }
    let f = two_f / 2;
    let mut out = Vec::with_capacity(t * f);
    for row in gu.data().chunks_exact(two_f) {
        let (gate, up) = row.split_at(f);
        out.extend(gate.iter().zip(up).map(|(&g, &u)| silu(g) * u));
    }
    Tensor::new(&[t, f], out)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE's tables (D70): `cos` and `sin`, each `[n_pos, d/2]`, of the angle
/// `pos * theta^(-2i/d)` for pair i. Computed in f64 and rounded once, so the CPU and the GPU
/// rotate by the same f32 numbers.
pub fn rope_tables(n_pos: usize, d: usize, theta: f64) -> (Tensor, Tensor) {
    let half = d / 2;
    let mut cos = Vec::with_capacity(n_pos * half);
    let mut sin = Vec::with_capacity(n_pos * half);
    for pos in 0..n_pos {
        for i in 0..half {
            let angle = pos as f64 * theta.powf(-2.0 * i as f64 / d as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let t = |v| Tensor::new(&[n_pos, half], v).expect("n_pos * d/2 elements");
    (t(cos), t(sin))
}

/// Rotary position embedding (D70) on the first `n_rot` heads of every row of `x: [T, W]`
/// (head width d; the rest of the row is copied): row t sits at position `start + t`, and
/// dimension i of a head turns with dimension i + d/2,
/// `(a, b) -> (a cos - b sin, b cos + a sin)`. In the fused q|k|v row the query heads and then
/// the key heads come first, so `n_rot = H + KV` rotates exactly q and k.
pub fn rope(
    x: &Tensor,
    n_rot: usize,
    d: usize,
    start: usize,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let (t, w) = shape::rope(x.shape(), n_rot, d, start, cos.shape(), sin.shape())?;
    let half = d / 2;
    let mut out = x.data().to_vec();
    for (r, row) in out.chunks_exact_mut(w).enumerate() {
        let pos = start + r;
        let (c, s) = (
            &cos.data()[pos * half..][..half],
            &sin.data()[pos * half..][..half],
        );
        for head in row[..n_rot * d].chunks_exact_mut(d) {
            for i in 0..half {
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * c[i] - b * s[i];
                head[i + half] = b * c[i] + a * s[i];
            }
        }
    }
    debug_assert_eq!(out.len(), t * w);
    Tensor::new(x.shape(), out)
}

/// Input embedding: row t is `wte[ids[t]] + wpe[t]` (token embedding plus learned position
/// embedding). `wte: [V, E]`, `wpe: [n_ctx, E]`.
pub fn embed(wte: &Tensor, wpe: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let e = shape::embed(wte.shape(), wpe.shape(), ids, 0)?;
    let mut out = Vec::with_capacity(ids.len() * e);
    for (pos, &id) in ids.iter().enumerate() {
        let tok = &wte.data()[id as usize * e..][..e];
        let p = &wpe.data()[pos * e..][..e];
        out.extend(tok.iter().zip(p).map(|(a, b)| a + b));
    }
    Tensor::new(&[ids.len(), e], out)
}

/// Index of the largest value; the first one on ties, so greedy decoding is deterministic (D4).
/// `None` for an empty row or one containing NaN: NaN logits mean the model computed garbage,
/// and every comparison with NaN is false, so a plain max loop would silently pick a token.
pub fn argmax(row: &[f32]) -> Option<usize> {
    if row.is_empty() || row.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    Some(best)
}

/// Matrix transpose: `[r, c]` -> `[c, r]`.
pub fn transpose(a: &Tensor) -> Result<Tensor> {
    let &[r, c] = a.shape() else {
        return Err(Error::Shape(format!(
            "transpose: need 2-D, got {:?}",
            a.shape()
        )));
    };
    let x = a.data();
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = x[i * c + j];
        }
    }
    Tensor::new(&[c, r], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_small() {
        let a = Tensor::new(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::new(&[3], vec![10.0, 20.0, 30.0]).unwrap();
        assert_eq!(add(&a, &b).unwrap().data(), &[11.0, 22.0, 33.0]);
    }

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape, data.to_vec()).unwrap()
    }

    fn close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "[{i}]: got {g}, want {w} (tol {tol})");
        }
    }

    #[test]
    fn linear_by_hand() {
        // x [2, 3], w [out=2, in=3]: y[i][o] = x[i] . w[o] + b[o]
        let x = t(&[2, 3], &[1.0, 2.0, 3.0, -1.0, 0.0, 0.5]);
        let w = t(&[2, 3], &[1.0, 0.0, -1.0, 0.5, 0.5, 0.5]);
        let b = t(&[2], &[10.0, 20.0]);
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.data(), &[8.0, 23.0, 8.5, 19.75]);
        assert_eq!(
            linear(&x, &w, None).unwrap().data(),
            &[-2.0, 3.0, -1.5, -0.25]
        );
        // in = 3 vs w's in = 2; a bias of the wrong length.
        assert!(linear(&x, &t(&[3, 2], &[0.0; 6]), None).is_err());
        assert!(linear(&x, &w, Some(&t(&[3], &[0.0; 3]))).is_err());
    }

    #[test]
    fn threaded_linear_is_bitwise_the_serial_loop() {
        // Big enough to take the threaded path; compare against the textbook loop.
        let mut rng = crate::rng::Rng::new(9);
        let (t, n_in, n_out) = (5, 300, 701);
        let x = Tensor::new(&[t, n_in], rng.vec(t * n_in, -1.0, 1.0)).unwrap();
        let w = Tensor::new(&[n_out, n_in], rng.vec(n_out * n_in, -1.0, 1.0)).unwrap();
        let b = Tensor::new(&[n_out], rng.vec(n_out, -1.0, 1.0)).unwrap();
        assert!(t * n_out * n_in >= 1 << 20);
        let got = linear(&x, &w, Some(&b)).unwrap();
        for i in 0..t {
            for o in 0..n_out {
                let mut acc = 0.0f32;
                for k in 0..n_in {
                    acc += x.data()[i * n_in + k] * w.data()[o * n_in + k];
                }
                let want = acc + b.data()[o];
                assert_eq!(
                    got.data()[i * n_out + o].to_bits(),
                    want.to_bits(),
                    "({i}, {o})"
                );
            }
        }
    }

    #[test]
    fn layer_norm_by_hand() {
        let x = t(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]);
        let ones = t(&[4], &[1.0; 4]);
        let zeros = t(&[4], &[0.0; 4]);
        let y = layer_norm(&x, &ones, &zeros, 1e-5).unwrap();
        // mean 2.5, biased variance 1.25; a constant row has variance 0 and normalizes to 0
        // (eps keeps that from being 0 / 0).
        let n = [-1.341_635_4, -0.447_211_8, 0.447_211_8, 1.341_635_4];
        close(&y.data()[..4], &n, 1e-6);
        assert_eq!(&y.data()[4..], &[0.0; 4]);
        let g = t(&[4], &[1.0, 2.0, -1.0, 0.0]);
        let b = t(&[4], &[0.5, 0.0, 0.0, 7.0]);
        let y = layer_norm(&x, &g, &b, 1e-5).unwrap();
        close(&y.data()[..4], &[n[0] + 0.5, 2.0 * n[1], -n[2], 7.0], 1e-6);
    }

    #[test]
    fn layer_norm_survives_a_large_offset() {
        // The same row shifted by 3000: one-pass E[x^2] - E[x]^2 would compute ~9e6 - ~9e6 in
        // f32 (spacing 1.0 there) and lose the variance of 1.25 entirely.
        let ones = t(&[4], &[1.0; 4]);
        let zeros = t(&[4], &[0.0; 4]);
        let x = t(&[1, 4], &[3001.0, 3002.0, 3003.0, 3004.0]);
        let y = layer_norm(&x, &ones, &zeros, 1e-5).unwrap();
        close(
            y.data(),
            &[-1.341_635_4, -0.447_211_8, 0.447_211_8, 1.341_635_4],
            1e-5,
        );
    }

    #[test]
    fn gelu_values() {
        // From the float64 formula in scripts/gpt2_golden.py, rounded to f32.
        let x = t(&[5], &[1.0, -1.0, 0.5, 3.0, -3.0]);
        let want = [
            0.841_192,
            -0.158_808,
            0.345_714,
            2.996_362_7,
            -0.003_637_392,
        ];
        close(gelu(&x).data(), &want, 1e-6);
        assert_eq!(gelu_scalar(0.0), 0.0);
        assert_eq!(gelu_scalar(20.0), 20.0);
        assert!(gelu_scalar(-20.0).abs() < 1e-30);
    }

    #[test]
    fn softmax_values_and_stability() {
        let y = softmax_rows(&t(&[2, 3], &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0])).unwrap();
        close(
            &y.data()[..3],
            &[0.090_030_57, 0.244_728_47, 0.665_240_96],
            1e-7,
        );
        close(&y.data()[3..], &[1.0 / 3.0; 3], 1e-7);
        // exp(1000) overflows; with the max subtracted these are exp(0) and exp(-1).
        let y = softmax_rows(&t(&[1, 2], &[1000.0, 999.0])).unwrap();
        close(y.data(), &[0.731_058_6, 0.268_941_4], 1e-7);
        assert!(softmax_rows(&Tensor::zeros(&[0, 4])).unwrap().is_empty());
    }

    #[test]
    fn attention_one_head_by_hand() {
        // T = 2, E = 2, one head. Row layout: q0 q1 | k0 k1 | v0 v1.
        let qkv = t(
            &[2, 6],
            &[1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 0.0, 1.0, 3.0, -1.0],
        );
        let y = causal_attention(&qkv, 1).unwrap();
        // Position 0 can only see itself: its output is exactly v at position 0.
        assert_eq!(&y.data()[..2], &[1.0, 2.0]);
        // Position 1: scores (0.5*1 + 1*2, 0.5*0 + 1*1) / sqrt(2), softmax, mix the v rows.
        close(&y.data()[2..], &[1.514_366_6, 1.228_450_1], 1e-6);
    }

    #[test]
    fn attention_heads_are_independent() {
        // Same numbers, two heads of width 1: head 0 sees column 0 of q, k, v only.
        let qkv = t(
            &[2, 6],
            &[1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 0.0, 1.0, 3.0, -1.0],
        );
        let y = causal_attention(&qkv, 2).unwrap();
        assert_eq!(&y.data()[..2], &[1.0, 2.0]);
        close(&y.data()[2..], &[1.755_081_3, 1.193_175_7], 1e-6);
        assert!(causal_attention(&qkv, 4).is_err());
    }

    #[test]
    fn attention_is_causal() {
        // Changing the last position must not change any earlier output.
        let mut rng = crate::rng::Rng::new(5);
        let a = rng.vec(5 * 12, -2.0, 2.0);
        let mut b = a.clone();
        b[4 * 12..].iter_mut().for_each(|v| *v += 1.0);
        let ya = causal_attention(&t(&[5, 12], &a), 2).unwrap();
        let yb = causal_attention(&t(&[5, 12], &b), 2).unwrap();
        assert_eq!(&ya.data()[..4 * 4], &yb.data()[..4 * 4]);
        assert_ne!(&ya.data()[4 * 4..], &yb.data()[4 * 4..]);
    }

    #[test]
    fn embed_adds_positions() {
        let wte = t(&[3, 2], &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let wpe = t(&[2, 2], &[0.5, 0.0, 0.0, 0.25]);
        let y = embed(&wte, &wpe, &[2, 2]).unwrap();
        assert_eq!(y.data(), &[2.5, 2.0, 2.0, 2.25]);
        assert!(embed(&wte, &wpe, &[3]).is_err()); // id outside the vocab
        assert!(embed(&wte, &wpe, &[0, 0, 0]).is_err()); // longer than the context
        assert!(embed(&wte, &wpe, &[]).unwrap().is_empty());
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[-1.0]), Some(0));
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::INFINITY]), Some(1));
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[f32::NAN, 0.0, 1.0]), None);
        assert_eq!(argmax(&[0.0, 1.0, f32::NAN]), None);
    }

    #[test]
    fn transpose_2x3() {
        let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = transpose(&a).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&t).unwrap(), a);
        assert!(transpose(&Tensor::zeros(&[2, 2, 2])).is_err());
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[3, 2]);
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn rms_norm_scales_by_root_mean_square() {
        // Row [3, 4]: mean square 12.5, so each value is divided by sqrt(12.5) (eps 0).
        let y = rms_norm(&t(&[1, 2], &[3.0, 4.0]), &t(&[2], &[1.0, 2.0]), 0.0).unwrap();
        let r = 12.5f64.sqrt();
        assert_eq!(y.data(), &[(3.0 / r) as f32, (8.0 / r) as f32]);
        // No mean subtraction: a constant row is not zeroed (LayerNorm would zero it).
        let y = rms_norm(&t(&[1, 2], &[5.0, 5.0]), &t(&[2], &[1.0, 1.0]), 0.0).unwrap();
        assert_eq!(y.data(), &[1.0, 1.0]);
        assert!(rms_norm(&t(&[1, 2], &[1.0, 2.0]), &t(&[3], &[1.0; 3]), 1e-5).is_err());
    }

    #[test]
    fn silu_mul_pairs_gate_with_up() {
        // [gate | up] = [0, 2, 3 | 5, 7, -1]
        let y = silu_mul(&t(&[1, 6], &[0.0, 2.0, 3.0, 5.0, 7.0, -1.0])).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        assert_eq!(y.data(), &[0.0, silu(2.0) * 7.0, -silu(3.0)]);
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(10.0) - 10.0).abs() < 1e-3 && silu(-10.0).abs() < 1e-3);
        assert!(silu_mul(&t(&[1, 3], &[0.0; 3])).is_err());
    }

    #[test]
    fn rope_turns_pair_i_with_i_plus_half() {
        // d = 4, theta = 10000: pair 0 (dims 0, 2) turns by pos, pair 1 (dims 1, 3) by
        // pos / 100. One head rotated, a second (the "value") copied.
        let (cos, sin) = rope_tables(4, 4, 10000.0);
        assert_eq!(cos.shape(), &[4, 2]);
        assert_eq!(cos.data()[2 * 2], 2f64.cos() as f32);
        assert_eq!(sin.data()[3 * 2 + 1], 0.03f64.sin() as f32);
        let x = t(&[2, 8], &[1.0, 2.0, 3.0, 4.0, 9.0, 9.0, 9.0, 9.0].repeat(2));
        // Position 0 is the identity.
        assert_eq!(
            rope(&x, 1, 4, 0, &cos, &sin).unwrap().data()[..8],
            x.data()[..8]
        );
        let y = rope(&x, 1, 4, 2, &cos, &sin).unwrap(); // rows at positions 2 and 3
        let (c, s) = (cos.data()[2 * 2], sin.data()[2 * 2]);
        assert_eq!(y.data()[0], 1.0 * c - 3.0 * s);
        assert_eq!(y.data()[2], 3.0 * c + 1.0 * s);
        assert_eq!(&y.data()[4..8], &[9.0; 4]); // not rotated
        // A rotation keeps each pair's length.
        for (row, pos) in [(0, 2), (1, 3)] {
            let r = &y.data()[row * 8..];
            for (i, j) in [(0, 2), (1, 3)] {
                let before = x.data()[i].hypot(x.data()[j]);
                assert!((r[i].hypot(r[j]) - before).abs() < 1e-6, "pos {pos}");
            }
        }
        // The tables must cover the positions, and d must be even.
        assert!(rope(&x, 1, 4, 3, &cos, &sin).is_err());
        assert!(rope(&x, 1, 3, 0, &cos, &sin).is_err());
        assert!(rope(&x, 3, 4, 0, &cos, &sin).is_err());
    }

    #[test]
    fn gqa_is_attention_with_shared_heads_repeated() {
        // 4 query heads over 2 key/value heads (d = 2): the same as plain multi-head attention
        // where K and V head g are copied to heads 2g and 2g + 1.
        let (n, d, h, kv) = (5, 2, 4, 2);
        let mut rng = crate::rng::Rng::new(9);
        let x: Vec<f32> = (0..n * (h + 2 * kv) * d)
            .map(|_| rng.uniform(-2.0, 2.0))
            .collect();
        let gqa = causal_attention_gqa(&t(&[n, (h + 2 * kv) * d], &x), h, kv).unwrap();
        let mut mha = Vec::new();
        for row in x.chunks_exact((h + 2 * kv) * d) {
            let (q, rest) = row.split_at(h * d);
            let (k, v) = rest.split_at(kv * d);
            mha.extend_from_slice(q);
            for part in [k, v] {
                for head in 0..h {
                    mha.extend_from_slice(&part[head / 2 * d..][..d]);
                }
            }
        }
        let want = causal_attention(&t(&[n, 3 * h * d], &mha), h).unwrap();
        assert_eq!(gqa.data(), want.data());
        assert!(causal_attention_gqa(&t(&[n, 16], &x[..n * 16]), 4, 3).is_err());
    }
}
