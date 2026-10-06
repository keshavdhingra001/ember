//! The CPU reference forward pass (D3): GPT-2 in plain f32 Rust, built only from the ops in
//! `cpu.rs`. This is the oracle the GPU model (M3) is compared against, so it reads like the
//! paper, not like fast code.
//!
//! ```text
//! x = wte[ids] + wpe[0..T]                                   [T, E]
//! for each block:
//!     x = x + attn_out(causal_attention(qkv(ln_1(x))))       (pre-norm residual)
//!     x = x + fc_out(gelu(fc(ln_2(x))))
//! logits = ln_f(x) @ wte^T                                   [T, V]  (tied LM head)
//! ```

use crate::cpu;
use crate::error::{Error, Result};
use crate::gpt2::{Block, Weights};
use crate::tensor::Tensor;

/// Logits for every position, `[T, V]`: row t scores the token that follows `ids[..=t]`.
pub fn forward(w: &Weights, ids: &[u32]) -> Result<Tensor> {
    let h = hidden(w, ids)?;
    cpu::linear(&h, &w.wte, None)
}

/// The final hidden states after `ln_f`, `[T, E]`: everything except the LM head.
pub fn hidden(w: &Weights, ids: &[u32]) -> Result<Tensor> {
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    let mut x = cpu::embed(&w.wte, &w.wpe, ids)?;
    for block in &w.blocks {
        x = block_forward(w, block, &x)?;
    }
    let eps = w.config.ln_eps;
    cpu::layer_norm(&x, &w.ln_f.gain, &w.ln_f.bias, eps)
}

fn block_forward(w: &Weights, b: &Block, x: &Tensor) -> Result<Tensor> {
    let eps = w.config.ln_eps;
    let h = cpu::layer_norm(x, &b.ln_1.gain, &b.ln_1.bias, eps)?;
    let qkv = cpu::linear(&h, &b.qkv.w, Some(&b.qkv.b))?;
    let a = cpu::causal_attention(&qkv, w.config.n_head)?;
    let a = cpu::linear(&a, &b.attn_out.w, Some(&b.attn_out.b))?;
    let x = cpu::add(x, &a)?;

    let h = cpu::layer_norm(&x, &b.ln_2.gain, &b.ln_2.bias, eps)?;
    let m = cpu::gelu(&cpu::linear(&h, &b.fc.w, Some(&b.fc.b))?);
    let m = cpu::linear(&m, &b.fc_out.w, Some(&b.fc_out.b))?;
    cpu::add(&x, &m)
}

/// Logits for the next token only, `[V]`. Same numbers as the last row of [`forward`], but the
/// LM head (V x E, the single biggest matrix) runs on one row instead of T.
pub fn next_logits(w: &Weights, ids: &[u32]) -> Result<Vec<f32>> {
    let h = hidden(w, ids)?;
    let e = w.config.n_embd;
    let last = Tensor::new(&[1, e], h.data()[h.len() - e..].to_vec())?;
    Ok(cpu::linear(&last, &w.wte, None)?.data().to_vec())
}

/// Greedy decoding: `n` times, append the highest-scoring next token. The reference has no KV
/// cache (only the GPU model does, M4): every step recomputes the whole sequence, which is
/// O(n^2) work but obviously correct.
/// Stops early at the context length. Calls `on_token` after each token (for streaming output).
pub fn generate_greedy(
    w: &Weights,
    prompt: &[u32],
    n: usize,
    on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    greedy(
        prompt,
        n,
        w.config.n_ctx,
        |ids| next_logits(w, ids),
        on_token,
    )
}

/// The greedy loop, shared by the CPU reference and the GPU model (`gpt2::gpu`): only
/// `next_logits` differs between them.
pub(crate) fn greedy(
    prompt: &[u32],
    n: usize,
    n_ctx: usize,
    mut next_logits: impl FnMut(&[u32]) -> Result<Vec<f32>>,
    mut on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    let mut ids = prompt.to_vec();
    let mut out = Vec::with_capacity(n);
    while out.len() < n && ids.len() < n_ctx {
        let logits = next_logits(&ids)?;
        let next = cpu::argmax(&logits)
            .ok_or_else(|| Error::Input("logits contain NaN; the model is broken".into()))?
            as u32;
        ids.push(next);
        out.push(next);
        on_token(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt2::Config;

    fn tiny() -> Weights {
        let config = Config {
            n_layer: 2,
            n_head: 3,
            n_embd: 12,
            n_ctx: 16,
            vocab_size: 37,
            ln_eps: 1e-5,
        };
        Weights::random(config, 7)
    }

    #[test]
    fn shapes_and_errors() {
        let w = tiny();
        let logits = forward(&w, &[1, 2, 3]).unwrap();
        assert_eq!(logits.shape(), &[3, 37]);
        assert!(forward(&w, &[]).is_err());
        assert!(forward(&w, &[37]).is_err());
        assert!(forward(&w, &[0; 17]).is_err());
        assert!(forward(&w, &[0; 16]).is_ok());
    }

    #[test]
    fn a_prefix_gives_the_same_rows_bitwise() {
        // Causality end to end: positions 0..k can't see what comes after them, so running only
        // the prefix must reproduce those rows exactly (same ops, same order, same bits).
        let w = tiny();
        let ids = [5, 0, 36, 17, 17, 2, 30, 11];
        let full = forward(&w, &ids).unwrap();
        for k in 1..ids.len() {
            let part = forward(&w, &ids[..k]).unwrap();
            assert_eq!(part.data(), &full.data()[..k * 37], "prefix {k}");
        }
    }

    #[test]
    fn positions_matter() {
        // Same tokens, different order: the last row must change (wpe and attention are used).
        let w = tiny();
        let a = next_logits(&w, &[1, 2, 3]).unwrap();
        let b = next_logits(&w, &[2, 1, 3]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn next_logits_is_the_last_row() {
        let w = tiny();
        let ids = [3, 1, 4, 1, 5];
        let full = forward(&w, &ids).unwrap();
        assert_eq!(next_logits(&w, &ids).unwrap(), &full.data()[4 * 37..]);
    }

    #[test]
    fn greedy_is_argmax_and_stops_at_the_context() {
        let w = tiny();
        let mut streamed = Vec::new();
        let out = generate_greedy(&w, &[1, 2], 3, |t| streamed.push(t)).unwrap();
        assert_eq!(out, streamed);
        let first = cpu::argmax(&next_logits(&w, &[1, 2]).unwrap()).unwrap() as u32;
        assert_eq!(out[0], first);
        // A 14-token prompt leaves room for 2 more in a 16-token context.
        assert_eq!(generate_greedy(&w, &[0; 14], 10, |_| {}).unwrap().len(), 2);
    }
}
