//! The CPU reference forward pass for Llama-family models (D69): plain f32 Rust from the ops
//! in `cpu.rs`, the oracle the GPU path is compared against.
//!
//! ```text
//! x = embed[ids]                                              [T, E]  (no position embedding)
//! for each block:
//!     qkv = rope(norm(x) @ wqkv^T)                            q and k turned by position (D70)
//!     x = x + attention_gqa(qkv) @ wo^T                       H query heads share KV heads (D71)
//!     x = x + (silu(gate(norm(x))) * up(norm(x))) @ wdown^T   SwiGLU
//! logits = norm(x) @ embed^T                                  [T, V]  (tied LM head)
//! ```

use crate::cpu;
use crate::error::{Error, Result};
use crate::llama::{Block, Weights};
use crate::tensor::Tensor;

/// Logits for every position, `[T, V]`.
pub fn forward(w: &Weights, ids: &[u32]) -> Result<Tensor> {
    let h = hidden(w, ids)?;
    cpu::linear(&h, &w.embed, None)
}

/// The final hidden states after the last norm, `[T, E]`.
pub fn hidden(w: &Weights, ids: &[u32]) -> Result<Tensor> {
    let c = &w.config;
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    if ids.len() > c.n_ctx {
        return Err(Error::Input(format!(
            "{} tokens exceed the model's {} positions",
            ids.len(),
            c.n_ctx
        )));
    }
    let e = c.n_embd;
    let mut x = Vec::with_capacity(ids.len() * e);
    for &id in ids {
        if id as usize >= c.vocab_size {
            return Err(Error::Input(format!("token id {id} is outside the vocab")));
        }
        x.extend_from_slice(&w.embed.data()[id as usize * e..][..e]);
    }
    let mut x = Tensor::new(&[ids.len(), e], x)?;
    let (cos, sin) = cpu::rope_tables(ids.len(), c.head_dim(), c.rope_theta);
    for block in &w.blocks {
        x = block_forward(w, block, &x, &cos, &sin)?;
    }
    cpu::rms_norm(&x, &w.norm, c.rms_eps)
}

fn block_forward(w: &Weights, b: &Block, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let c = &w.config;
    let h = cpu::rms_norm(x, &b.attn_norm, c.rms_eps)?;
    let qkv = cpu::linear(&h, &b.qkv, None)?;
    let qkv = cpu::rope(&qkv, c.n_head + c.n_kv_head, c.head_dim(), 0, cos, sin)?;
    let a = cpu::causal_attention_gqa(&qkv, c.n_head, c.n_kv_head)?;
    let x = cpu::add(x, &cpu::linear(&a, &b.o, None)?)?;

    let h = cpu::rms_norm(&x, &b.mlp_norm, c.rms_eps)?;
    let m = cpu::silu_mul(&cpu::linear(&h, &b.gate_up, None)?)?;
    cpu::add(&x, &cpu::linear(&m, &b.down, None)?)
}

/// Logits for the next token only, `[V]`: the LM head on the last row.
pub fn next_logits(w: &Weights, ids: &[u32]) -> Result<Vec<f32>> {
    let h = hidden(w, ids)?;
    let e = w.config.n_embd;
    let last = Tensor::new(&[1, e], h.data()[h.len() - e..].to_vec())?;
    Ok(cpu::linear(&last, &w.embed, None)?.data().to_vec())
}

/// Greedy decoding with full recompute each step (the reference has no KV cache).
pub fn generate_greedy(
    w: &Weights,
    prompt: &[u32],
    n: usize,
    on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    crate::gpt2::greedy(
        prompt,
        n,
        w.config.n_ctx,
        |ids| next_logits(w, ids),
        on_token,
    )
}
