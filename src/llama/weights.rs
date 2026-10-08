//! Llama weights in the reference layout (D7): every matrix `[out, in]`, which is how Hugging
//! Face stores them already (`nn.Linear`), so nothing is transposed. Two pairs are fused at
//! load (D72): q, k, v into one `[(H + 2 KV) d, E]` matrix, and gate, up into one `[2F, E]`.

use std::path::Path;

use crate::error::{Error, Result};
use crate::llama::Config;
use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

/// One block: `x += o(attention(rope(qkv(norm(x)))))`, then
/// `x += down(silu(gate(norm(x))) * up(norm(x)))`.
#[derive(Debug, Clone)]
pub struct Block {
    /// RMSNorm gain before attention (`input_layernorm`), `[E]`.
    pub attn_norm: Tensor,
    /// Fused projection `[(H + 2 KV) d, E]`: output columns `0..H d` are q, then `KV d` of k,
    /// then `KV d` of v.
    pub qkv: Tensor,
    /// Attention output projection (`o_proj`), `[E, E]`.
    pub o: Tensor,
    /// RMSNorm gain before the MLP (`post_attention_layernorm`), `[E]`.
    pub mlp_norm: Tensor,
    /// Fused `[2F, E]`: output columns `0..F` are the gate, `F..2F` up.
    pub gate_up: Tensor,
    /// `[E, F]`.
    pub down: Tensor,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub config: Config,
    /// Token embedding `[V, E]`, also the LM head (tied): `logits = h @ embed^T`.
    pub embed: Tensor,
    pub blocks: Vec<Block>,
    /// Final RMSNorm gain, `[E]`.
    pub norm: Tensor,
}

impl Weights {
    /// Load `config.json` and `model.safetensors` from a Hugging Face model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let st = SafeTensors::open(&dir.join("model.safetensors"))?;
        Self::from_safetensors(config, &st)
    }

    pub fn from_safetensors(config: Config, st: &SafeTensors) -> Result<Self> {
        let get = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let t = st.tensor(name)?;
            if t.shape() != shape {
                return Err(Error::Format(format!(
                    "`{name}` has shape {:?}, config implies {shape:?}",
                    t.shape()
                )));
            }
            Ok(t)
        };
        let c = &config;
        let (e, f, v) = (c.n_embd, c.n_ff, c.vocab_size);
        let (q_dim, kv_dim) = (c.n_head * c.head_dim(), c.kv_dim());
        let blocks = (0..c.n_layer)
            .map(|i| {
                let p = format!("model.layers.{i}");
                let w = |name: &str, rows: usize, cols: usize| {
                    get(&format!("{p}.{name}.weight"), &[rows, cols])
                };
                Ok(Block {
                    attn_norm: get(&format!("{p}.input_layernorm.weight"), &[e])?,
                    qkv: stack(&[
                        w("self_attn.q_proj", q_dim, e)?,
                        w("self_attn.k_proj", kv_dim, e)?,
                        w("self_attn.v_proj", kv_dim, e)?,
                    ])?,
                    o: w("self_attn.o_proj", e, q_dim)?,
                    mlp_norm: get(&format!("{p}.post_attention_layernorm.weight"), &[e])?,
                    gate_up: stack(&[w("mlp.gate_proj", f, e)?, w("mlp.up_proj", f, e)?])?,
                    down: w("mlp.down_proj", e, f)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if st.entry("lm_head.weight").is_some() {
            return Err(Error::Format(
                "`lm_head.weight` present: untied embeddings are not supported".into(),
            ));
        }
        Ok(Weights {
            embed: get("model.embed_tokens.weight", &[v, e])?,
            blocks,
            norm: get("model.norm.weight", &[e])?,
            config,
        })
    }

    /// Learned parameters: every tensor's elements. (SmolLM2-135M: 134,515,008.)
    pub fn param_count(&self) -> usize {
        let block = |b: &Block| {
            b.attn_norm.len()
                + b.qkv.len()
                + b.o.len()
                + b.mlp_norm.len()
                + b.gate_up.len()
                + b.down.len()
        };
        self.embed.len() + self.blocks.iter().map(block).sum::<usize>() + self.norm.len()
    }
}

/// Matrices with the same column count, one under the other: rows of the first, then the next.
fn stack(parts: &[Tensor]) -> Result<Tensor> {
    let cols = parts[0].shape()[1];
    let mut data = Vec::new();
    for p in parts {
        if p.shape()[1] != cols {
            return Err(Error::Shape(format!(
                "stack: {:?} vs {cols} columns",
                p.shape()
            )));
        }
        data.extend_from_slice(p.data());
    }
    Tensor::new(&[data.len() / cols, cols], data)
}
