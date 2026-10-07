//! GPT-2 weights in the layout the reference forward pass wants (D7): every linear layer's
//! weight is `[out, in]` row-major, so output `o` is a dot product of the input row with the
//! contiguous row `w[o, :]`.
//!
//! The checkpoint stores them the other way round. OpenAI's GPT-2 used a `Conv1D` module whose
//! weight is `[in, out]` (y = x @ W), and Hugging Face kept that layout for compatibility. So
//! every matrix except the embeddings is transposed once here, at load time.

use std::path::Path;

use crate::cpu;
use crate::error::{Error, Result};
use crate::gpt2::Config;
use crate::rng::Rng;
use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

/// `y = x @ w^T + b` with `w: [out, in]`, `b: [out]`.
#[derive(Debug, Clone)]
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

/// LayerNorm's learned per-feature scale (`gain`, called gamma or `weight`) and shift (`bias`).
#[derive(Debug, Clone)]
pub struct Norm {
    pub gain: Tensor,
    pub bias: Tensor,
}

/// One transformer block: `x += attn(ln_1(x)); x += mlp(ln_2(x))`.
#[derive(Debug, Clone)]
pub struct Block {
    pub ln_1: Norm,
    /// Fused Q, K, V projection: `[3E, E]`. Output columns `0..E` are Q, `E..2E` K, `2E..3E` V.
    pub qkv: Linear,
    /// Attention output projection, `[E, E]`.
    pub attn_out: Linear,
    pub ln_2: Norm,
    /// MLP up-projection, `[4E, E]`.
    pub fc: Linear,
    /// MLP down-projection, `[E, 4E]`.
    pub fc_out: Linear,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub config: Config,
    /// Token embedding `[V, E]`. Also the output projection: GPT-2 ties the LM head to it,
    /// `logits = h @ wte^T`, which is already the `[out, in]` layout.
    pub wte: Tensor,
    /// Learned position embedding `[n_ctx, E]`.
    pub wpe: Tensor,
    pub blocks: Vec<Block>,
    pub ln_f: Norm,
}

impl Weights {
    /// Load `config.json` and `model.safetensors` from a Hugging Face model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let st = SafeTensors::open(&dir.join("model.safetensors"))?;
        Self::from_safetensors(config, &st)
    }

    pub fn from_safetensors(config: Config, st: &SafeTensors) -> Result<Self> {
        // Some checkpoints save the bare transformer (`wte.weight`), others the LM-head model
        // (`transformer.wte.weight`). Same tensors either way.
        let prefix = if st.entry("transformer.wte.weight").is_some() {
            "transformer."
        } else {
            ""
        };
        let get = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let full = format!("{prefix}{name}");
            let t = st.tensor(&full)?;
            if t.shape() != shape {
                return Err(Error::Format(format!(
                    "`{full}` has shape {:?}, config implies {shape:?}",
                    t.shape()
                )));
            }
            Ok(t)
        };
        let (e, v) = (config.n_embd, config.vocab_size);
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                gain: get(&format!("{name}.weight"), &[e])?,
                bias: get(&format!("{name}.bias"), &[e])?,
            })
        };
        // Conv1D stores [in, out]; transpose to [out, in].
        let conv1d = |name: &str, n_in: usize, n_out: usize| -> Result<Linear> {
            Ok(Linear {
                w: cpu::transpose(&get(&format!("{name}.weight"), &[n_in, n_out])?)?,
                b: get(&format!("{name}.bias"), &[n_out])?,
            })
        };
        let blocks = (0..config.n_layer)
            .map(|i| {
                let p = format!("h.{i}");
                Ok(Block {
                    ln_1: norm(&format!("{p}.ln_1"))?,
                    qkv: conv1d(&format!("{p}.attn.c_attn"), e, 3 * e)?,
                    attn_out: conv1d(&format!("{p}.attn.c_proj"), e, e)?,
                    ln_2: norm(&format!("{p}.ln_2"))?,
                    fc: conv1d(&format!("{p}.mlp.c_fc"), e, 4 * e)?,
                    fc_out: conv1d(&format!("{p}.mlp.c_proj"), 4 * e, e)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Weights {
            wte: get("wte.weight", &[v, e])?,
            wpe: get("wpe.weight", &[config.n_ctx, e])?,
            blocks,
            ln_f: norm("ln_f")?,
            config,
        })
    }

    /// Learned parameters: every tensor's elements. (GPT-2 124M: 124,439,808.)
    pub fn param_count(&self) -> usize {
        let norm = |n: &Norm| n.gain.len() + n.bias.len();
        let lin = |l: &Linear| l.w.len() + l.b.len();
        let block = |b: &Block| {
            norm(&b.ln_1)
                + lin(&b.qkv)
                + lin(&b.attn_out)
                + norm(&b.ln_2)
                + lin(&b.fc)
                + lin(&b.fc_out)
        };
        self.wte.len()
            + self.wpe.len()
            + self.blocks.iter().map(block).sum::<usize>()
            + norm(&self.ln_f)
    }

    /// A model with seeded random weights, for tests that don't need the real checkpoint (D15).
    /// Scales keep activations O(1) and attention far from uniform, so bugs show up as big
    /// differences instead of hiding in noise.
    pub fn random(config: Config, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let (e, v) = (config.n_embd, config.vocab_size);
        let wte = uniform(&mut rng, &[v, e], -1.0, 1.0);
        let wpe = uniform(&mut rng, &[config.n_ctx, e], -0.5, 0.5);
        let blocks = (0..config.n_layer)
            .map(|_| Block {
                ln_1: random_norm(&mut rng, e),
                qkv: random_linear(&mut rng, e, 3 * e),
                attn_out: random_linear(&mut rng, e, e),
                ln_2: random_norm(&mut rng, e),
                fc: random_linear(&mut rng, e, 4 * e),
                fc_out: random_linear(&mut rng, 4 * e, e),
            })
            .collect();
        let ln_f = random_norm(&mut rng, e);
        Weights {
            config,
            wte,
            wpe,
            blocks,
            ln_f,
        }
    }
}

fn uniform(rng: &mut Rng, shape: &[usize], lo: f32, hi: f32) -> Tensor {
    Tensor::new(shape, rng.vec(shape.iter().product(), lo, hi)).unwrap()
}

fn random_norm(rng: &mut Rng, e: usize) -> Norm {
    Norm {
        gain: uniform(rng, &[e], 0.8, 1.2),
        bias: uniform(rng, &[e], -0.1, 0.1),
    }
}

/// Uniform in +-1/sqrt(n_in), so each output (a sum of n_in products) stays O(1).
fn random_linear(rng: &mut Rng, n_in: usize, n_out: usize) -> Linear {
    let s = 1.0 / (n_in as f32).sqrt();
    Linear {
        w: uniform(rng, &[n_out, n_in], -s, s),
        b: uniform(rng, &[n_out], -0.1, 0.1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::serialize;

    fn config() -> Config {
        Config {
            n_layer: 1,
            n_head: 1,
            n_embd: 2,
            n_ctx: 3,
            vocab_size: 5,
            ln_eps: 1e-5,
        }
    }

    /// Every tensor of a 1-layer, E=2 checkpoint in the Hugging Face layout, filled with
    /// distinct values (so a missed or doubled transpose can't go unnoticed).
    fn checkpoint(prefix: &str, skip: &str) -> Vec<u8> {
        let shapes: [(&str, &[usize]); 16] = [
            ("wte.weight", &[5, 2]),
            ("wpe.weight", &[3, 2]),
            ("h.0.ln_1.weight", &[2]),
            ("h.0.ln_1.bias", &[2]),
            ("h.0.attn.c_attn.weight", &[2, 6]),
            ("h.0.attn.c_attn.bias", &[6]),
            ("h.0.attn.c_proj.weight", &[2, 2]),
            ("h.0.attn.c_proj.bias", &[2]),
            ("h.0.ln_2.weight", &[2]),
            ("h.0.ln_2.bias", &[2]),
            ("h.0.mlp.c_fc.weight", &[2, 8]),
            ("h.0.mlp.c_fc.bias", &[8]),
            ("h.0.mlp.c_proj.weight", &[8, 2]),
            ("h.0.mlp.c_proj.bias", &[2]),
            ("ln_f.weight", &[2]),
            ("ln_f.bias", &[2]),
        ];
        let mut next = 0.0;
        let tensors: Vec<(String, Tensor)> = shapes
            .iter()
            .filter(|(name, _)| *name != skip)
            .map(|&(name, shape)| {
                let n: usize = shape.iter().product();
                let data = (0..n).map(|i| next + i as f32).collect();
                next += 100.0;
                (format!("{prefix}{name}"), Tensor::new(shape, data).unwrap())
            })
            .collect();
        let refs: Vec<(&str, &Tensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        serialize(&refs)
    }

    #[test]
    fn loads_and_transposes_conv1d_weights() {
        let st = SafeTensors::from_bytes(checkpoint("", "")).unwrap();
        let w = Weights::from_safetensors(config(), &st).unwrap();
        // c_attn.weight is [in=2, out=6] holding 400..412 row-major: [in, out] element (i, o)
        // is 400 + 6i + o. After the transpose qkv.w is [out=6, in=2] with (o, i) the same value.
        let qkv = &w.blocks[0].qkv.w;
        assert_eq!(qkv.shape(), &[6, 2]);
        for o in 0..6 {
            for i in 0..2 {
                assert_eq!(qkv.data()[o * 2 + i], 400.0 + (6 * i + o) as f32);
            }
        }
        assert_eq!(w.blocks[0].fc_out.w.shape(), &[2, 8]);
        // Embeddings are not transposed: wte stays [V, E].
        assert_eq!(w.wte.shape(), &[5, 2]);
        assert_eq!(&w.wte.data()[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(w.ln_f.bias.data(), &[1500.0, 1501.0]);
    }

    #[test]
    fn accepts_the_transformer_prefix() {
        let st = SafeTensors::from_bytes(checkpoint("transformer.", "")).unwrap();
        assert!(Weights::from_safetensors(config(), &st).is_ok());
    }

    #[test]
    fn rejects_missing_tensors_and_wrong_shapes() {
        let st = SafeTensors::from_bytes(checkpoint("", "h.0.mlp.c_fc.bias")).unwrap();
        let e = Weights::from_safetensors(config(), &st).unwrap_err();
        assert!(e.to_string().contains("h.0.mlp.c_fc.bias"), "{e}");
        // A config claiming a bigger vocabulary than the checkpoint has.
        let st = SafeTensors::from_bytes(checkpoint("", "")).unwrap();
        let big = Config {
            vocab_size: 6,
            ..config()
        };
        let e = Weights::from_safetensors(big, &st).unwrap_err();
        assert!(
            e.to_string().contains("`wte.weight` has shape [5, 2]"),
            "{e}"
        );
    }

    #[test]
    fn random_is_seeded() {
        let a = Weights::random(config(), 1);
        assert_eq!(a.wte, Weights::random(config(), 1).wte);
        assert_ne!(a.wte, Weights::random(config(), 2).wte);
    }

    #[test]
    fn param_count_counts_every_tensor() {
        // wte 5x2 + wpe 3x2, one block (2 norms of 2+2; qkv 2x6+6, attn_out 2x2+2, fc 2x8+8,
        // fc_out 8x2+2), ln_f 2+2.
        let block = 4 + 18 + 6 + 4 + 24 + 18;
        assert_eq!(
            Weights::random(config(), 1).param_count(),
            10 + 6 + block + 4
        );
    }
}
