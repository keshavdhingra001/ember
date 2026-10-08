//! Llama-family hyperparameters, read from the model's `config.json` (D67). SmolLM2-135M and
//! -360M load the same way; a config asking for something this code doesn't compute (rope
//! scaling, biases, interleaved rotary pairs, another activation) is refused, not run wrong.

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub n_layer: usize,
    /// Query heads.
    pub n_head: usize,
    /// Key/value heads: each serves `n_head / n_kv_head` query heads (grouped-query attention,
    /// D71).
    pub n_kv_head: usize,
    /// Model width E.
    pub n_embd: usize,
    /// Width of the SwiGLU hidden layer (`intermediate_size`).
    pub n_ff: usize,
    pub vocab_size: usize,
    /// Positions the model was trained for (`max_position_embeddings`). The KV cache may be
    /// smaller (D76).
    pub n_ctx: usize,
    /// RoPE base θ: pair i of a head turns by `pos * θ^(-2i/d)`.
    pub rope_theta: f64,
    pub rms_eps: f32,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        Self::from_json(&text).map_err(|e| Error::Format(format!("{}: {e}", path.display())))
    }

    pub fn from_json(text: &str) -> std::result::Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let int = |key: &str| -> std::result::Result<usize, String> {
            match v.get(key).and_then(Value::as_u64) {
                Some(n) if n > 0 => Ok(n as usize),
                _ => Err(format!("`{key}` must be a positive integer")),
            }
        };
        let positive = |key: &str| -> std::result::Result<f64, String> {
            match v.get(key).and_then(Value::as_f64) {
                Some(x) if x.is_finite() && x > 0.0 => Ok(x),
                _ => Err(format!("`{key}` must be a positive finite number")),
            }
        };
        // Settings that would change the math. Absent counts as the plain Llama default.
        let is = |key: &str, want: Value| v.get(key).is_none_or(|x| *x == want);
        let refuse = |what: &str| Err(format!("{what} is not supported"));
        if v.get("model_type").and_then(Value::as_str) != Some("llama") {
            return refuse(&format!("model_type {}", v["model_type"]));
        }
        if !is("hidden_act", "silu".into()) {
            return refuse(&format!("hidden_act {}", v["hidden_act"]));
        }
        if !is("rope_scaling", Value::Null) {
            return refuse("rope_scaling");
        }
        if !is("rope_interleaved", false.into()) {
            return refuse("interleaved rotary pairs (D70)");
        }
        if !is("attention_bias", false.into()) || !is("mlp_bias", false.into()) {
            return refuse("a bias in attention or the MLP");
        }
        if !is("tie_word_embeddings", true.into()) {
            return refuse("a separate lm_head (untied embeddings)");
        }
        let c = Config {
            n_layer: int("num_hidden_layers")?,
            n_head: int("num_attention_heads")?,
            n_kv_head: v
                .get("num_key_value_heads")
                .map_or(int("num_attention_heads"), |_| int("num_key_value_heads"))?,
            n_embd: int("hidden_size")?,
            n_ff: int("intermediate_size")?,
            vocab_size: int("vocab_size")?,
            n_ctx: int("max_position_embeddings")?,
            rope_theta: positive("rope_theta")?,
            rms_eps: positive("rms_norm_eps")? as f32,
        };
        if !c.n_embd.is_multiple_of(c.n_head) {
            return Err(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                c.n_embd, c.n_head
            ));
        }
        if !c.n_head.is_multiple_of(c.n_kv_head) {
            return Err(format!(
                "num_attention_heads {} is not a multiple of num_key_value_heads {}",
                c.n_head, c.n_kv_head
            ));
        }
        if !c.head_dim().is_multiple_of(2) {
            return Err(format!(
                "head dimension {} is odd: RoPE turns pairs",
                c.head_dim()
            ));
        }
        if let Some(d) = v.get("head_dim").filter(|d| !d.is_null())
            && d.as_u64() != Some(c.head_dim() as u64)
        {
            return refuse(&format!("head_dim {d} other than hidden_size / heads"));
        }
        Ok(c)
    }

    /// Width of one head, d = E / n_head (the same for query and key/value heads).
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Width of the key (or value) part of a row: n_kv_head × d.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SmolLM2-135M's config.json, verbatim apart from line breaks.
    const SMOL: &str = r#"{"architectures": ["LlamaForCausalLM"], "attention_bias": false,
        "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 0, "hidden_act": "silu",
        "hidden_size": 576, "initializer_range": 0.041666666666666664, "intermediate_size": 1536,
        "is_llama_config": true, "max_position_embeddings": 8192, "model_type": "llama",
        "num_attention_heads": 9, "num_hidden_layers": 30, "num_key_value_heads": 3,
        "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_interleaved": false,
        "rope_scaling": null, "rope_theta": 100000, "tie_word_embeddings": true,
        "torch_dtype": "bfloat16", "transformers_version": "4.40.1", "use_cache": true,
        "vocab_size": 49152}"#;

    #[test]
    fn parses_smollm2_135m() {
        let c = Config::from_json(SMOL).unwrap();
        assert_eq!(
            c,
            Config {
                n_layer: 30,
                n_head: 9,
                n_kv_head: 3,
                n_embd: 576,
                n_ff: 1536,
                vocab_size: 49152,
                n_ctx: 8192,
                rope_theta: 100000.0,
                rms_eps: 1e-5,
            }
        );
        assert_eq!((c.head_dim(), c.kv_dim()), (64, 192));
        // Without num_key_value_heads every head has its own K and V (plain multi-head).
        let mha = Config::from_json(&SMOL.replace("\"num_key_value_heads\": 3,", "")).unwrap();
        assert_eq!(mha.n_kv_head, 9);
    }

    #[test]
    fn rejects_what_we_cant_run() {
        let bad = [
            (SMOL.replace("\"llama\"", "\"qwen2\""), "model_type"),
            (SMOL.replace("\"silu\"", "\"gelu\""), "hidden_act"),
            (
                SMOL.replace(
                    "\"rope_scaling\": null",
                    "\"rope_scaling\": {\"factor\": 2}",
                ),
                "rope_scaling",
            ),
            (
                SMOL.replace("\"rope_interleaved\": false", "\"rope_interleaved\": true"),
                "interleaved",
            ),
            (
                SMOL.replace("\"attention_bias\": false", "\"attention_bias\": true"),
                "bias",
            ),
            (
                SMOL.replace(
                    "\"tie_word_embeddings\": true",
                    "\"tie_word_embeddings\": false",
                ),
                "lm_head",
            ),
            (
                SMOL.replace("\"num_attention_heads\": 9", "\"num_attention_heads\": 7"),
                "not divisible",
            ),
            (
                SMOL.replace("\"num_key_value_heads\": 3", "\"num_key_value_heads\": 2"),
                "not a multiple",
            ),
            (
                SMOL.replace("\"num_key_value_heads\": 3", "\"num_key_value_heads\": 0"),
                "`num_key_value_heads`",
            ),
            (
                SMOL.replace("\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": 0"),
                "`rms_norm_eps`",
            ),
            (
                SMOL.replace("\"rope_theta\": 100000", "\"rope_theta\": -1"),
                "`rope_theta`",
            ),
            (
                SMOL.replace("\"vocab_size\": 49152", "\"vocab_size\": 0"),
                "`vocab_size`",
            ),
            (
                SMOL.replace("\"is_llama_config\": true", "\"head_dim\": 32"),
                "head_dim",
            ),
            // E = 576 over 192 heads gives d = 3: RoPE needs pairs.
            (
                SMOL.replace("\"num_attention_heads\": 9", "\"num_attention_heads\": 192")
                    .replace("\"num_key_value_heads\": 3", "\"num_key_value_heads\": 64"),
                "odd",
            ),
        ];
        for (json, want) in bad {
            assert_ne!(json, SMOL, "the case must change the config");
            let e = Config::from_json(&json).unwrap_err();
            assert!(e.contains(want), "got `{e}`, wanted `{want}`");
        }
    }
}
