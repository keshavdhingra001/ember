//! GPT-2 hyperparameters, read from the model's `config.json` (D14). Nothing about the model's
//! size is hard-coded, so gpt2-medium/large/xl and the tiny test models load the same way.

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    /// Model width E: the size of every residual-stream vector.
    pub n_embd: usize,
    /// Context length: the number of learned position embeddings (`n_positions`).
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub ln_eps: f32,
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
        // GPT-2's MLP uses the tanh approximation of GELU ("gelu_new"). A config asking for
        // anything else describes a model this code would silently compute wrong, so refuse it.
        let act = v.get("activation_function").and_then(Value::as_str);
        if !matches!(act, None | Some("gelu_new")) {
            return Err(format!(
                "activation_function {act:?} is not supported (only gelu_new)"
            ));
        }
        let c = Config {
            n_layer: int("n_layer")?,
            n_head: int("n_head")?,
            n_embd: int("n_embd")?,
            n_ctx: int("n_positions")?,
            vocab_size: int("vocab_size")?,
            ln_eps: v
                .get("layer_norm_epsilon")
                .and_then(Value::as_f64)
                .ok_or("`layer_norm_epsilon` must be a number")? as f32,
        };
        if !c.n_embd.is_multiple_of(c.n_head) {
            return Err(format!(
                "n_embd {} is not divisible by n_head {}",
                c.n_embd, c.n_head
            ));
        }
        Ok(c)
    }

    /// Width of one attention head, D = E / H.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPT2: &str = r#"{"activation_function": "gelu_new", "layer_norm_epsilon": 1e-05,
        "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024,
        "vocab_size": 50257, "task_specific_params": {"text-generation": {"max_length": 50}}}"#;

    #[test]
    fn parses_gpt2_small() {
        let c = Config::from_json(GPT2).unwrap();
        assert_eq!(
            c,
            Config {
                n_layer: 12,
                n_head: 12,
                n_embd: 768,
                n_ctx: 1024,
                vocab_size: 50257,
                ln_eps: 1e-5,
            }
        );
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn rejects_what_we_cant_run() {
        let bad = [
            (
                GPT2.replace("\"n_head\": 12", "\"n_head\": 7"),
                "not divisible",
            ),
            (
                GPT2.replace("\"n_layer\": 12", "\"n_layer\": 0"),
                "`n_layer`",
            ),
            (
                GPT2.replace("\"vocab_size\": 50257", "\"vocab_size\": -1"),
                "`vocab_size`",
            ),
            (GPT2.replace("\"n_positions\": 1024,", ""), "`n_positions`"),
            (GPT2.replace("gelu_new", "relu"), "not supported"),
            (GPT2.replace("1e-05", "\"tiny\""), "layer_norm_epsilon"),
        ];
        for (json, want) in bad {
            let e = Config::from_json(&json).unwrap_err();
            assert!(e.contains(want), "got `{e}`, wanted `{want}`");
        }
    }
}
