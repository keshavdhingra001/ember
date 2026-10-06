//! GPT-2 on the GPU (M3): the forward pass of `forward.rs`, op for op, with the M2 kernels.
//! Weights are uploaded once (D24); each op submits its own dispatch (D25); nothing is read
//! back until the logits.
//!
//! Two ways to run it: `forward` / `next_logits` recompute the whole sequence (M3, the
//! comparison partner for D33), and `extend` runs only new tokens against a `KvCache` (M4).

use crate::error::{Error, Result};
use crate::gpt2::forward::greedy;
use crate::gpt2::{Config, Linear, Norm, Weights};
use crate::gpu::Gpu;
use crate::ops;
use crate::tensor::GpuTensor;

pub struct GpuLinear {
    pub w: GpuTensor,
    pub b: GpuTensor,
}

pub struct GpuNorm {
    pub gain: GpuTensor,
    pub bias: GpuTensor,
}

pub struct GpuBlock {
    pub ln_1: GpuNorm,
    pub qkv: GpuLinear,
    pub attn_out: GpuLinear,
    pub ln_2: GpuNorm,
    pub fc: GpuLinear,
    pub fc_out: GpuLinear,
}

/// `Weights` in GPU buffers, same layout (D7, D24).
pub struct GpuWeights {
    pub config: Config,
    pub wte: GpuTensor,
    pub wpe: GpuTensor,
    pub blocks: Vec<GpuBlock>,
    pub ln_f: GpuNorm,
}

impl GpuWeights {
    pub fn upload(gpu: &Gpu, w: &Weights) -> Self {
        let linear = |l: &Linear| GpuLinear {
            w: gpu.upload(&l.w),
            b: gpu.upload(&l.b),
        };
        let norm = |n: &Norm| GpuNorm {
            gain: gpu.upload(&n.gain),
            bias: gpu.upload(&n.bias),
        };
        GpuWeights {
            config: w.config.clone(),
            wte: gpu.upload(&w.wte),
            wpe: gpu.upload(&w.wpe),
            blocks: w
                .blocks
                .iter()
                .map(|b| GpuBlock {
                    ln_1: norm(&b.ln_1),
                    qkv: linear(&b.qkv),
                    attn_out: linear(&b.attn_out),
                    ln_2: norm(&b.ln_2),
                    fc: linear(&b.fc),
                    fc_out: linear(&b.fc_out),
                })
                .collect(),
            ln_f: norm(&w.ln_f),
        }
    }
}

/// Logits for every position, `[T, V]`, left on the GPU.
pub fn forward(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<GpuTensor> {
    let h = hidden(gpu, w, ids)?;
    ops::linear(gpu, &h, &w.wte, None)
}

/// Final hidden states after `ln_f`, `[T, E]`.
pub fn hidden(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<GpuTensor> {
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    let eps = w.config.ln_eps;
    let mut x = ops::embed(gpu, &w.wte, &w.wpe, ids, 0)?;
    for b in &w.blocks {
        let h = ops::layer_norm(gpu, &x, &b.ln_1.gain, &b.ln_1.bias, eps)?;
        let qkv = ops::linear(gpu, &h, &b.qkv.w, Some(&b.qkv.b))?;
        let a = ops::causal_attention(gpu, &qkv, w.config.n_head)?;
        let a = ops::linear(gpu, &a, &b.attn_out.w, Some(&b.attn_out.b))?;
        x = ops::add(gpu, &x, &a)?;

        let h = ops::layer_norm(gpu, &x, &b.ln_2.gain, &b.ln_2.bias, eps)?;
        let m = ops::gelu(gpu, &ops::linear(gpu, &h, &b.fc.w, Some(&b.fc.b))?)?;
        let m = ops::linear(gpu, &m, &b.fc_out.w, Some(&b.fc_out.b))?;
        x = ops::add(gpu, &x, &m)?;
    }
    ops::layer_norm(gpu, &x, &w.ln_f.gain, &w.ln_f.bias, eps)
}

/// Next-token logits `[V]`, read back to the host (D28). The LM head runs on the last row only
/// (D26).
pub fn next_logits(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<Vec<f32>> {
    let h = hidden(gpu, w, ids)?;
    let last = ops::row(gpu, &h, ids.len() - 1)?;
    let logits = ops::linear(gpu, &last, &w.wte, None)?;
    Ok(gpu.read(&logits)?.data().to_vec())
}

/// Per-layer K and V for every position so far (D30). Rows `0..len` are valid; rows past `len`
/// hold stale data from earlier use and are never read (attention reads rows `0..=pos`).
pub struct KvCache {
    layers: Vec<(GpuTensor, GpuTensor)>,
    len: usize,
}

impl KvCache {
    /// Allocate K and V of `[n_ctx, E]` for every layer: 75.5 MB for GPT-2 124M.
    pub fn new(gpu: &Gpu, config: &Config) -> Self {
        let shape = [config.n_ctx, config.n_embd];
        KvCache {
            layers: (0..config.n_layer)
                .map(|_| (gpu.alloc(&shape), gpu.alloc(&shape)))
                .collect(),
            len: 0,
        }
    }

    /// Positions cached so far: the next token goes to position `len`.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start a new sequence. Old rows are simply overwritten as the new one grows.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// Run `ids` at positions `cache.len()..` and append their K and V to the cache. Returns the
/// final hidden states of these rows, `[T, E]`. Prefill is one call with the whole prompt;
/// each decode step is a call with one token.
pub fn hidden_cached(
    gpu: &Gpu,
    w: &GpuWeights,
    cache: &mut KvCache,
    ids: &[u32],
) -> Result<GpuTensor> {
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    // zip() below would silently stop at the shorter of the two: a cache built for a model with
    // fewer layers would leave the later blocks attending over nothing.
    if cache.layers.len() != w.blocks.len() {
        return Err(Error::Shape(format!(
            "KV cache has {} layers, the model {}",
            cache.layers.len(),
            w.blocks.len()
        )));
    }
    let start = cache.len;
    if start + ids.len() > w.config.n_ctx {
        return Err(Error::Input(format!(
            "positions {start}..{} exceed the context length {}",
            start + ids.len(),
            w.config.n_ctx
        )));
    }
    let eps = w.config.ln_eps;
    let mut x = ops::embed(gpu, &w.wte, &w.wpe, ids, start)?;
    for (b, (k, v)) in w.blocks.iter().zip(&cache.layers) {
        let h = ops::layer_norm(gpu, &x, &b.ln_1.gain, &b.ln_1.bias, eps)?;
        let qkv = ops::linear(gpu, &h, &b.qkv.w, Some(&b.qkv.b))?;
        ops::kv_write(gpu, &qkv, k, v, start)?;
        let a = ops::attention_cached(gpu, &qkv, k, v, start, w.config.n_head)?;
        let a = ops::linear(gpu, &a, &b.attn_out.w, Some(&b.attn_out.b))?;
        x = ops::add(gpu, &x, &a)?;

        let h = ops::layer_norm(gpu, &x, &b.ln_2.gain, &b.ln_2.bias, eps)?;
        let m = ops::gelu(gpu, &ops::linear(gpu, &h, &b.fc.w, Some(&b.fc.b))?)?;
        let m = ops::linear(gpu, &m, &b.fc_out.w, Some(&b.fc_out.b))?;
        x = ops::add(gpu, &x, &m)?;
    }
    // Only now: if an op above failed, the cache still describes the old sequence.
    cache.len += ids.len();
    ops::layer_norm(gpu, &x, &w.ln_f.gain, &w.ln_f.bias, eps)
}

/// `hidden_cached`, then the LM head on the last row (D26): the next-token logits after `ids`,
/// read back (D28).
pub fn extend(gpu: &Gpu, w: &GpuWeights, cache: &mut KvCache, ids: &[u32]) -> Result<Vec<f32>> {
    let h = hidden_cached(gpu, w, cache, ids)?;
    let last = ops::row(gpu, &h, ids.len() - 1)?;
    let logits = ops::linear(gpu, &last, &w.wte, None)?;
    Ok(gpu.read(&logits)?.data().to_vec())
}

/// Greedy decoding with a KV cache: one prefill call for the prompt, then one single-token call
/// per step. Same loop as the CPU reference; only the new tokens go through the model.
pub fn generate_greedy(
    gpu: &Gpu,
    w: &GpuWeights,
    prompt: &[u32],
    n: usize,
    on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    let mut cache = KvCache::new(gpu, &w.config);
    greedy(
        prompt,
        n,
        w.config.n_ctx,
        |ids| {
            let new = &ids[cache.len()..];
            extend(gpu, w, &mut cache, new)
        },
        on_token,
    )
}

/// Greedy decoding without a cache: every step recomputes the whole sequence (M3). Kept as the
/// baseline `ember bench` compares against.
pub fn generate_greedy_uncached(
    gpu: &Gpu,
    w: &GpuWeights,
    prompt: &[u32],
    n: usize,
    on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    greedy(
        prompt,
        n,
        w.config.n_ctx,
        |ids| next_logits(gpu, w, ids),
        on_token,
    )
}
