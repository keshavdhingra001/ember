//! GPT-2 on the GPU (M3): the forward pass of `forward.rs`, op for op, with the M2 kernels.
//! Weights are uploaded once (D24); each op submits its own dispatch (D25); nothing is read
//! back until the logits.

use crate::error::Result;
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
        return Err(crate::Error::Input("empty token sequence".into()));
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

/// Greedy decoding on the GPU; same loop as the CPU reference (no KV cache until M4).
pub fn generate_greedy(
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
