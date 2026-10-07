//! GPT-2 on the GPU (M3): the forward pass of `forward.rs`, op for op, with the M2 kernels.
//! Weights are uploaded once (D24), every matrix in the `[in, out]` layout the GPU's matmul
//! kernels read fast (D45). A call records all its ops into one command encoder and submits
//! once (D58); nothing is read back until the logits.
//!
//! Two ways to run it: `forward` / `next_logits` recompute the whole sequence (M3, the
//! comparison partner for D33), and `extend` runs only new tokens against a `KvCache` (M4).

use crate::cpu;
use crate::error::{Error, Result};
use crate::gpt2::forward::greedy;
use crate::gpt2::{Config, Linear, Norm, Weights};
use crate::gpu::{Binds, Gpu, Rec};
use crate::ops::{self, Epilogue};
use crate::tensor::{GpuTensor, Tensor};

pub struct GpuLinear {
    /// `[in, out]`: the transpose of the CPU's `[out, in]` (D45).
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

/// `Weights` in GPU buffers (D24), matrices transposed to `[in, out]` (D45).
pub struct GpuWeights {
    pub config: Config,
    /// The token table transposed, `[E, V]`: the tied LM head's `[in, out]`, and the only copy
    /// (D45). `ops::embed` reads a token's column.
    pub wte_t: GpuTensor,
    pub wpe: GpuTensor,
    pub blocks: Vec<GpuBlock>,
    pub ln_f: GpuNorm,
}

impl GpuWeights {
    pub fn upload(gpu: &Gpu, w: &Weights) -> Result<Self> {
        let transposed = |t: &Tensor| -> Result<GpuTensor> { Ok(gpu.upload(&cpu::transpose(t)?)) };
        let linear = |l: &Linear| -> Result<GpuLinear> {
            Ok(GpuLinear {
                w: transposed(&l.w)?,
                b: gpu.upload(&l.b),
            })
        };
        let norm = |n: &Norm| GpuNorm {
            gain: gpu.upload(&n.gain),
            bias: gpu.upload(&n.bias),
        };
        let blocks = w
            .blocks
            .iter()
            .map(|b| {
                Ok(GpuBlock {
                    ln_1: norm(&b.ln_1),
                    qkv: linear(&b.qkv)?,
                    attn_out: linear(&b.attn_out)?,
                    ln_2: norm(&b.ln_2),
                    fc: linear(&b.fc)?,
                    fc_out: linear(&b.fc_out)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(GpuWeights {
            config: w.config.clone(),
            wte_t: transposed(&w.wte)?,
            wpe: gpu.upload(&w.wpe),
            blocks,
            ln_f: norm(&w.ln_f),
        })
    }
}

/// Logits for every position, `[T, V]`, left on the GPU.
pub fn forward(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<GpuTensor> {
    let mut rec = gpu.rec();
    let h = run(
        &mut rec,
        &Bufs::Alloc,
        w,
        ids,
        &gpu.upload_u32(ids),
        &Attend::Recompute,
    )?;
    let logits = gpu.alloc(&[ids.len(), w.config.vocab_size]);
    ops::linear_into(&mut rec, &h, &w.wte_t, None, Epilogue::None, &logits)?;
    rec.submit();
    Ok(logits)
}

/// Final hidden states after `ln_f`, `[T, E]`.
pub fn hidden(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<GpuTensor> {
    let mut rec = gpu.rec();
    let h = run(
        &mut rec,
        &Bufs::Alloc,
        w,
        ids,
        &gpu.upload_u32(ids),
        &Attend::Recompute,
    )?;
    rec.submit();
    Ok(h)
}

/// Where a forward pass puts its intermediates: new buffers, or the workspace's (D59).
enum Bufs<'w> {
    Alloc,
    Workspace(&'w WsBuffers),
}

/// The intermediates a forward pass writes. In the workspace each one is a fixed buffer; the
/// residual stream alternates between `ResA` and `ResB`.
#[derive(Clone, Copy)]
enum Role {
    ResA,
    ResB,
    Norm,
    Qkv,
    Att,
    Fc,
    Last,
    Logits,
}

impl Bufs<'_> {
    /// A tensor of `shape` for `role`'s output: a new buffer, or a view of the workspace's.
    fn out(&self, gpu: &Gpu, role: Role, shape: &[usize]) -> Result<GpuTensor> {
        match self {
            Bufs::Alloc => Ok(gpu.alloc(shape)),
            Bufs::Workspace(ws) => ws.bufs[role as usize].view(shape),
        }
    }
}

/// How a block attends: over its own keys only (full recompute, M3), or through a KV cache
/// whose rows `0..start` are already filled (M4).
enum Attend<'c> {
    Recompute,
    Cached {
        layers: &'c [(GpuTensor, GpuTensor)],
        start: usize,
    },
}

/// The model up to `ln_f` for `ids`, recorded into `rec`. `ids_buf` holds the ids (or gets
/// them written in this recording).
fn run(
    rec: &mut Rec,
    bufs: &Bufs,
    w: &GpuWeights,
    ids: &[u32],
    ids_buf: &wgpu::Buffer,
    attend: &Attend,
) -> Result<GpuTensor> {
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    let gpu = rec.gpu;
    let (t, e) = (ids.len(), w.config.n_embd);
    let start = match attend {
        Attend::Recompute => 0,
        Attend::Cached { start, .. } => *start,
    };
    let mut x = bufs.out(gpu, Role::ResA, &[t, e])?;
    ops::embed_into(rec, &w.wte_t, &w.wpe, ids, ids_buf, start, &x)?;
    let n_head = w.config.n_head;
    for (l, b) in w.blocks.iter().enumerate() {
        x = block(rec, bufs, w, b, &x, |rec, qkv, out| match attend {
            Attend::Recompute => {
                // A temporary cache of just these rows: the M2 op, a special case of D31.
                let (k, v) = (gpu.alloc(&[t, e]), gpu.alloc(&[t, e]));
                ops::kv_write_into(rec, qkv, &k, &v, 0)?;
                ops::attention_into(rec, qkv, &k, &v, 0, n_head, out)
            }
            Attend::Cached { layers, start } => {
                let (k, v) = &layers[l];
                ops::kv_write_into(rec, qkv, k, v, *start)?;
                ops::attention_into(rec, qkv, k, v, *start, n_head, out)
            }
        })?;
    }
    let h = bufs.out(gpu, Role::Norm, &[t, e])?;
    ops::layer_norm_into(rec, &x, &w.ln_f.gain, &w.ln_f.bias, w.config.ln_eps, &h)?;
    Ok(h)
}

/// One transformer block, `x + attn(ln_1(x))` then `+ mlp(ln_2(x))`, with `x` in `ResA`; the
/// result is in `ResA` again (the middle residual in `ResB`). `attend` writes the heads'
/// outputs `[T, E]` for the block's `qkv: [T, 3E]` into its last argument: full causal
/// attention, or attention through a KV cache. Everything else is the same op sequence for
/// both paths, which D33's bitwise test relies on.
fn block(
    rec: &mut Rec,
    bufs: &Bufs,
    w: &GpuWeights,
    b: &GpuBlock,
    x: &GpuTensor,
    attend: impl FnOnce(&mut Rec, &GpuTensor, &GpuTensor) -> Result<()>,
) -> Result<GpuTensor> {
    let gpu = rec.gpu;
    let eps = w.config.ln_eps;
    let &[t, e] = x.shape() else {
        return Err(Error::Shape(format!("block: x {:?}", x.shape())));
    };
    let out = |role, cols| bufs.out(gpu, role, &[t, cols]);

    let h = out(Role::Norm, e)?;
    ops::layer_norm_into(rec, x, &b.ln_1.gain, &b.ln_1.bias, eps, &h)?;
    let qkv = out(Role::Qkv, 3 * e)?;
    let none = Epilogue::None;
    ops::linear_into(rec, &h, &b.qkv.w, Some(&b.qkv.b), none, &qkv)?;
    let a = out(Role::Att, e)?;
    attend(rec, &qkv, &a)?;
    // The residual adds and GELU happen in the linear kernels' epilogue (D62).
    let mid = out(Role::ResB, e)?;
    let (attn_out, fc, fc_out) = (&b.attn_out, &b.fc, &b.fc_out);
    ops::linear_into(
        rec,
        &a,
        &attn_out.w,
        Some(&attn_out.b),
        Epilogue::Residual(x),
        &mid,
    )?;

    ops::layer_norm_into(rec, &mid, &b.ln_2.gain, &b.ln_2.bias, eps, &h)?;
    let f = out(Role::Fc, fc.b.len())?;
    ops::linear_into(rec, &h, &fc.w, Some(&fc.b), Epilogue::Gelu, &f)?;
    let y = out(Role::ResA, e)?;
    ops::linear_into(
        rec,
        &f,
        &fc_out.w,
        Some(&fc_out.b),
        Epilogue::Residual(&mid),
        &y,
    )?;
    Ok(y)
}

/// Next-token logits `[V]`, read back to the host (D28). The LM head runs on the last row only
/// (D26).
pub fn next_logits(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<Vec<f32>> {
    let mut rec = gpu.rec();
    let h = run(
        &mut rec,
        &Bufs::Alloc,
        w,
        ids,
        &gpu.upload_u32(ids),
        &Attend::Recompute,
    )?;
    let logits = last_logits(&mut rec, &Bufs::Alloc, w, &h)?;
    Ok(rec.read(&logits)?.data().to_vec())
}

/// The tied LM head on the last row of `h: [T, E]`, recorded: `[1, V]`.
fn last_logits(rec: &mut Rec, bufs: &Bufs, w: &GpuWeights, h: &GpuTensor) -> Result<GpuTensor> {
    let gpu = rec.gpu;
    let last = bufs.out(gpu, Role::Last, &[1, w.config.n_embd])?;
    ops::row_into(rec, h, h.shape()[0] - 1, &last)?;
    let logits = bufs.out(gpu, Role::Logits, &[1, w.config.vocab_size])?;
    ops::linear_into(rec, &last, &w.wte_t, None, Epilogue::None, &logits)?;
    Ok(logits)
}

/// Per-layer K and V for every position so far (D30), and the workspace the cached path runs
/// on (D59). Rows `0..len` are valid; rows past `len` hold stale data from earlier use and are
/// never read (attention reads rows `0..=pos`).
pub struct KvCache {
    layers: Vec<(GpuTensor, GpuTensor)>,
    len: usize,
    ws: Workspace,
}

/// The fixed buffers of the cached path (D59): one per `Role`, each for `n_ctx` rows, plus the
/// token ids and the readback buffer, and the bind groups made over them. Allocated once, so a
/// steady-state decode step creates no GPU objects (D60).
struct Workspace {
    bufs: WsBuffers,
    binds: Binds,
}

struct WsBuffers {
    /// Indexed by `Role as usize`.
    bufs: Vec<GpuTensor>,
    ids: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl Workspace {
    fn new(gpu: &Gpu, config: &Config) -> Self {
        let (rows, e, v) = (config.n_ctx, config.n_embd, config.vocab_size);
        // GPT-2's MLP is 4E wide; another width fails `Bufs::out`'s size check.
        let floats = |role| match role {
            Role::ResA | Role::ResB | Role::Norm | Role::Att => rows * e,
            Role::Qkv => rows * 3 * e,
            Role::Fc => rows * 4 * e,
            Role::Last => e,
            Role::Logits => v,
        };
        let roles = [
            Role::ResA,
            Role::ResB,
            Role::Norm,
            Role::Qkv,
            Role::Att,
            Role::Fc,
            Role::Last,
            Role::Logits,
        ];
        debug_assert!(roles.iter().enumerate().all(|(i, &r)| r as usize == i));
        Workspace {
            bufs: WsBuffers {
                bufs: roles.iter().map(|&r| gpu.alloc(&[floats(r)])).collect(),
                ids: gpu.upload_u32(&vec![0; rows]),
                staging: gpu.staging(v),
            },
            binds: Binds::default(),
        }
    }
}

impl KvCache {
    /// Allocate K and V of `[n_ctx, E]` for every layer (75.5 MB for GPT-2 124M) and the
    /// workspace (D59; 35 MB for GPT-2 124M).
    pub fn new(gpu: &Gpu, config: &Config) -> Self {
        let shape = [config.n_ctx, config.n_embd];
        KvCache {
            layers: (0..config.n_layer)
                .map(|_| (gpu.alloc(&shape), gpu.alloc(&shape)))
                .collect(),
            len: 0,
            ws: Workspace::new(gpu, config),
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

    /// Forget every position from `len` on, keeping the first `len` (the next token goes to
    /// position `len`). Rows `0..len` are untouched, so they stay exactly what the prefix
    /// computed; the rest are overwritten as the sequence grows again. Fails past the end.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(Error::Input(format!(
                "truncate to {len}: the cache holds only {}",
                self.len
            )));
        }
        self.len = len;
        Ok(())
    }

    /// Bind groups the workspace has made so far (D59): constant once every shape has run.
    pub fn bind_groups(&self) -> usize {
        self.ws.binds.len()
    }
}

/// Run `ids` at positions `cache.len()..`, append their K and V to the cache, and return the
/// next-token logits after them, read back (D28). Prefill is one call with the whole prompt;
/// each decode step is a call with one token. One recording and one submit (D58) on the
/// cache's workspace (D59).
pub fn extend(gpu: &Gpu, w: &GpuWeights, cache: &mut KvCache, ids: &[u32]) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Err(Error::Input("empty token sequence".into()));
    }
    // A cache built for a model with fewer layers would leave the later blocks without one.
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
    let Workspace { bufs, binds } = &mut cache.ws;
    let bufs = &*bufs;
    let mut rec = gpu.rec_with(binds);
    if (bufs.ids.size() as usize) < ids.len() * 4 {
        return Err(Error::Shape(format!(
            "workspace: {} ids don't fit its id buffer",
            ids.len()
        )));
    }
    rec.write(&bufs.ids, ids);
    let ws = Bufs::Workspace(bufs);
    let attend = Attend::Cached {
        layers: &cache.layers,
        start,
    };
    let h = run(&mut rec, &ws, w, ids, &bufs.ids, &attend)?;
    let logits = last_logits(&mut rec, &ws, w, &h)?;
    let out = rec.read_via(&logits, &bufs.staging)?.data().to_vec();
    // Only now: if anything above failed, the cache still describes the old sequence.
    cache.len += ids.len();
    Ok(out)
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
