//! Llama-family models on the GPU: `forward.rs` op for op (D69), on the shared KV cache and
//! workspace (D74). Weights are uploaded once, every matrix transposed to `[in, out]` (D45), q|k|v
//! and gate|up already fused (D72). RoPE's tables go up once too, for every position the model
//! takes (D70). Two ways to run it, as for GPT-2: `forward` / `next_logits` recompute the whole
//! sequence, and `extend` runs only new tokens against a `KvCache`.

pub use crate::cache::KvCache;
use crate::cache::{Bufs, CacheLayout, CacheModel};
use crate::cpu;
use crate::error::{Error, Result};
use crate::gpt2::greedy;
use crate::gpu::{Gpu, Rec};
use crate::llama::{Config, Weights};
use crate::ops::{self, Epilogue, Heads};
use crate::tensor::{GpuTensor, numel};

pub struct GpuBlock {
    pub attn_norm: GpuTensor,
    /// `[E, (H + 2 KV) d]`.
    pub qkv: GpuTensor,
    /// `[E, E]`.
    pub o: GpuTensor,
    pub mlp_norm: GpuTensor,
    /// `[E, 2F]`: gate columns, then up.
    pub gate_up: GpuTensor,
    /// `[F, E]`.
    pub down: GpuTensor,
}

pub struct GpuWeights {
    pub config: Config,
    /// The token table transposed, `[E, V]`: the tied LM head's `[in, out]` and the only copy
    /// (D45); `ops::gather` reads a token's column.
    pub embed_t: GpuTensor,
    pub blocks: Vec<GpuBlock>,
    pub norm: GpuTensor,
    /// RoPE's `[n_ctx, d/2]` tables from `cpu::rope_tables` (D70): 2 MB for SmolLM2's 8192.
    pub cos: GpuTensor,
    pub sin: GpuTensor,
}

impl GpuWeights {
    pub fn upload(gpu: &Gpu, w: &Weights) -> Result<Self> {
        let t = |m| -> Result<GpuTensor> { Ok(gpu.upload(&cpu::transpose(m)?)) };
        let blocks = w
            .blocks
            .iter()
            .map(|b| {
                Ok(GpuBlock {
                    attn_norm: gpu.upload(&b.attn_norm),
                    qkv: t(&b.qkv)?,
                    o: t(&b.o)?,
                    mlp_norm: gpu.upload(&b.mlp_norm),
                    gate_up: t(&b.gate_up)?,
                    down: t(&b.down)?,
                })
            })
            .collect::<Result<_>>()?;
        let c = &w.config;
        let (cos, sin) = cpu::rope_tables(c.n_ctx, c.head_dim(), c.rope_theta);
        Ok(GpuWeights {
            config: c.clone(),
            embed_t: t(&w.embed)?,
            blocks,
            norm: gpu.upload(&w.norm),
            cos: gpu.upload(&cos),
            sin: gpu.upload(&sin),
        })
    }

    fn heads(&self) -> Heads {
        Heads {
            q: self.config.n_head,
            kv: self.config.n_kv_head,
        }
    }
}

/// The intermediates a forward pass writes; in the workspace each is a fixed buffer.
#[derive(Clone, Copy)]
enum Role {
    ResA,
    ResB,
    Norm,
    Qkv,
    Att,
    Parts,
    GateUp,
    Ff,
    Last,
    Logits,
}

impl Role {
    fn at(self) -> usize {
        self as usize
    }
}

/// The cache keeps 2048 positions unless asked for more (D76): at SmolLM2's full 8192 the K
/// and V alone would be 377 MB.
const DEFAULT_CTX: usize = 2048;

impl CacheModel for Config {
    fn cache_layout(&self, rows: usize) -> CacheLayout {
        let (e, v) = (self.n_embd, self.vocab_size);
        let floats = |role| match role {
            Role::ResA | Role::ResB | Role::Norm | Role::Att => rows * e,
            Role::Qkv => rows * (e + 2 * self.kv_dim()),
            Role::Parts => numel(&ops::attention_parts_shape(rows, 0, e, self.n_head)),
            Role::GateUp => rows * 2 * self.n_ff,
            Role::Ff => rows * self.n_ff,
            Role::Last => e,
            Role::Logits => v,
        };
        let roles = [
            Role::ResA,
            Role::ResB,
            Role::Norm,
            Role::Qkv,
            Role::Att,
            Role::Parts,
            Role::GateUp,
            Role::Ff,
            Role::Last,
            Role::Logits,
        ];
        debug_assert!(roles.iter().enumerate().all(|(i, &r)| r.at() == i));
        CacheLayout {
            n_layer: self.n_layer,
            n_ctx: rows,
            kv_width: self.kv_dim(),
            roles: roles.iter().map(|&r| floats(r)).collect(),
            readback: v,
        }
    }

    fn default_ctx(&self) -> usize {
        DEFAULT_CTX.min(self.n_ctx)
    }

    fn max_ctx(&self) -> usize {
        self.n_ctx
    }
}

/// How a block attends: over its own keys only (full recompute), or through a KV cache whose
/// rows `0..start` are already filled.
enum Attend<'c> {
    Recompute,
    Cached {
        layers: &'c [(GpuTensor, GpuTensor)],
        start: usize,
    },
}

/// The model up to the final norm for `ids`, recorded into `rec`; `ids_buf` holds the ids.
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
    let c = &w.config;
    let (t, e) = (ids.len(), c.n_embd);
    let start = match attend {
        Attend::Recompute => 0,
        Attend::Cached { start, .. } => *start,
    };
    let heads = w.heads();
    let mut x = bufs.out(gpu, Role::ResA.at(), &[t, e])?;
    ops::gather_into(rec, &w.embed_t, ids, ids_buf, &x)?;
    for (l, b) in w.blocks.iter().enumerate() {
        x = block(rec, bufs, w, b, &x, start, |rec, qkv, out| {
            let parts_shape = ops::attention_parts_shape(t, start, e, heads.q);
            let parts = bufs.out(gpu, Role::Parts.at(), &parts_shape)?;
            match attend {
                Attend::Recompute => {
                    let (k, v) = (gpu.alloc(&[t, c.kv_dim()]), gpu.alloc(&[t, c.kv_dim()]));
                    ops::kv_write_into(rec, qkv, &k, &v, 0, heads)?;
                    ops::attention_into(rec, qkv, &k, &v, 0, heads, &parts, out)
                }
                Attend::Cached { layers, start } => {
                    let (k, v) = &layers[l];
                    ops::kv_write_into(rec, qkv, k, v, *start, heads)?;
                    ops::attention_into(rec, qkv, k, v, *start, heads, &parts, out)
                }
            }
        })?;
    }
    let h = bufs.out(gpu, Role::Norm.at(), &[t, e])?;
    ops::rms_norm_into(rec, &x, &w.norm, c.rms_eps, &h)?;
    Ok(h)
}

/// One block, `x + o(attn(rope(qkv(norm(x)))))` then `+ down(silu(gate) * up)`, with `x` in
/// `ResA`; the result is in `ResA` again (the middle residual in `ResB`). RoPE turns q and k in
/// place in the qkv buffer before `attend` caches k (D70). Everything except `attend` is the same
/// op sequence for both paths, which the bitwise decode test relies on (D33).
#[allow(clippy::too_many_arguments)]
fn block(
    rec: &mut Rec,
    bufs: &Bufs,
    w: &GpuWeights,
    b: &GpuBlock,
    x: &GpuTensor,
    start: usize,
    attend: impl FnOnce(&mut Rec, &GpuTensor, &GpuTensor) -> Result<()>,
) -> Result<GpuTensor> {
    let gpu = rec.gpu;
    let c = &w.config;
    let (t, e) = (x.shape()[0], c.n_embd);
    let out = |role: Role, cols| bufs.out(gpu, role.at(), &[t, cols]);
    let none = Epilogue::None;

    let h = out(Role::Norm, e)?;
    ops::rms_norm_into(rec, x, &b.attn_norm, c.rms_eps, &h)?;
    let qkv = out(Role::Qkv, e + 2 * c.kv_dim())?;
    ops::linear_into(rec, &h, &b.qkv, None, none, &qkv)?;
    let n_rot = c.n_head + c.n_kv_head;
    ops::rope_into(rec, &qkv, n_rot, c.head_dim(), start, &w.cos, &w.sin)?;
    let a = out(Role::Att, e)?;
    attend(rec, &qkv, &a)?;
    let mid = out(Role::ResB, e)?;
    ops::linear_into(rec, &a, &b.o, None, Epilogue::Residual(x), &mid)?;

    ops::rms_norm_into(rec, &mid, &b.mlp_norm, c.rms_eps, &h)?;
    let gu = out(Role::GateUp, 2 * c.n_ff)?;
    ops::linear_into(rec, &h, &b.gate_up, None, none, &gu)?;
    let f = out(Role::Ff, c.n_ff)?;
    ops::silu_mul_into(rec, &gu, &f)?;
    let y = out(Role::ResA, e)?;
    ops::linear_into(rec, &f, &b.down, None, Epilogue::Residual(&mid), &y)?;
    Ok(y)
}

/// Logits for every position, `[T, V]`, left on the GPU.
pub fn forward(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<GpuTensor> {
    check_positions(w, ids.len())?;
    let mut rec = gpu.rec();
    let ids_buf = gpu.upload_u32(ids);
    let h = run(&mut rec, &Bufs::Alloc, w, ids, &ids_buf, &Attend::Recompute)?;
    let logits = gpu.alloc(&[ids.len(), w.config.vocab_size]);
    ops::linear_into(&mut rec, &h, &w.embed_t, None, Epilogue::None, &logits)?;
    rec.submit();
    Ok(logits)
}

/// Next-token logits `[V]`, recomputing the whole sequence, read back.
pub fn next_logits(gpu: &Gpu, w: &GpuWeights, ids: &[u32]) -> Result<Vec<f32>> {
    check_positions(w, ids.len())?;
    let mut rec = gpu.rec();
    let ids_buf = gpu.upload_u32(ids);
    let h = run(&mut rec, &Bufs::Alloc, w, ids, &ids_buf, &Attend::Recompute)?;
    let logits = last_logits(&mut rec, &Bufs::Alloc, w, &h)?;
    Ok(rec.read(&logits)?.data().to_vec())
}

fn check_positions(w: &GpuWeights, n: usize) -> Result<()> {
    if n > w.config.n_ctx {
        return Err(Error::Input(format!(
            "{n} tokens exceed the model's {} positions",
            w.config.n_ctx
        )));
    }
    Ok(())
}

/// The tied LM head on the last row of `h: [T, E]`, recorded: `[1, V]`.
fn last_logits(rec: &mut Rec, bufs: &Bufs, w: &GpuWeights, h: &GpuTensor) -> Result<GpuTensor> {
    let gpu = rec.gpu;
    let last = bufs.out(gpu, Role::Last.at(), &[1, w.config.n_embd])?;
    ops::row_into(rec, h, h.shape()[0] - 1, &last)?;
    let logits = bufs.out(gpu, Role::Logits.at(), &[1, w.config.vocab_size])?;
    ops::linear_into(rec, &last, &w.embed_t, None, Epilogue::None, &logits)?;
    Ok(logits)
}

/// Run `ids` at positions `cache.len()..`, append their K and V to the cache, and return the
/// next-token logits after them, read back. One recording and one submit on the cache's
/// workspace (D58, D59).
pub fn extend(gpu: &Gpu, w: &GpuWeights, cache: &mut KvCache, ids: &[u32]) -> Result<Vec<f32>> {
    let start = cache.check_extend(w.blocks.len(), ids.len())?;
    let ws = &mut cache.ws;
    let bufs = &ws.bufs;
    let mut rec = gpu.rec_with(&mut ws.binds);
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

/// Greedy decoding with a KV cache of the default size (D76): one prefill call for the prompt,
/// then one single-token call per step.
pub fn generate_greedy(
    gpu: &Gpu,
    w: &GpuWeights,
    prompt: &[u32],
    n: usize,
    on_token: impl FnMut(u32),
) -> Result<Vec<u32>> {
    let mut cache = KvCache::new(gpu, &w.config);
    let n_ctx = cache.n_ctx();
    greedy(
        prompt,
        n,
        n_ctx,
        |ids| {
            let new = &ids[cache.len()..];
            extend(gpu, w, &mut cache, new)
        },
        on_token,
    )
}

/// Greedy decoding without a cache: every step recomputes the whole sequence.
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
