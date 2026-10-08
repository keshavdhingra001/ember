//! The KV cache and the fixed workspace the cached path runs on (D30, D59), shared by every
//! model (D74). A model describes what it needs in a [`CacheLayout`]: layers, positions, the
//! width of a cached K or V row, and the floats each of its workspace roles holds. The roles
//! themselves (which intermediate lives where) stay the model's business.

use crate::error::{Error, Result};
use crate::gpu::{Binds, Gpu, byte_size};
use crate::tensor::{GpuTensor, numel};

/// What a model asks of a cache.
pub struct CacheLayout {
    pub n_layer: usize,
    /// Positions the cache holds (D76: may be fewer than the model was trained for).
    pub n_ctx: usize,
    /// Floats in one cached K (or V) row: `n_kv_head * d` (D71).
    pub kv_width: usize,
    /// Floats per workspace buffer, indexed by the model's role enum (`role as usize`).
    pub roles: Vec<usize>,
    /// Floats read back per call: the next-token logits.
    pub readback: usize,
}

/// A model that can run with a [`KvCache`].
pub trait CacheModel {
    /// The layout for a cache of `n_ctx` positions.
    fn cache_layout(&self, n_ctx: usize) -> CacheLayout;
    /// The cache size `KvCache::new` uses.
    fn default_ctx(&self) -> usize;
    /// Positions the model can attend to at all.
    fn max_ctx(&self) -> usize;
}

/// Per-layer K and V for every position so far (D30), and the workspace (D59). Rows `0..len`
/// are valid; rows past `len` hold stale data from earlier use and are never read (attention
/// reads rows `0..=pos`).
pub struct KvCache {
    pub(crate) layers: Vec<(GpuTensor, GpuTensor)>,
    pub(crate) len: usize,
    pub(crate) ws: Workspace,
    n_ctx: usize,
}

/// The fixed buffers of the cached path (D59): one per role, plus the token ids and the
/// readback buffer, and the bind groups made over them. Allocated once, so a steady-state
/// decode step creates no GPU objects (D60).
pub(crate) struct Workspace {
    pub bufs: WsBuffers,
    pub binds: Binds,
}

pub(crate) struct WsBuffers {
    /// Indexed by the model's `Role as usize`.
    pub bufs: Vec<GpuTensor>,
    pub ids: wgpu::Buffer,
    pub staging: wgpu::Buffer,
}

impl KvCache {
    /// A cache of the model's default size (GPT-2: its 1024 positions; SmolLM2: 2048, D76).
    pub fn new(gpu: &Gpu, model: &impl CacheModel) -> Self {
        Self::with_ctx(gpu, model, model.default_ctx()).expect("the default size is valid")
    }

    /// A cache of `n_ctx` positions, at most what the model was trained for.
    pub fn with_ctx(gpu: &Gpu, model: &impl CacheModel, n_ctx: usize) -> Result<Self> {
        if n_ctx == 0 || n_ctx > model.max_ctx() {
            return Err(Error::Input(format!(
                "a cache of {n_ctx} positions: the model takes 1..={}",
                model.max_ctx()
            )));
        }
        let l = model.cache_layout(n_ctx);
        let shape = [n_ctx, l.kv_width];
        // Check before allocating: the driver rejects an oversized buffer with a panic, not an
        // error. Attention's partials (D63) grow with positions squared and hit this first
        // (SmolLM2-135M at 8192 positions: 2.49 GB, D76).
        let floats = l
            .roles
            .iter()
            .copied()
            .chain([n_ctx * l.kv_width, l.readback]);
        check_fits(
            gpu,
            &format!("a cache of {n_ctx} positions"),
            floats.max().unwrap_or(0),
        )?;
        Ok(KvCache {
            layers: (0..l.n_layer)
                .map(|_| (gpu.alloc(&shape), gpu.alloc(&shape)))
                .collect(),
            len: 0,
            ws: Workspace {
                bufs: WsBuffers {
                    bufs: l.roles.iter().map(|&n| gpu.alloc(&[n])).collect(),
                    ids: gpu.upload_u32(&vec![0; n_ctx]),
                    staging: gpu.staging(l.readback),
                },
                binds: Binds::default(),
            },
            n_ctx,
        })
    }

    /// Positions cached so far: the next token goes to position `len`.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions the cache can hold.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
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

    /// The checks every model's `extend` makes before recording: the cache was built for this
    /// many layers, and `n` new positions fit. Returns the first new position.
    pub(crate) fn check_extend(&self, n_layer: usize, n: usize) -> Result<usize> {
        if n == 0 {
            return Err(Error::Input("empty token sequence".into()));
        }
        // A cache built for a model with fewer layers would leave the later blocks without one.
        if self.layers.len() != n_layer {
            return Err(Error::Shape(format!(
                "KV cache has {} layers, the model {n_layer}",
                self.layers.len()
            )));
        }
        let start = self.len;
        if start + n > self.n_ctx {
            return Err(Error::Input(format!(
                "positions {start}..{} exceed the cache's {} positions",
                start + n,
                self.n_ctx
            )));
        }
        if (self.ws.bufs.ids.size() as usize) < n * 4 {
            return Err(Error::Shape(format!(
                "workspace: {n} ids don't fit its id buffer"
            )));
        }
        Ok(start)
    }
}

/// Where a forward pass puts its intermediates: new buffers, or the workspace's (D59).
pub(crate) enum Bufs<'w> {
    Alloc,
    Workspace(&'w WsBuffers),
}

impl Bufs<'_> {
    /// A tensor of `shape` for role `role` (the model's `Role as usize`): a new buffer, or a
    /// view of the workspace's.
    pub fn out(&self, gpu: &Gpu, role: usize, shape: &[usize]) -> Result<GpuTensor> {
        match self {
            Bufs::Alloc => {
                check_fits(gpu, &format!("a {shape:?} intermediate"), numel(shape))?;
                Ok(gpu.alloc(shape))
            }
            Bufs::Workspace(ws) => ws.bufs[role].view(shape),
        }
    }
}

/// An error if a buffer of `floats` f32s is more than the device takes. Checked before
/// allocating: the driver rejects an oversized buffer with a panic, not an error. Attention's
/// partials (D63) grow with positions squared and hit this first (SmolLM2-135M at 8192
/// positions: 2.49 GB, D76).
fn check_fits(gpu: &Gpu, what: &str, floats: usize) -> Result<()> {
    let limit = gpu
        .limits
        .max_storage_buffer_binding_size
        .min(gpu.limits.max_buffer_size);
    if byte_size(floats) > limit {
        return Err(Error::Input(format!(
            "{what} needs a {} MB buffer, which exceeds the device's {} MB",
            byte_size(floats) >> 20,
            limit >> 20
        )));
    }
    Ok(())
}
