//! GPU ops: host-side wrappers that check shapes and record a kernel. Each one has a CPU twin
//! in `cpu.rs` that it is tested against.
//!
//! The ops the model runs come in two forms (D61): `op_into(rec, …, out)` records into a
//! caller's `Rec` and writes a caller's output tensor (the workspace path, D59), and `op(gpu, …)`
//! allocates the output, records into a recording of its own and submits it. Both run the same
//! recording code.

use crate::error::{Error, Result};
use crate::gpu::{Gpu, Kernel, Rec};
use crate::shape;
use crate::tensor::{GpuTensor, numel};

/// Keys per attention chunk (D63). Must match `KC` in attention.wgsl and attention_combine.wgsl.
pub const ATTENTION_CHUNK: usize = 64;

/// Largest head dimension the attention kernel takes: a chunk's keys are staged in shared
/// memory in rows of this many floats. Must match `MAX_D` in attention.wgsl.
pub const ATTENTION_MAX_D: usize = 64;

/// Threads per workgroup for 1-D elementwise kernels. Must match `WG` in the shaders.
pub const ELEMENTWISE_WG: u32 = 256;

/// Elementwise `a + b` on the GPU.
pub fn add(gpu: &Gpu, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    add_with_max_groups(gpu, a, b, gpu.max_groups())
}

/// `add` with a cap on the workgroup count. Tests pass a tiny cap to exercise the
/// grid-stride loop without allocating 17M-element buffers.
pub fn add_with_max_groups(
    gpu: &Gpu,
    a: &GpuTensor,
    b: &GpuTensor,
    max_groups: u32,
) -> Result<GpuTensor> {
    shape::same("add", a.shape(), b.shape())?;
    once(gpu, a.shape(), |rec, out| {
        add_groups(rec, a, b, out, max_groups)
    })
}

/// `add` into `out` (D61).
pub fn add_into(rec: &mut Rec, a: &GpuTensor, b: &GpuTensor, out: &GpuTensor) -> Result<()> {
    add_groups(rec, a, b, out, rec.gpu.max_groups())
}

fn add_groups(
    rec: &mut Rec,
    a: &GpuTensor,
    b: &GpuTensor,
    out: &GpuTensor,
    max_groups: u32,
) -> Result<()> {
    shape::same("add", a.shape(), b.shape())?;
    check_out("add", out, a.shape())?;
    let n = a.len();
    if n == 0 {
        return Ok(());
    }
    let params = Params4::new(len_u32(n)?, 0, 0, 0);
    let kernel = &rec.gpu.kernels.add;
    elementwise(
        rec,
        kernel,
        &[&a.buffer, &b.buffer, &out.buffer],
        params,
        max_groups,
    )
}

/// The one-op form (D61): allocate `shape`, record `f` writing it, submit.
fn once(
    gpu: &Gpu,
    shape: &[usize],
    f: impl FnOnce(&mut Rec, &GpuTensor) -> Result<()>,
) -> Result<GpuTensor> {
    let out = gpu.alloc(shape);
    let mut rec = gpu.rec();
    f(&mut rec, &out)?;
    rec.submit();
    Ok(out)
}

/// An `_into` op's output must have the op's shape, and its buffer may be larger (a workspace
/// view, D59) but not smaller: kernels bound their writes by the shape, never the buffer (D9).
fn check_out(op: &str, out: &GpuTensor, shape: &[usize]) -> Result<()> {
    if out.shape() != shape {
        return Err(Error::Shape(format!(
            "{op}: output {:?}, expected {shape:?}",
            out.shape()
        )));
    }
    if (out.buffer.size() as usize) < out.len() * 4 {
        return Err(Error::Shape(format!(
            "{op}: output buffer of {} bytes holds less than {:?}",
            out.buffer.size(),
            out.shape()
        )));
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

impl Params4 {
    fn new(a: u32, b: u32, c: u32, d: u32) -> Self {
        Params4 { a, b, c, d }
    }
}

/// Record a grid-stride elementwise kernel (D8): `buffers` in binding order, then `params` as
/// the last binding, whose first field is the element count.
fn elementwise(
    rec: &mut Rec,
    kernel: &Kernel,
    buffers: &[&wgpu::Buffer],
    params: Params4,
    max_groups: u32,
) -> Result<()> {
    let groups = elementwise_groups(params.a as usize, max_groups);
    rec.dispatch(kernel, buffers, bytemuck::bytes_of(&params), (groups, 1, 1))
}

/// Elementwise GPT-2 GELU (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    once(gpu, x.shape(), |rec, out| gelu_into(rec, x, out))
}

/// `gelu` into `out` (D61).
pub fn gelu_into(rec: &mut Rec, x: &GpuTensor, out: &GpuTensor) -> Result<()> {
    check_out("gelu", out, x.shape())?;
    if x.is_empty() {
        return Ok(());
    }
    let kernel = &rec.gpu.kernels.gelu;
    elementwise(
        rec,
        kernel,
        &[&x.buffer, &out.buffer],
        Params4::new(len_u32(x.len())?, 0, 0, 0),
        rec.gpu.max_groups(),
    )
}

/// `out[t] = wte[ids[t]] + wpe[start + t]`: token embedding plus the embedding of its absolute
/// position (`start` > 0 when decoding after a cached prefix, D32). The token table comes
/// transposed, `wte_t: [E, V]` (D45). Ids and positions are checked here: the kernel would
/// silently read the wrong row (or a clamped one, D9).
pub fn embed(
    gpu: &Gpu,
    wte_t: &GpuTensor,
    wpe: &GpuTensor,
    ids: &[u32],
    start: usize,
) -> Result<GpuTensor> {
    let (e, _) = embed_dims(wte_t, wpe, ids, start)?;
    let ids_buf = gpu.upload_u32(ids);
    once(gpu, &[ids.len(), e], |rec, out| {
        embed_into(rec, wte_t, wpe, ids, &ids_buf, start, out)
    })
}

/// `embed` into `out` (D61), with the ids taken from `ids_buf`, which must already hold `ids`
/// (or have them written in this recording). `ids` itself is only checked.
pub fn embed_into(
    rec: &mut Rec,
    wte_t: &GpuTensor,
    wpe: &GpuTensor,
    ids: &[u32],
    ids_buf: &wgpu::Buffer,
    start: usize,
    out: &GpuTensor,
) -> Result<()> {
    let (e, v) = embed_dims(wte_t, wpe, ids, start)?;
    check_out("embed", out, &[ids.len(), e])?;
    if ids.is_empty() {
        return Ok(());
    }
    if (ids_buf.size() as usize) < ids.len() * 4 {
        return Err(Error::Shape(format!(
            "embed: id buffer of {} bytes for {} ids",
            ids_buf.size(),
            ids.len()
        )));
    }
    let n = len_u32(ids.len() * e)?;
    let kernel = &rec.gpu.kernels.embed;
    elementwise(
        rec,
        kernel,
        &[&wte_t.buffer, &wpe.buffer, ids_buf, &out.buffer],
        Params4::new(n, e as u32, start as u32, len_u32(v)?),
        rec.gpu.max_groups(),
    )
}

/// `(E, V)` after checking the tables, the ids and the positions.
fn embed_dims(
    wte_t: &GpuTensor,
    wpe: &GpuTensor,
    ids: &[u32],
    start: usize,
) -> Result<(usize, usize)> {
    let &[e_rows, v] = wte_t.shape() else {
        return Err(Error::Shape(format!(
            "embed: wte_t {:?} must be 2-D",
            wte_t.shape()
        )));
    };
    Ok((shape::embed(&[v, e_rows], wpe.shape(), ids, start)?, v))
}

/// `[rows, cols]` for the row kernels, which launch one workgroup per row.
fn rows_cols(gpu: &Gpu, op: &str, x: &GpuTensor) -> Result<(u32, u32)> {
    let &[rows, cols] = x.shape() else {
        return Err(Error::Shape(format!("{op}: x {:?} must be 2-D", x.shape())));
    };
    if rows > gpu.max_groups() as usize {
        return Err(Error::Shape(format!(
            "{op}: {rows} rows exceed one dispatch dimension"
        )));
    }
    len_u32(rows * cols)?;
    Ok((rows as u32, cols as u32))
}

/// Softmax over the last dimension of `x: [rows, cols]`.
pub fn softmax_rows(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    let (rows, cols) = rows_cols(gpu, "softmax_rows", x)?;
    once(gpu, x.shape(), |rec, out| {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        rec.dispatch(
            &gpu.kernels.softmax,
            &[&x.buffer, &out.buffer],
            bytemuck::bytes_of(&Params4::new(cols, 0, 0, 0)),
            (rows, 1, 1),
        )
    })
}

/// LayerNorm over the last dimension of `x: [rows, cols]` with `gain`, `bias`: `[cols]`.
pub fn layer_norm(
    gpu: &Gpu,
    x: &GpuTensor,
    gain: &GpuTensor,
    bias: &GpuTensor,
    eps: f32,
) -> Result<GpuTensor> {
    once(gpu, x.shape(), |rec, out| {
        layer_norm_into(rec, x, gain, bias, eps, out)
    })
}

/// `layer_norm` into `out` (D61).
pub fn layer_norm_into(
    rec: &mut Rec,
    x: &GpuTensor,
    gain: &GpuTensor,
    bias: &GpuTensor,
    eps: f32,
    out: &GpuTensor,
) -> Result<()> {
    shape::layer_norm(x.shape(), gain.shape(), bias.shape())?;
    let (rows, cols) = rows_cols(rec.gpu, "layer_norm", x)?;
    check_out("layer_norm", out, x.shape())?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    rec.dispatch(
        &rec.gpu.kernels.layer_norm,
        &[&x.buffer, &gain.buffer, &bias.buffer, &out.buffer],
        bytemuck::bytes_of(&Params4::new(cols, eps.to_bits(), 0, 0)),
        (rows, 1, 1),
    )
}

/// RMSNorm of each row of `x: [rows, cols]` with `gain: [cols]` (see `cpu::rms_norm`).
pub fn rms_norm(gpu: &Gpu, x: &GpuTensor, gain: &GpuTensor, eps: f32) -> Result<GpuTensor> {
    once(gpu, x.shape(), |rec, out| {
        rms_norm_into(rec, x, gain, eps, out)
    })
}

/// `rms_norm` into `out` (D61).
pub fn rms_norm_into(
    rec: &mut Rec,
    x: &GpuTensor,
    gain: &GpuTensor,
    eps: f32,
    out: &GpuTensor,
) -> Result<()> {
    shape::rms_norm(x.shape(), gain.shape())?;
    let (rows, cols) = rows_cols(rec.gpu, "rms_norm", x)?;
    check_out("rms_norm", out, x.shape())?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    rec.dispatch(
        &rec.gpu.kernels.rms_norm,
        &[&x.buffer, &gain.buffer, &out.buffer],
        bytemuck::bytes_of(&Params4::new(cols, eps.to_bits(), 0, 0)),
        (rows, 1, 1),
    )
}

/// SwiGLU's gate: `gu: [T, 2F]` (gate columns, then up) -> `silu(gate) * up`, `[T, F]`.
pub fn silu_mul(gpu: &Gpu, gu: &GpuTensor) -> Result<GpuTensor> {
    let (t, f) = silu_mul_dims(gu)?;
    once(gpu, &[t, f], |rec, out| silu_mul_into(rec, gu, out))
}

fn silu_mul_dims(gu: &GpuTensor) -> Result<(usize, usize)> {
    match *gu.shape() {
        [t, two_f] if two_f.is_multiple_of(2) => Ok((t, two_f / 2)),
        _ => Err(Error::Shape(format!(
            "silu_mul: {:?} must be [T, 2F]",
            gu.shape()
        ))),
    }
}

/// `silu_mul` into `out` (D61).
pub fn silu_mul_into(rec: &mut Rec, gu: &GpuTensor, out: &GpuTensor) -> Result<()> {
    let (t, f) = silu_mul_dims(gu)?;
    check_out("silu_mul", out, &[t, f])?;
    if t * f == 0 {
        return Ok(());
    }
    let kernel = &rec.gpu.kernels.silu_mul;
    elementwise(
        rec,
        kernel,
        &[&gu.buffer, &out.buffer],
        Params4::new(len_u32(t * f)?, f as u32, 0, 0),
        rec.gpu.max_groups(),
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    n: u32,
    w: u32,
    half: u32,
    start: u32,
    n_rot: u32,
    _pad: [u32; 3],
}

/// Rotary position embedding (D70) on a copy of `x: [T, W]`: the first `n_rot` heads of width
/// `d` of each row, rows at positions `start..`, tables `[n_pos, d/2]` from `cpu::rope_tables`.
pub fn rope(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rot: usize,
    d: usize,
    start: usize,
    cos: &GpuTensor,
    sin: &GpuTensor,
) -> Result<GpuTensor> {
    let out = gpu.alloc(x.shape());
    let mut rec = gpu.rec();
    rec.copy(&x.buffer, 0, &out.buffer, crate::gpu::byte_size(x.len()));
    rope_into(&mut rec, &out, n_rot, d, start, cos, sin)?;
    rec.submit();
    Ok(out)
}

/// `rope` recorded into `rec`, in place on `x` (each invocation owns one pair, so in place is
/// safe; the workspace path rotates its qkv buffer without a second one).
pub fn rope_into(
    rec: &mut Rec,
    x: &GpuTensor,
    n_rot: usize,
    d: usize,
    start: usize,
    cos: &GpuTensor,
    sin: &GpuTensor,
) -> Result<()> {
    let (t, w) = shape::rope(x.shape(), n_rot, d, start, cos.shape(), sin.shape())?;
    let n = len_u32(t * n_rot * (d / 2))?;
    len_u32(numel(cos.shape()))?;
    if n == 0 {
        return Ok(());
    }
    let params = RopeParams {
        n,
        w: w as u32,
        half: (d / 2) as u32,
        start: start as u32,
        n_rot: n_rot as u32,
        _pad: [0; 3],
    };
    let groups = elementwise_groups(n as usize, rec.gpu.max_groups());
    rec.dispatch(
        &rec.gpu.kernels.rope,
        &[&x.buffer, &cos.buffer, &sin.buffer],
        bytemuck::bytes_of(&params),
        (groups, 1, 1),
    )
}

/// Threads per side of `linear_naive`'s 2-D workgroup. Must match `@workgroup_size(16, 16)`.
const NAIVE_TILE: usize = 16;

/// Rows and columns of the output per `matmul` workgroup. Must match `BM` and `BN` in matmul.wgsl.
const MATMUL_TILE: usize = 64;

/// Threads per `matvec` workgroup. Must match `WG` in matvec.wgsl.
const MATVEC_WG: usize = 256;

/// Most rows `matvec_rows` takes (D57). Must match `ROWS` in matvec_rows.wgsl.
const MATVEC_ROWS: usize = 8;

/// Up to this many outputs, the matvec splits K four ways (D56): 768 outputs are 768 threads
/// without the split, too few to keep the memory busy. Wider matrices have enough threads, and
/// the split's shorter runs per row cost more than it saves. Measured at 768 (split wins) and
/// 2304 (split loses); the boundary between them is not measured.
const MATVEC_SPLIT: usize = 1024;

/// From this many outputs on, the matvec runs without lookahead (D48): measured on GPT-2's
/// shapes, lookahead wins at 2304-3072 outputs and loses at the LM head's 50257. The boundary
/// between them is not measured.
const MATVEC_WIDE: usize = 16384;

/// What a linear kernel does after the bias (D62).
#[derive(Clone, Copy)]
pub enum Epilogue<'a> {
    None,
    /// GELU of every output (the MLP's `fc`).
    Gelu,
    /// `res + y`, `res` of the output's shape: the residual add after `attn_out` and `fc_out`.
    Residual(&'a GpuTensor),
}

impl Epilogue<'_> {
    /// Index into the kernels' epilogue arrays; must match `EPILOGUE` in epilogue.wgsl.
    fn index(self) -> usize {
        match self {
            Epilogue::None => 0,
            Epilogue::Gelu => 1,
            Epilogue::Residual(_) => 2,
        }
    }
}

/// `y = x @ w + b`: `x: [T, in]`, `w: [in, out]` (the GPU layout, D45), `b: [out]` or none.
/// One row (a decode step) goes to the matrix-vector kernel, up to 8 rows (a short prefill) to
/// its multi-row form, more rows to the tiled matmul (D43, D44, D57). All compute each output as
/// the same chunked sum of fused multiply-adds (D51),
/// so a row's bits don't depend on which kernel ran it or what else was in the batch (D46).
pub fn linear(gpu: &Gpu, x: &GpuTensor, w: &GpuTensor, b: Option<&GpuTensor>) -> Result<GpuTensor> {
    let (t, _, n_out) = shape::linear_in_out(x.shape(), w.shape(), b.map(GpuTensor::shape))?;
    once(gpu, &[t, n_out], |rec, out| {
        linear_into(rec, x, w, b, Epilogue::None, out)
    })
}

/// `linear` into `out` (D61), then `epilogue` (D62).
pub fn linear_into(
    rec: &mut Rec,
    x: &GpuTensor,
    w: &GpuTensor,
    b: Option<&GpuTensor>,
    epilogue: Epilogue,
    out: &GpuTensor,
) -> Result<()> {
    let (t, n_in, n_out) = shape::linear_in_out(x.shape(), w.shape(), b.map(GpuTensor::shape))?;
    let dims = (t, n_in, n_out);
    let gpu = rec.gpu;
    let ep = epilogue.index();
    if t <= MATVEC_ROWS {
        let k = &gpu.kernels;
        let (kernel, slices) = match (t == 1, n_out) {
            (true, ..=MATVEC_SPLIT) => (&k.matvec_split[ep], 4),
            (true, ..MATVEC_WIDE) => (&k.matvec[ep], 1),
            (true, _) => (&k.matvec_wide[ep], 1),
            (false, ..=MATVEC_SPLIT) => (&k.matvec_rows_split[ep], 4),
            (false, ..MATVEC_WIDE) => (&k.matvec_rows[ep], 1),
            (false, _) => (&k.matvec_rows_wide[ep], 1),
        };
        // Must match the shader's outs = WG / SLICES.
        let groups = (n_out.div_ceil(MATVEC_WG / slices), 1);
        linear_dispatch(rec, kernel, x, w, b, dims, groups, Some(epilogue), out)
    } else {
        let groups = (n_out.div_ceil(MATMUL_TILE), t.div_ceil(MATMUL_TILE));
        let kernel = &gpu.kernels.matmul[ep];
        linear_dispatch(rec, kernel, x, w, b, dims, groups, Some(epilogue), out)
    }
}

/// The M2 kernel (D19), kept as M6's baseline and as a second differential check (D49):
/// `y = x @ w^T + b` with `w: [out, in]`, the CPU's layout (D7). A plain serial sum, so its bits
/// differ from `linear`'s chunked sum (D53).
pub fn linear_naive(
    gpu: &Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    b: Option<&GpuTensor>,
) -> Result<GpuTensor> {
    let dims = shape::linear(x.shape(), w.shape(), b.map(GpuTensor::shape))?;
    let (t, _, n_out) = dims;
    let groups = (n_out.div_ceil(NAIVE_TILE), t.div_ceil(NAIVE_TILE));
    once(gpu, &[t, n_out], |rec, out| {
        linear_dispatch(
            rec,
            &gpu.kernels.linear_naive,
            x,
            w,
            b,
            dims,
            groups,
            None,
            out,
        )
    })
}

/// Shared by the linear kernels: they take the same bindings and `(t, in, out, has_bias)`
/// parameters, `dims = (t, in, out)` already checked against the kernel's weight layout.
/// `epilogue` is `None` for `linear_naive`, which has none (and no `res` binding).
#[allow(clippy::too_many_arguments)]
fn linear_dispatch(
    rec: &mut Rec,
    kernel: &Kernel,
    x: &GpuTensor,
    w: &GpuTensor,
    b: Option<&GpuTensor>,
    (t, n_in, n_out): (usize, usize, usize),
    (gx, gy): (usize, usize),
    epilogue: Option<Epilogue>,
    out: &GpuTensor,
) -> Result<()> {
    check_out("linear", out, &[t, n_out])?;
    if let Some(Epilogue::Residual(r)) = epilogue {
        if r.shape() != out.shape() {
            return Err(Error::Shape(format!(
                "linear: residual {:?} for output {:?}",
                r.shape(),
                out.shape()
            )));
        }
        // Read-only and read-write in one dispatch is a validation error, not an Err.
        if r.buffer == out.buffer {
            return Err(Error::Shape(
                "linear: residual and output share a buffer".into(),
            ));
        }
    }
    if t == 0 || n_out == 0 {
        return Ok(());
    }
    len_u32(t * n_in.max(n_out))?;
    len_u32(n_out * n_in)?;
    let max = rec.gpu.max_groups() as usize;
    if gx > max || gy > max {
        return Err(Error::Shape(format!(
            "linear: [{t}, {n_out}] output needs more than {max} workgroups per dimension"
        )));
    }
    let params = Params4::new(t as u32, n_in as u32, n_out as u32, b.is_some() as u32);
    // Without a bias the binding still needs some buffer. `w` serves: it is never read through
    // that binding, and binding one buffer twice read-only is allowed (`out`, read-write,
    // would be a validation error). A buffer allocated for it would be a creation per call (D60).
    let bias = b.map_or(&w.buffer, |b| &b.buffer);
    let groups = (gx as u32, gy as u32, 1);
    let params = bytemuck::bytes_of(&params);
    match epilogue {
        None => rec.dispatch(
            kernel,
            &[&x.buffer, &w.buffer, bias, &out.buffer],
            params,
            groups,
        ),
        // The residual binding without a residual gets `x`, as the bias binding gets `w`.
        Some(ep) => {
            let res = match ep {
                Epilogue::Residual(r) => &r.buffer,
                _ => &x.buffer,
            };
            let buffers = [&x.buffer, &w.buffer, bias, &out.buffer, res];
            rec.dispatch(kernel, &buffers, params, groups)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionParams {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,
    start: u32,
    n_chunks: u32,
    kv: u32,
    group: u32,
}

/// Query heads and key/value heads (D71). Each key/value head serves `q / kv` query heads;
/// GPT-2 has as many of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heads {
    pub q: usize,
    pub kv: usize,
}

impl Heads {
    /// Plain multi-head attention: every head has its own keys and values.
    pub fn mha(n: usize) -> Self {
        Heads { q: n, kv: n }
    }
}

/// Shape of attention's per-chunk partials for `T` queries from `start` (D63):
/// `[T, heads, n_chunks, d + 2]`, each chunk's (max, sum of exponentials, unnormalized output).
/// `n_chunks` covers the last query's keys; earlier rows leave their trailing chunks unwritten.
pub fn attention_parts_shape(t: usize, start: usize, e: usize, n_head: usize) -> [usize; 4] {
    let d = e.checked_div(n_head).unwrap_or(0);
    [t, n_head, (start + t).div_ceil(ATTENTION_CHUNK), d + 2]
}

/// `[T, (q + 2 kv) d]` -> `(T, d)`, checking that the row splits into the heads.
fn qkv_dims(op: &str, qkv: &GpuTensor, heads: Heads) -> Result<(usize, usize)> {
    let (t, d) = shape::qkv_gqa(op, qkv.shape(), heads.q, heads.kv)?;
    len_u32(numel(qkv.shape()))?;
    Ok((t, d))
}

/// `[n_ctx, width]` cache buffers for `start + T` positions.
fn check_cache(op: &str, k: &GpuTensor, v: &GpuTensor, e: usize, end: usize) -> Result<()> {
    let &[rows, ke] = k.shape() else {
        return Err(Error::Shape(format!(
            "{op}: cache {:?} must be 2-D",
            k.shape()
        )));
    };
    if v.shape() != k.shape() || ke != e {
        return Err(Error::Shape(format!(
            "{op}: caches {:?} and {:?} for width {e}",
            k.shape(),
            v.shape()
        )));
    }
    if end > rows {
        return Err(Error::Input(format!(
            "{op}: positions up to {end} exceed the cache's {rows}"
        )));
    }
    len_u32(rows * e)?;
    Ok(())
}

/// Write the K and V thirds of `qkv: [T, 3E]` into cache rows `start..start + T` (D32), for
/// as many key/value heads as query heads (the GPT-2 layout).
pub fn kv_write(
    gpu: &Gpu,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
) -> Result<()> {
    let mut rec = gpu.rec();
    kv_write_into(&mut rec, qkv, k_cache, v_cache, start, Heads::mha(1))?;
    rec.submit();
    Ok(())
}

/// `kv_write` recorded into `rec` (D61), for `qkv: [T, (q + 2 kv) d]`: the k and v parts go to
/// caches `[n_ctx, kv d]` (D71).
pub fn kv_write_into(
    rec: &mut Rec,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
    heads: Heads,
) -> Result<()> {
    let (t, d) = qkv_dims("kv_write", qkv, heads)?;
    let (q, kv) = (heads.q * d, heads.kv * d);
    check_cache("kv_write", k_cache, v_cache, kv, start + t)?;
    if t == 0 || kv == 0 {
        return Ok(());
    }
    let kernel = &rec.gpu.kernels.kv_write;
    elementwise(
        rec,
        kernel,
        &[&qkv.buffer, &k_cache.buffer, &v_cache.buffer],
        Params4::new((t * kv) as u32, kv as u32, start as u32, q as u32),
        rec.gpu.max_groups(),
    )
}

/// Attention for the queries in `qkv: [T, 3E]` at absolute positions `start..start + T`, over
/// keys and values already in the cache (D31). Returns `[T, E]`.
pub fn attention_cached(
    gpu: &Gpu,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
    n_head: usize,
) -> Result<GpuTensor> {
    attention_cached_gqa(gpu, qkv, k_cache, v_cache, start, Heads::mha(n_head))
}

/// `attention_cached` with grouped key/value heads (D71): `qkv: [T, (q + 2 kv) d]`, caches
/// `[n_ctx, kv d]`. Returns `[T, q d]`.
pub fn attention_cached_gqa(
    gpu: &Gpu,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
    heads: Heads,
) -> Result<GpuTensor> {
    let (t, d) = qkv_dims("attention", qkv, heads)?;
    let e = heads.q * d;
    let parts = gpu.alloc(&attention_parts_shape(t, start, e, heads.q));
    once(gpu, &[t, e], |rec, out| {
        attention_into(rec, qkv, k_cache, v_cache, start, heads, &parts, out)
    })
}

/// `attention_cached` into `out` (D61), with `parts` as the scratch for the chunk partials
/// (shape from `attention_parts_shape`; a larger buffer is fine). Two dispatches (D63): every
/// (row, head, chunk of 64 keys) into `parts`, then every (row, head) merges its chunks.
#[allow(clippy::too_many_arguments)]
pub fn attention_into(
    rec: &mut Rec,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
    heads: Heads,
    parts: &GpuTensor,
    out: &GpuTensor,
) -> Result<()> {
    let n_head = heads.q;
    let (t, d) = qkv_dims("attention", qkv, heads)?;
    let (e, kv) = (heads.q * d, heads.kv * d);
    check_cache("attention", k_cache, v_cache, kv, start + t)?;
    check_out("attention", out, &[t, e])?;
    let shape = attention_parts_shape(t, start, e, n_head);
    check_out("attention parts", parts, &shape)?;
    if d > ATTENTION_MAX_D {
        return Err(Error::Shape(format!(
            "attention: head dimension {d} exceeds the kernel's {ATTENTION_MAX_D}"
        )));
    }
    let max = rec.gpu.max_groups() as usize;
    if t > max || n_head > max || shape[2] > max {
        return Err(Error::Shape(format!(
            "attention: {t} rows, {n_head} heads or {} chunks exceed one dispatch dimension",
            shape[2]
        )));
    }
    len_u32(numel(&shape))?;
    if t == 0 || e == 0 {
        return Ok(());
    }
    let params = AttentionParams {
        t: t as u32,
        e: e as u32,
        d: d as u32,
        scale: 1.0 / (d as f32).sqrt(),
        start: start as u32,
        n_chunks: shape[2] as u32,
        kv: kv as u32,
        group: (heads.q / heads.kv) as u32,
    };
    let params = bytemuck::bytes_of(&params);
    rec.dispatch(
        &rec.gpu.kernels.attention,
        &[&qkv.buffer, &k_cache.buffer, &v_cache.buffer, &parts.buffer],
        params,
        (t as u32, n_head as u32, shape[2] as u32),
    )?;
    rec.dispatch(
        &rec.gpu.kernels.attention_combine,
        &[&parts.buffer, &out.buffer],
        params,
        (t as u32, n_head as u32, 1),
    )
}

/// Causal multi-head attention from the fused `qkv: [T, 3E]` alone; returns `[T, E]`. Writes a
/// temporary `[T, E]` cache and attends over it: the M2 op, now a special case of D31.
pub fn causal_attention(gpu: &Gpu, qkv: &GpuTensor, n_head: usize) -> Result<GpuTensor> {
    causal_attention_gqa(gpu, qkv, Heads::mha(n_head))
}

/// `causal_attention` with grouped key/value heads (D71): `qkv: [T, (q + 2 kv) d]`, returns
/// `[T, q d]`.
pub fn causal_attention_gqa(gpu: &Gpu, qkv: &GpuTensor, heads: Heads) -> Result<GpuTensor> {
    let (t, d) = qkv_dims("attention", qkv, heads)?;
    let (k, v) = (gpu.alloc(&[t, heads.kv * d]), gpu.alloc(&[t, heads.kv * d]));
    let mut rec = gpu.rec();
    kv_write_into(&mut rec, qkv, &k, &v, 0, heads)?;
    rec.submit();
    attention_cached_gqa(gpu, qkv, &k, &v, 0, heads)
}

/// A copy of `x` made by a kernel, 16 bytes per load and store: the bandwidth probe (D41). The
/// length must be a multiple of 4 (whole vec4s). For moving data, `Gpu::copy` / `row` are the
/// tools; this exists to be timed.
pub fn copy(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    copy_with_max_groups(gpu, x, gpu.max_groups())
}

/// `copy` with a cap on the workgroup count, so tests can exercise the grid-stride loop (as
/// `add_with_max_groups`).
pub fn copy_with_max_groups(gpu: &Gpu, x: &GpuTensor, max_groups: u32) -> Result<GpuTensor> {
    if !x.len().is_multiple_of(4) {
        return Err(Error::Shape(format!(
            "copy: {} elements aren't whole vec4s",
            x.len()
        )));
    }
    once(gpu, x.shape(), |rec, out| {
        if x.is_empty() {
            return Ok(());
        }
        elementwise(
            rec,
            &gpu.kernels.copy,
            &[&x.buffer, &out.buffer],
            Params4::new(len_u32(x.len() / 4)?, 0, 0, 0),
            max_groups,
        )
    })
}

/// Fused multiply-adds per invocation and step of `fma_peak`: its 8 vec4 chains.
pub const FMA_PEAK_CHAINS: usize = 32;

/// The compute-roof probe (D47): `groups` workgroups of 256 invocations each run
/// `FMA_PEAK_CHAINS` independent fma chains for `iters` steps, which is
/// `2 * FMA_PEAK_CHAINS * iters * 256 * groups` flops. Returns one value per invocation, so the
/// work can't be optimised away. Like `copy`, it exists to be timed.
pub fn fma_peak(gpu: &Gpu, groups: u32, iters: u32) -> Result<GpuTensor> {
    let n = groups as usize * ELEMENTWISE_WG as usize;
    if groups == 0 || groups > gpu.max_groups() {
        return Err(Error::Shape(format!("fma_peak: {groups} workgroups")));
    }
    once(gpu, &[n], |rec, out| {
        rec.dispatch(
            &gpu.kernels.fma_peak,
            &[&out.buffer],
            bytemuck::bytes_of(&Params4::new(iters, 0, 0, 0)),
            (groups, 1, 1),
        )
    })
}

/// Workgroups `read_peak` launches: enough to fill the GPU, few enough that its one store per
/// invocation is small next to what it reads.
const READ_PEAK_GROUPS: usize = 1024;

/// The read-bandwidth probe (D47): per-invocation sums over `x`, read once, 16 bytes per load.
/// The length must be a multiple of 4. Like `copy`, it exists to be timed.
pub fn read_peak(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    if x.is_empty() || !x.len().is_multiple_of(4) {
        return Err(Error::Shape(format!(
            "read_peak: {} elements aren't one or more whole vec4s",
            x.len()
        )));
    }
    let groups = READ_PEAK_GROUPS
        .min(x.len().div_ceil(4 * ELEMENTWISE_WG as usize))
        .max(1);
    let params = Params4::new(len_u32(x.len() / 4)?, 0, 0, 0);
    once(gpu, &[groups * ELEMENTWISE_WG as usize * 4], |rec, out| {
        rec.dispatch(
            &gpu.kernels.read_peak,
            &[&x.buffer, &out.buffer],
            bytemuck::bytes_of(&params),
            (groups as u32, 1, 1),
        )
    })
}

/// Row `i` of `x: [R, C]` as a new `[1, C]` tensor: a buffer-to-buffer copy, no kernel (D26).
pub fn row(gpu: &Gpu, x: &GpuTensor, i: usize) -> Result<GpuTensor> {
    let &[_, c] = x.shape() else {
        return Err(Error::Shape(format!("row: x {:?} must be 2-D", x.shape())));
    };
    once(gpu, &[1, c], |rec, out| row_into(rec, x, i, out))
}

/// `row` into `out: [1, C]` (D61).
pub fn row_into(rec: &mut Rec, x: &GpuTensor, i: usize, out: &GpuTensor) -> Result<()> {
    let &[r, c] = x.shape() else {
        return Err(Error::Shape(format!("row: x {:?} must be 2-D", x.shape())));
    };
    if i >= r {
        return Err(Error::Shape(format!("row {i} of a {r}-row tensor")));
    }
    check_out("row", out, &[1, c])?;
    if c > 0 {
        rec.copy(&x.buffer, (i * c * 4) as u64, &out.buffer, (c * 4) as u64);
    }
    Ok(())
}

/// Kernels index with u32 (WGSL has no 64-bit integers by default).
fn len_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Shape(format!("{n} elements exceed u32 indexing")))
}

/// Workgroups for `n` elements with the grid-stride loop (D8): enough for one element per
/// invocation, capped at the per-dimension limit.
pub fn elementwise_groups(n: usize, max_groups: u32) -> u32 {
    let wanted = n.div_ceil(ELEMENTWISE_WG as usize);
    wanted.min(max_groups as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_cover_n_and_respect_cap() {
        assert_eq!(elementwise_groups(1, 65535), 1);
        assert_eq!(elementwise_groups(256, 65535), 1);
        assert_eq!(elementwise_groups(257, 65535), 2);
        assert_eq!(elementwise_groups(38_597_376, 65535), 65535);
        assert_eq!(elementwise_groups(10_000, 3), 3);
    }
}
