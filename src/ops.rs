//! GPU ops: host-side wrappers that check shapes, allocate the output and dispatch a kernel.
//! Each one has a CPU twin in `cpu.rs` that it is tested against.

use crate::error::{Error, Result};
use crate::gpu::{Gpu, Kernel};
use crate::shape;
use crate::tensor::GpuTensor;

/// Longest sequence the attention kernel takes: its scores live in a shared-memory array of
/// this size. Must match `MAX_CTX` in attention.wgsl.
pub const ATTENTION_MAX_CTX: usize = 1024;

/// Threads per workgroup for 1-D elementwise kernels. Must match `WG` in the shaders.
pub const ELEMENTWISE_WG: u32 = 256;

/// Elementwise `a + b` on the GPU.
pub fn add(gpu: &Gpu, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    add_with_max_groups(gpu, a, b, gpu.limits.max_compute_workgroups_per_dimension)
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
    let out = gpu.alloc(a.shape());
    let n = a.len();
    if n == 0 {
        return Ok(out);
    }
    let params = Params4::new(len_u32(n)?, 0, 0, 0);
    elementwise(
        gpu,
        &gpu.kernels.add,
        &[&a.buffer, &b.buffer, &out.buffer],
        params,
        max_groups,
    );
    Ok(out)
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

/// Dispatch a grid-stride elementwise kernel (D8): `buffers` in binding order, then `params` as
/// the last binding, whose first field is the element count.
fn elementwise(
    gpu: &Gpu,
    kernel: &Kernel,
    buffers: &[&wgpu::Buffer],
    params: Params4,
    max_groups: u32,
) {
    let uniform = gpu.uniform(&params);
    let mut bindings = buffers.to_vec();
    bindings.push(&uniform);
    let groups = elementwise_groups(params.a as usize, max_groups);
    gpu.dispatch(kernel, &bindings, (groups, 1, 1));
}

/// Elementwise GPT-2 GELU (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    let out = gpu.alloc(x.shape());
    if x.is_empty() {
        return Ok(out);
    }
    elementwise(
        gpu,
        &gpu.kernels.gelu,
        &[&x.buffer, &out.buffer],
        Params4::new(len_u32(x.len())?, 0, 0, 0),
        gpu.limits.max_compute_workgroups_per_dimension,
    );
    Ok(out)
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
    let &[e_rows, v] = wte_t.shape() else {
        return Err(Error::Shape(format!(
            "embed: wte_t {:?} must be 2-D",
            wte_t.shape()
        )));
    };
    let e = shape::embed(&[v, e_rows], wpe.shape(), ids, start)?;
    let out = gpu.alloc(&[ids.len(), e]);
    if ids.is_empty() {
        return Ok(out);
    }
    let n = len_u32(ids.len() * e)?;
    let ids_buf = gpu.upload_u32(ids);
    elementwise(
        gpu,
        &gpu.kernels.embed,
        &[&wte_t.buffer, &wpe.buffer, &ids_buf, &out.buffer],
        Params4::new(n, e as u32, start as u32, len_u32(v)?),
        gpu.limits.max_compute_workgroups_per_dimension,
    );
    Ok(out)
}

/// `[rows, cols]` for the row kernels, which launch one workgroup per row.
fn rows_cols(gpu: &Gpu, op: &str, x: &GpuTensor) -> Result<(u32, u32)> {
    let &[rows, cols] = x.shape() else {
        return Err(Error::Shape(format!("{op}: x {:?} must be 2-D", x.shape())));
    };
    if rows > gpu.limits.max_compute_workgroups_per_dimension as usize {
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
    let out = gpu.alloc(x.shape());
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    let params = gpu.uniform(&Params4::new(cols, 0, 0, 0));
    gpu.dispatch(
        &gpu.kernels.softmax,
        &[&x.buffer, &out.buffer, &params],
        (rows, 1, 1),
    );
    Ok(out)
}

/// LayerNorm over the last dimension of `x: [rows, cols]` with `gain`, `bias`: `[cols]`.
pub fn layer_norm(
    gpu: &Gpu,
    x: &GpuTensor,
    gain: &GpuTensor,
    bias: &GpuTensor,
    eps: f32,
) -> Result<GpuTensor> {
    shape::layer_norm(x.shape(), gain.shape(), bias.shape())?;
    let (rows, cols) = rows_cols(gpu, "layer_norm", x)?;
    let out = gpu.alloc(x.shape());
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    let params = gpu.uniform(&Params4::new(cols, eps.to_bits(), 0, 0));
    gpu.dispatch(
        &gpu.kernels.layer_norm,
        &[&x.buffer, &gain.buffer, &bias.buffer, &out.buffer, &params],
        (rows, 1, 1),
    );
    Ok(out)
}

/// Threads per side of `linear_naive`'s 2-D workgroup. Must match `@workgroup_size(16, 16)`.
const NAIVE_TILE: usize = 16;

/// Rows and columns of the output per `matmul` workgroup. Must match `BM` and `BN` in matmul.wgsl.
const MATMUL_TILE: usize = 64;

/// Outputs per `matvec` workgroup. Must match `WG` in matvec.wgsl.
const MATVEC_WG: usize = 256;

/// `y = x @ w + b`: `x: [T, in]`, `w: [in, out]` (the GPU layout, D45), `b: [out]` or none.
/// One row (a decode step) goes to the matrix-vector kernel, more rows to the tiled matmul
/// (D43, D44). Both compute each output with the same serial sequence of fused multiply-adds,
/// so a row's bits don't depend on which kernel ran it or what else was in the batch (D46).
pub fn linear(gpu: &Gpu, x: &GpuTensor, w: &GpuTensor, b: Option<&GpuTensor>) -> Result<GpuTensor> {
    let (t, n_in, n_out) = shape::linear_in_out(x.shape(), w.shape(), b.map(GpuTensor::shape))?;
    let dims = (t, n_in, n_out);
    if t == 1 {
        let groups = (n_out.div_ceil(MATVEC_WG), 1);
        linear_dispatch(gpu, &gpu.kernels.matvec, x, w, b, dims, groups)
    } else {
        let groups = (n_out.div_ceil(MATMUL_TILE), t.div_ceil(MATMUL_TILE));
        linear_dispatch(gpu, &gpu.kernels.matmul, x, w, b, dims, groups)
    }
}

/// The M2 kernel (D19), kept as M6's baseline and as a second differential check (D49):
/// `y = x @ w^T + b` with `w: [out, in]`, the CPU's layout (D7).
pub fn linear_naive(
    gpu: &Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    b: Option<&GpuTensor>,
) -> Result<GpuTensor> {
    let dims = shape::linear(x.shape(), w.shape(), b.map(GpuTensor::shape))?;
    let (t, _, n_out) = dims;
    let groups = (n_out.div_ceil(NAIVE_TILE), t.div_ceil(NAIVE_TILE));
    linear_dispatch(gpu, &gpu.kernels.linear_naive, x, w, b, dims, groups)
}

/// Shared by the three linear kernels: they take the same bindings and `(t, in, out, has_bias)`
/// parameters, `dims = (t, in, out)` already checked against the kernel's weight layout.
fn linear_dispatch(
    gpu: &Gpu,
    kernel: &Kernel,
    x: &GpuTensor,
    w: &GpuTensor,
    b: Option<&GpuTensor>,
    (t, n_in, n_out): (usize, usize, usize),
    (gx, gy): (usize, usize),
) -> Result<GpuTensor> {
    let out = gpu.alloc(&[t, n_out]);
    if t == 0 || n_out == 0 {
        return Ok(out);
    }
    len_u32(t * n_in.max(n_out))?;
    len_u32(n_out * n_in)?;
    let max = gpu.limits.max_compute_workgroups_per_dimension as usize;
    if gx > max || gy > max {
        return Err(Error::Shape(format!(
            "linear: [{t}, {n_out}] output needs more than {max} workgroups per dimension"
        )));
    }
    let params = gpu.uniform(&Params4::new(
        t as u32,
        n_in as u32,
        n_out as u32,
        b.is_some() as u32,
    ));
    // Without a bias the binding still needs some buffer, and it can't be `out`: one buffer
    // bound read-only and read-write in the same dispatch is a validation error.
    let dummy;
    let bias = match b {
        Some(b) => &b.buffer,
        None => {
            dummy = gpu.alloc(&[1]);
            &dummy.buffer
        }
    };
    gpu.dispatch(
        kernel,
        &[&x.buffer, &w.buffer, bias, &out.buffer, &params],
        (gx as u32, gy as u32, 1),
    );
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionParams {
    t: u32,
    e: u32,
    d: u32,
    scale: f32,
    start: u32,
    _pad: [u32; 3],
}

/// `[T, 3E]` -> `(T, E)`, checking that 3E splits into 3 x `n_head` heads.
fn qkv_dims(op: &str, qkv: &GpuTensor, n_head: usize) -> Result<(usize, usize)> {
    let (t, e) = shape::qkv(op, qkv.shape(), n_head)?;
    len_u32(t * 3 * e)?;
    Ok((t, e))
}

/// `[n_ctx, E]` cache buffers for `start + T` positions of width `e`.
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

/// Write the K and V thirds of `qkv: [T, 3E]` into cache rows `start..start + T` (D32).
pub fn kv_write(
    gpu: &Gpu,
    qkv: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    start: usize,
) -> Result<()> {
    let (t, e) = qkv_dims("kv_write", qkv, 1)?;
    check_cache("kv_write", k_cache, v_cache, e, start + t)?;
    if t == 0 || e == 0 {
        return Ok(());
    }
    elementwise(
        gpu,
        &gpu.kernels.kv_write,
        &[&qkv.buffer, &k_cache.buffer, &v_cache.buffer],
        Params4::new((t * e) as u32, e as u32, start as u32, 0),
        gpu.limits.max_compute_workgroups_per_dimension,
    );
    Ok(())
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
    let (t, e) = qkv_dims("attention", qkv, n_head)?;
    check_cache("attention", k_cache, v_cache, e, start + t)?;
    if start + t > ATTENTION_MAX_CTX {
        return Err(Error::Input(format!(
            "attention: {} positions exceed the kernel's {ATTENTION_MAX_CTX}",
            start + t
        )));
    }
    if n_head > gpu.limits.max_compute_workgroups_per_dimension as usize {
        return Err(Error::Shape(format!(
            "attention: {n_head} heads exceed one dispatch dimension"
        )));
    }
    let out = gpu.alloc(&[t, e]);
    if t == 0 || e == 0 {
        return Ok(out);
    }
    let d = e / n_head;
    let params = gpu.uniform(&AttentionParams {
        t: t as u32,
        e: e as u32,
        d: d as u32,
        scale: 1.0 / (d as f32).sqrt(),
        start: start as u32,
        _pad: [0; 3],
    });
    gpu.dispatch(
        &gpu.kernels.attention,
        &[
            &qkv.buffer,
            &k_cache.buffer,
            &v_cache.buffer,
            &out.buffer,
            &params,
        ],
        (t as u32, n_head as u32, 1),
    );
    Ok(out)
}

/// Causal multi-head attention from the fused `qkv: [T, 3E]` alone; returns `[T, E]`. Writes a
/// temporary `[T, E]` cache and attends over it: the M2 op, now a special case of D31.
pub fn causal_attention(gpu: &Gpu, qkv: &GpuTensor, n_head: usize) -> Result<GpuTensor> {
    let (t, e) = qkv_dims("attention", qkv, n_head)?;
    let (k, v) = (gpu.alloc(&[t, e]), gpu.alloc(&[t, e]));
    kv_write(gpu, qkv, &k, &v, 0)?;
    attention_cached(gpu, qkv, &k, &v, 0, n_head)
}

/// A copy of `x` made by a kernel, 16 bytes per load and store: the bandwidth probe (D41). The
/// length must be a multiple of 4 (whole vec4s). For moving data, `Gpu::copy` / `row` are the
/// tools; this exists to be timed.
pub fn copy(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    copy_with_max_groups(gpu, x, gpu.limits.max_compute_workgroups_per_dimension)
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
    let out = gpu.alloc(x.shape());
    if x.is_empty() {
        return Ok(out);
    }
    elementwise(
        gpu,
        &gpu.kernels.copy,
        &[&x.buffer, &out.buffer],
        Params4::new(len_u32(x.len() / 4)?, 0, 0, 0),
        max_groups,
    );
    Ok(out)
}

/// Fused multiply-adds per invocation and step of `fma_peak`. Must match `CHAINS` in the shader.
pub const FMA_PEAK_CHAINS: usize = 8;

/// The compute-roof probe (D47): `groups` workgroups of 256 invocations each run
/// `FMA_PEAK_CHAINS` independent fma chains for `iters` steps, which is
/// `2 * FMA_PEAK_CHAINS * iters * 256 * groups` flops. Returns one value per invocation, so the
/// work can't be optimised away. Like `copy`, it exists to be timed.
pub fn fma_peak(gpu: &Gpu, groups: u32, iters: u32) -> Result<GpuTensor> {
    let n = groups as usize * ELEMENTWISE_WG as usize;
    if groups == 0 || groups > gpu.limits.max_compute_workgroups_per_dimension {
        return Err(Error::Shape(format!("fma_peak: {groups} workgroups")));
    }
    let out = gpu.alloc(&[n]);
    let params = gpu.uniform(&Params4::new(iters, 0, 0, 0));
    gpu.dispatch(
        &gpu.kernels.fma_peak,
        &[&out.buffer, &params],
        (groups, 1, 1),
    );
    Ok(out)
}

/// Row `i` of `x: [R, C]` as a new `[1, C]` tensor: a buffer-to-buffer copy, no kernel (D26).
pub fn row(gpu: &Gpu, x: &GpuTensor, i: usize) -> Result<GpuTensor> {
    let &[r, c] = x.shape() else {
        return Err(Error::Shape(format!("row: x {:?} must be 2-D", x.shape())));
    };
    if i >= r {
        return Err(Error::Shape(format!("row {i} of a {r}-row tensor")));
    }
    let out = gpu.alloc(&[1, c]);
    if c > 0 {
        gpu.copy(&x.buffer, (i * c * 4) as u64, &out.buffer, (c * 4) as u64);
    }
    Ok(out)
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
