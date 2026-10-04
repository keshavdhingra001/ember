//! GPU ops: host-side wrappers that check shapes, allocate the output and dispatch a kernel.
//! Each one has a CPU twin in `cpu.rs` that it is tested against.

use crate::cpu::same_shape_dims;
use crate::error::{Error, Result};
use crate::gpu::Gpu;
use crate::tensor::GpuTensor;

/// Longest sequence the attention kernel takes: its scores live in a shared-memory array of
/// this size. Must match `MAX_CTX` in attention.wgsl.
pub const ATTENTION_MAX_CTX: usize = 1024;

/// Threads per workgroup for 1-D elementwise kernels. Must match `WG` in the shaders.
pub const ELEMENTWISE_WG: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LenParams {
    n: u32,
    _pad: [u32; 3],
}

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
    same_shape_dims("add", a.shape(), b.shape())?;
    let out = gpu.alloc(a.shape());
    let n = a.len();
    if n == 0 {
        return Ok(out);
    }
    let n = len_u32(n)?;
    let params = gpu.uniform(&LenParams { n, _pad: [0; 3] });
    gpu.dispatch(
        &gpu.kernels.add,
        &[&a.buffer, &b.buffer, &out.buffer, &params],
        (elementwise_groups(n as usize, max_groups), 1, 1),
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

/// Elementwise GPT-2 GELU (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuTensor) -> Result<GpuTensor> {
    let out = gpu.alloc(x.shape());
    if x.is_empty() {
        return Ok(out);
    }
    let n = len_u32(x.len())?;
    let params = gpu.uniform(&Params4::new(n, 0, 0, 0));
    let groups = elementwise_groups(x.len(), gpu.limits.max_compute_workgroups_per_dimension);
    gpu.dispatch(
        &gpu.kernels.gelu,
        &[&x.buffer, &out.buffer, &params],
        (groups, 1, 1),
    );
    Ok(out)
}

/// `out[t] = wte[ids[t]] + wpe[t]`. Ids and length are checked here: the kernel would
/// silently read the wrong row (or a clamped one, D9).
pub fn embed(gpu: &Gpu, wte: &GpuTensor, wpe: &GpuTensor, ids: &[u32]) -> Result<GpuTensor> {
    let (&[v, e], &[n_ctx, e2]) = (wte.shape(), wpe.shape()) else {
        return Err(Error::Shape(format!(
            "embed: wte {:?} and wpe {:?} must be 2-D",
            wte.shape(),
            wpe.shape()
        )));
    };
    if e != e2 {
        return Err(Error::Shape(format!(
            "embed: wte {:?} vs wpe {:?}",
            wte.shape(),
            wpe.shape()
        )));
    }
    if ids.len() > n_ctx {
        return Err(Error::Input(format!(
            "{} tokens exceed the context length {n_ctx}",
            ids.len()
        )));
    }
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= v) {
        return Err(Error::Input(format!(
            "token id {bad} is outside the vocab (size {v})"
        )));
    }
    let out = gpu.alloc(&[ids.len(), e]);
    if ids.is_empty() {
        return Ok(out);
    }
    let n = len_u32(ids.len() * e)?;
    let ids_buf = gpu.upload_u32(ids);
    let params = gpu.uniform(&Params4::new(n, e as u32, 0, 0));
    let groups = elementwise_groups(n as usize, gpu.limits.max_compute_workgroups_per_dimension);
    gpu.dispatch(
        &gpu.kernels.embed,
        &[&wte.buffer, &wpe.buffer, &ids_buf, &out.buffer, &params],
        (groups, 1, 1),
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
    let (rows, cols) = rows_cols(gpu, "layer_norm", x)?;
    if gain.shape() != [cols as usize] || bias.shape() != [cols as usize] {
        return Err(Error::Shape(format!(
            "layer_norm: x {:?}, gain {:?}, bias {:?}",
            x.shape(),
            gain.shape(),
            bias.shape()
        )));
    }
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

/// Threads per side of the 2-D `linear` workgroup. Must match `@workgroup_size(16, 16)`.
const LINEAR_TILE: usize = 16;

/// `y = x @ w^T + b`: `x: [T, in]`, `w: [out, in]` (D7), `b: [out]` or none.
pub fn linear(gpu: &Gpu, x: &GpuTensor, w: &GpuTensor, b: Option<&GpuTensor>) -> Result<GpuTensor> {
    let (&[t, n_in], &[n_out, w_in]) = (x.shape(), w.shape()) else {
        return Err(Error::Shape(format!(
            "linear: x {:?} and w {:?} must be 2-D",
            x.shape(),
            w.shape()
        )));
    };
    if n_in != w_in || b.is_some_and(|b| b.shape() != [n_out]) {
        return Err(Error::Shape(format!(
            "linear: x {:?}, w {:?}, b {:?}",
            x.shape(),
            w.shape(),
            b.map(GpuTensor::shape)
        )));
    }
    let out = gpu.alloc(&[t, n_out]);
    if t == 0 || n_out == 0 {
        return Ok(out);
    }
    len_u32(t * n_in.max(n_out))?;
    len_u32(n_out * n_in)?;
    let (gx, gy) = (n_out.div_ceil(LINEAR_TILE), t.div_ceil(LINEAR_TILE));
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
        &gpu.kernels.linear,
        &[&x.buffer, &w.buffer, bias, &out.buffer, &params],
        (gx as u32, gy as u32, 1),
    );
    Ok(out)
}

/// Causal multi-head attention from the fused `qkv: [T, 3E]`; returns `[T, E]`.
pub fn causal_attention(gpu: &Gpu, qkv: &GpuTensor, n_head: usize) -> Result<GpuTensor> {
    let &[t, three_e] = qkv.shape() else {
        return Err(Error::Shape(format!(
            "attention: qkv {:?} must be 2-D",
            qkv.shape()
        )));
    };
    if n_head == 0 || three_e % (3 * n_head) != 0 {
        return Err(Error::Shape(format!(
            "attention: qkv {:?} doesn't split into 3 x {n_head} heads",
            qkv.shape()
        )));
    }
    if t > ATTENTION_MAX_CTX {
        return Err(Error::Input(format!(
            "attention: {t} positions exceed the kernel's {ATTENTION_MAX_CTX}"
        )));
    }
    let e = three_e / 3;
    let d = e / n_head;
    let out = gpu.alloc(&[t, e]);
    if t == 0 {
        return Ok(out);
    }
    len_u32(t * three_e)?;
    let scale = 1.0 / (d as f32).sqrt();
    let params = gpu.uniform(&Params4::new(t as u32, e as u32, d as u32, scale.to_bits()));
    gpu.dispatch(
        &gpu.kernels.attention,
        &[&qkv.buffer, &out.buffer, &params],
        (t as u32, n_head as u32, 1),
    );
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
