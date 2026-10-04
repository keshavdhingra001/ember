//! GPU ops: host-side wrappers that check shapes, allocate the output and dispatch a kernel.
//! Each one has a CPU twin in `cpu.rs` that it is tested against.

use crate::cpu::same_shape_dims;
use crate::error::{Error, Result};
use crate::gpu::Gpu;
use crate::tensor::GpuTensor;

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
        elementwise_groups(n as usize, max_groups),
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
