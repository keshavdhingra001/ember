//! Host tensors and their GPU counterparts (D7): contiguous row-major f32, shape on the host.

use crate::error::{Error, Result};

/// A tensor in CPU memory. The CPU reference ops (`cpu.rs`) work on these, and they are what
/// gets uploaded to and read back from the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Fails if `data.len()` isn't the product of `shape`. A scalar has shape `[]` and 1 element.
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        let want = numel(shape);
        if data.len() != want {
            return Err(Error::Shape(format!(
                "shape {shape:?} needs {want} elements, got {}",
                data.len()
            )));
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            data,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            data: vec![0.0; numel(shape)],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A tensor living in a GPU storage buffer. Created by `Gpu::upload` or by a kernel;
/// read back with `Gpu::read`.
pub struct GpuTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) buffer: wgpu::Buffer,
}

impl GpuTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        numel(&self.shape)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Number of elements in a tensor of this shape (1 for the scalar shape `[]`).
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_checks_length() {
        assert!(Tensor::new(&[2, 3], vec![0.0; 6]).is_ok());
        assert!(Tensor::new(&[2, 3], vec![0.0; 5]).is_err());
        assert_eq!(Tensor::new(&[], vec![1.0]).unwrap().len(), 1);
        assert!(Tensor::new(&[0, 4], vec![]).unwrap().is_empty());
    }
}
