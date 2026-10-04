//! The CPU reference ops: plain, obviously correct Rust. This is the oracle every GPU kernel
//! is differential-tested against (D3). Clarity beats speed here, always.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Elementwise `a + b`. Shapes must match exactly (no broadcasting yet).
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    same_shape_dims("add", a.shape(), b.shape())?;
    let data = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    Tensor::new(a.shape(), data)
}

pub(crate) fn same_shape_dims(op: &str, a: &[usize], b: &[usize]) -> Result<()> {
    if a != b {
        return Err(Error::Shape(format!("{op}: {a:?} vs {b:?}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_small() {
        let a = Tensor::new(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::new(&[3], vec![10.0, 20.0, 30.0]).unwrap();
        assert_eq!(add(&a, &b).unwrap().data(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[3, 2]);
        assert!(add(&a, &b).is_err());
    }
}
