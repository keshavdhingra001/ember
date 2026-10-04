//! ember: an LLM inference engine on WebGPU (wgpu + WGSL). See DESIGN.md.

pub mod compare;
pub mod cpu;
pub mod error;
pub mod gpu;
pub mod ops;
pub mod rng;
pub mod tensor;

pub use error::{Error, Result};
pub use gpu::Gpu;
pub use tensor::{GpuTensor, Tensor};
