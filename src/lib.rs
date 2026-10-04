//! ember: an LLM inference engine on WebGPU (wgpu + WGSL). See DESIGN.md.

pub mod compare;
pub mod cpu;
pub mod error;
pub mod gpt2;
pub mod gpu;
pub mod ops;
pub mod rng;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;

pub use error::{Error, Result};
pub use gpu::Gpu;
pub use tensor::{GpuTensor, Tensor};
pub use tokenizer::Tokenizer;
