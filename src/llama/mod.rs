//! Llama-family models (SmolLM2, D67): config, weights, the CPU reference forward pass, and
//! the same forward pass on the GPU.

mod config;

pub use config::Config;
