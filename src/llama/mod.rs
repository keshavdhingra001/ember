//! Llama-family models (SmolLM2, D67): config, weights, the CPU reference forward pass, and
//! the same forward pass on the GPU.

mod config;
mod forward;
pub mod gpu;
mod weights;

pub use config::Config;
pub use forward::{forward, generate_greedy, hidden, next_logits};
pub use weights::{Block, Weights};
