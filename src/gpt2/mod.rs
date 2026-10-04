//! GPT-2 (124M and its larger siblings): config, weights, and the CPU reference forward pass.

mod config;
mod weights;

pub use config::Config;
pub use weights::{Block, Linear, Norm, Weights};
