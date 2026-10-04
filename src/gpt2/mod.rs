//! GPT-2 (124M and its larger siblings): config, weights, and the CPU reference forward pass.

mod config;
mod forward;
mod weights;

pub use config::Config;
pub use forward::{forward, generate_greedy, hidden, next_logits};
pub use weights::{Block, Linear, Norm, Weights};
