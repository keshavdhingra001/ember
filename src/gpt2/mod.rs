//! GPT-2 (124M and its larger siblings): config, weights, the CPU reference forward pass, and
//! the same forward pass on the GPU (`gpu`).

mod config;
mod forward;
pub mod gpu;
mod weights;

pub use config::Config;
pub use forward::{argmax_token, forward, generate_greedy, hidden, next_logits};
pub use weights::{Block, Linear, Norm, Weights};
