//! Shared test helpers. One GPU device for the whole test binary: creating a device costs
//! tens of milliseconds and the tests run in parallel threads (wgpu devices are Sync).
#![allow(dead_code)] // each test binary uses a different subset

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ember::Gpu;
use serde_json::Value;

pub fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        Gpu::new().unwrap_or_else(|e| {
            panic!("GPU tests need a WebGPU adapter (Vulkan driver): {e}. Try `ember info`.")
        })
    })
}

/// `data/gpt2/` if the GPT-2 checkpoint has been fetched; otherwise prints how to fetch it and
/// returns `None`, and the calling test returns early (D15).
pub fn gpt2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2");
    if dir.join("model.safetensors").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: GPT-2 weights not found in data/gpt2/. Fetch them with scripts/fetch_gpt2.sh"
        );
        None
    }
}

/// `data/smollm2-135m/` if SmolLM2-135M has been fetched; otherwise prints how and returns
/// `None` (D15).
pub fn smollm2_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/smollm2-135m");
    if dir.join("model.safetensors").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: SmolLM2-135M not found in data/smollm2-135m/. \
             Fetch it with scripts/fetch_smollm2.sh"
        );
        None
    }
}

/// `data/gpt2/golden/` if the reference outputs have been generated (needs the weights too).
pub fn gpt2_golden_dir() -> Option<PathBuf> {
    let dir = gpt2_dir()?.join("golden");
    if dir.join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: GPT-2 golden outputs not found. Generate them with \
             `python scripts/gpt2_golden.py` (see the script's header for the venv)"
        );
        None
    }
}

/// `data/smollm2-135m/golden/` if the reference outputs have been generated.
pub fn smollm2_golden_dir() -> Option<PathBuf> {
    let dir = smollm2_dir()?.join("golden");
    if dir.join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: SmolLM2 golden outputs not found. Generate them with \
             `python scripts/llama_golden.py smollm2` (see the script's header for the venv)"
        );
        None
    }
}

/// The committed 2-layer Llama-family model (bf16 weights, 6 query heads over 2 key/value
/// heads) and its numpy goldens.
pub fn tiny_llama_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama")
}

/// The committed 2-layer model, its tokenizer files and its numpy goldens.
pub fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_gpt2")
}

pub fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A JSON array of token ids.
pub fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}
