//! Shared test helpers. One GPU device for the whole test binary: creating a device costs
//! tens of milliseconds and the tests run in parallel threads (wgpu devices are Sync).
#![allow(dead_code)] // each test binary uses a different subset

use std::path::PathBuf;
use std::sync::OnceLock;

use ember::Gpu;

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
