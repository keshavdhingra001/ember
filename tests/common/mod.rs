//! Shared test helpers. One GPU device for the whole test binary: creating a device costs
//! tens of milliseconds and the tests run in parallel threads (wgpu devices are Sync).

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
