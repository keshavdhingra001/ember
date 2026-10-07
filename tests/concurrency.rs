//! D65: several threads sharing one `Gpu`, as the test binaries do. With wgpu's lazy zero-init
//! of new buffers (wgpu 30.0.1, Mesa ANV on Iris Xe), a kernel's output buffer sometimes held
//! another buffer's data: a few bad results per run of this test, none on one thread or with
//! an output that was already initialized. `Gpu::alloc` now creates buffers initialized.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use common::gpu;
use ember::{Tensor, ops};

#[test]
fn kernels_on_many_threads_write_their_own_outputs() {
    let g = gpu();
    let bad = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for th in 0..8u32 {
            let bad = &bad;
            s.spawn(move || {
                for it in 0..300u32 {
                    // Sizes vary so buffers of many sizes are created and freed concurrently.
                    let n = 1000 + ((th * 37 + it * 13) % 5000) as usize;
                    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + th as f32).collect();
                    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + it as f32).collect();
                    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                    let (a, b) = (
                        g.upload(&Tensor::new(&[n], a).unwrap()),
                        g.upload(&Tensor::new(&[n], b).unwrap()),
                    );
                    let got = g.read(&ops::add(g, &a, &b).unwrap()).unwrap();
                    if got.data() != &want[..] {
                        bad.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    assert_eq!(bad.into_inner(), 0, "outputs with another buffer's data");
}
