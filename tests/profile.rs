//! The timestamp profiler (D36, D37). Profiling is device-wide (every dispatch on the `Gpu` is
//! recorded while it's on), so these tests live in their own binary and take turns.

mod common;

use std::sync::Mutex;

use common::{gpu, tiny_dir};
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights, KvCache};
use ember::gpt2::{Config, Weights};
use ember::profile::{MAX_DISPATCHES, by_kernel};
use ember::{Gpu, Tensor, ops};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The shared GPU if it can profile; otherwise says why and the test returns early.
fn profiling_gpu() -> Option<&'static Gpu> {
    let g = gpu();
    if g.features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        Some(g)
    } else {
        eprintln!("SKIPPED: the adapter has no TIMESTAMP_QUERY");
        None
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn one_label_and_time_per_dispatch_in_order() {
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let Some(g) = profiling_gpu() else { return };
    let config = Config {
        n_layer: 2,
        n_head: 3,
        n_embd: 12,
        n_ctx: 16,
        vocab_size: 37,
        ln_eps: 1e-5,
    };
    let gw = GpuWeights::upload(g, &Weights::random(config, 7)).unwrap();
    let mut cache = KvCache::new(g, &gw.config);

    g.profile_start().unwrap();
    gpt2_gpu::extend(g, &gw, &mut cache, &[1, 2, 3]).unwrap();
    let times = g.profile_finish().unwrap();

    // Three rows: the block's matrices (12 to 48 outputs) run on the multi-row matvec that splits
    // K (D57, D56); the residual adds and GELU are their epilogues (D62).
    let block = [
        "layer_norm",
        "matvec_rows_split",
        "kv_write",
        "attention",
        "matvec_rows_split+res",
        "layer_norm",
        "matvec_rows_split+gelu",
        "matvec_rows_split+res",
    ];
    let mut want = vec!["embed"];
    want.extend(block);
    want.extend(block);
    want.extend(["layer_norm", "matvec_split"]); // ln_f, then the 37-wide LM head on the last row (D44, D56)
    let got: Vec<&str> = times.iter().map(|t| t.kernel).collect();
    assert_eq!(got, want);
    for t in &times {
        assert!(t.ns.is_finite() && t.ns > 0.0 && t.ns < 1e9, "{t:?}");
    }
    let rows = by_kernel(&times)
        .into_iter()
        .find(|k| k.0 == "matvec_rows_split")
        .unwrap();
    // One per block (qkv); attn_out, fc and fc_out are grouped under their epilogue's name.
    assert_eq!(rows.1, 2);
}

#[test]
fn profiling_does_not_change_results() {
    // D4, D37: the same logits, bit for bit, with timestamps on and off.
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let Some(g) = profiling_gpu() else { return };
    let gw = GpuWeights::upload(g, &Weights::load(&tiny_dir()).unwrap()).unwrap();
    let ids = [3, 1, 4, 1, 5, 9, 2, 6];
    let plain = gpt2_gpu::next_logits(g, &gw, &ids).unwrap();
    g.profile_start().unwrap();
    let profiled = gpt2_gpu::next_logits(g, &gw, &ids).unwrap();
    assert!(!g.profile_finish().unwrap().is_empty());
    assert_eq!(bits(&plain), bits(&profiled));
}

#[test]
fn windows_are_independent_and_must_be_started() {
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let Some(g) = profiling_gpu() else { return };
    assert!(g.profile_finish().is_err());
    let x = g.upload(&Tensor::new(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    g.profile_start().unwrap();
    ops::add(g, &x, &x).unwrap();
    ops::gelu(g, &x).unwrap();
    // A new start discards the unfinished window.
    g.profile_start().unwrap();
    ops::gelu(g, &x).unwrap();
    let times = g.profile_finish().unwrap();
    assert_eq!(times.len(), 1);
    assert_eq!(times[0].kernel, "gelu");
    // Off again: nothing is recorded, and finishing needs a new start.
    ops::add(g, &x, &x).unwrap();
    assert!(g.profile_finish().is_err());
}

#[test]
fn overflowing_the_window_is_an_error() {
    // Silently dropping dispatches would under-report the step.
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let Some(g) = profiling_gpu() else { return };
    let x = g.upload(&Tensor::new(&[1], vec![1.0]).unwrap());
    g.profile_start().unwrap();
    for _ in 0..=MAX_DISPATCHES {
        ops::gelu(g, &x).unwrap();
    }
    let e = g.profile_finish().unwrap_err();
    assert!(e.to_string().contains("overflowed: 1 dispatches"), "{e}");
}
