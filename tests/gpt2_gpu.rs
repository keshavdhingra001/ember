//! GPT-2 on the GPU against the CPU reference (D3, D27): the committed tiny model always, and
//! GPT-2 124M when its weights and goldens are in data/gpt2/.

mod common;

use std::path::{Path, PathBuf};

use common::gpu;
use ember::compare::{Tol, check};
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights};
use ember::gpt2::{self, Config, Weights};
use serde_json::Value;

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// End-to-end logits tolerance, GPU vs CPU reference: the per-op differences (D22) compounded
/// through every layer. Measured 2026-10-05 (D27): tiny model 1.2e-6 abs; GPT-2 124M worst
/// 2.1e-4 abs on logits of magnitude ~100, 2.5e-6 relative, so `rel` carries it (4x headroom).
const TINY_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};
const GPT2_TOL: Tol = Tol {
    abs: 1e-4,
    rel: 1e-5,
};

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_gpt2")
}

#[test]
fn tiny_model_matches_cpu() {
    let g = gpu();
    let w = Weights::load(&tiny_dir()).unwrap();
    let gw = GpuWeights::upload(g, &w);
    let golden = json(&tiny_dir().join("golden.json"));
    let ids = ids(&golden["ids"]);

    let want = gpt2::forward(&w, &ids).unwrap();
    let got = g.read(&gpt2_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
    assert_eq!(got.shape(), want.shape());
    let stats = check(got.data(), want.data(), TINY_TOL).unwrap();
    eprintln!("tiny GPU vs CPU: {stats:?}");

    let prompt = self::ids(&golden["greedy_prompt"]);
    let out = gpt2_gpu::generate_greedy(g, &gw, &prompt, 100, |_| {}).unwrap();
    assert_eq!(out, self::ids(&golden["greedy"]));
}

#[test]
fn gpu_properties_hold_bitwise() {
    // The CPU reference's bitwise properties must hold on the GPU too: every kernel computes a
    // row (or an element) the same way regardless of what else is in the batch (D4).
    let g = gpu();
    let config = Config {
        n_layer: 2,
        n_head: 3,
        n_embd: 12,
        n_ctx: 16,
        vocab_size: 37,
        ln_eps: 1e-5,
    };
    let gw = GpuWeights::upload(g, &Weights::random(config, 7));
    let ids = [5, 0, 36, 17, 17, 2, 30, 11];
    let full = g.read(&gpt2_gpu::forward(g, &gw, &ids).unwrap()).unwrap();

    // Same input, same bits.
    let again = g.read(&gpt2_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
    assert_eq!(bits(again.data()), bits(full.data()));
    // A prefix reproduces its rows exactly (causality, end to end).
    for k in [1, 4, 7] {
        let part = g
            .read(&gpt2_gpu::forward(g, &gw, &ids[..k]).unwrap())
            .unwrap();
        assert_eq!(
            bits(part.data()),
            bits(&full.data()[..k * 37]),
            "prefix {k}"
        );
    }
    // The last-row LM head (D26) gives exactly the last row of the full logits.
    let next = gpt2_gpu::next_logits(g, &gw, &ids).unwrap();
    assert_eq!(bits(&next), bits(&full.data()[7 * 37..]));
    // Errors come from the ops' validation, not from the GPU.
    assert!(gpt2_gpu::forward(g, &gw, &[]).is_err());
    assert!(gpt2_gpu::forward(g, &gw, &[37]).is_err());
    assert!(gpt2_gpu::forward(g, &gw, &[0; 17]).is_err());
}

#[test]
fn gpt2_matches_cpu() {
    let Some(dir) = common::gpt2_golden_dir() else {
        return;
    };
    let g = gpu();
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let gw = GpuWeights::upload(g, &w);
    let manifest = json(&dir.join("manifest.json"));

    for p in manifest["prompts"].as_array().unwrap() {
        let ids = ids(&p["ids"]);
        let want = gpt2::forward(&w, &ids).unwrap();
        let got = g.read(&gpt2_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
        let stats = check(got.data(), want.data(), GPT2_TOL)
            .unwrap_or_else(|e| panic!("{}: {e}", p["name"]));
        eprintln!("{} GPU vs CPU: {stats:?}", p["name"]);
    }

    // Greedy tokens identical on every golden continuation, HF's published one included.
    let mut cases: Vec<&Value> = manifest["prompts"].as_array().unwrap().iter().collect();
    cases.push(&manifest["published"]);
    for p in cases {
        let want = ids(&p["greedy"]);
        let got = gpt2_gpu::generate_greedy(g, &gw, &ids(&p["ids"]), want.len(), |_| {}).unwrap();
        assert_eq!(got, want, "{}", p["text"]);
    }
}
