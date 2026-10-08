//! The CPU reference Llama (SmolLM2) against outputs computed independently in float64 numpy
//! (scripts/llama_golden.py, D69): a committed tiny model that always runs, and SmolLM2-135M
//! when its weights and goldens are present.

mod common;

use common::{ids, json, tiny_llama_dir};

use std::path::Path;

use ember::compare::{self, Tol};
use ember::llama::{self, Weights};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Logits tolerance for the tiny model (2 layers, E = 24, values O(1)).
const TINY_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};

/// Logits tolerance for SmolLM2-135M against float64 (D69). Measured 2026-10-08: worst absolute
/// error 4.8e-4 (prompts: 4.8e-4, 3.9e-4, 2.3e-4, 1.8e-4) on logits of magnitude up to ~36.
/// GPT-2's 1e-4 doesn't hold: the logits are 5x smaller (GPT-2's reach ~300), so the relative
/// term barely helps, and there are 30 layers instead of 12. It is f32 rounding, not a bug: an
/// independent float32 numpy forward (BLAS sums) is off by 2.7e-4, 3.9e-4, 9.7e-5 and 1.2e-4 on
/// the same prompts. 4x headroom over the worst.
const SMOL_TOL: Tol = Tol {
    abs: 2e-3,
    rel: 1e-4,
};

#[test]
fn tiny_model_matches_numpy() {
    let dir = tiny_llama_dir();
    let w = Weights::load(&dir).unwrap();
    let golden = json(&dir.join("golden.json"));

    let logits = llama::forward(&w, &ids(&golden["ids"])).unwrap();
    let want = read_f32(&dir.join("logits.f32"));
    let stats = compare::check(logits.data(), &want, TINY_TOL).unwrap();
    eprintln!("tiny logits: {stats:?}");

    let out = llama::generate_greedy(&w, &ids(&golden["greedy_prompt"]), 100, |_| {}).unwrap();
    assert_eq!(out, ids(&golden["greedy"]));
}

#[test]
fn smollm2_logits_match_numpy() {
    let Some(dir) = common::smollm2_golden_dir() else {
        return;
    };
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    assert_eq!(w.param_count(), 134_515_008);
    let manifest = json(&dir.join("manifest.json"));
    for p in manifest["prompts"].as_array().unwrap() {
        let logits = llama::forward(&w, &ids(&p["ids"])).unwrap();
        let want = read_f32(&dir.join(p["logits"].as_str().unwrap()));
        let stats = compare::check(logits.data(), &want, SMOL_TOL)
            .unwrap_or_else(|e| panic!("{}: {e}", p["name"]));
        eprintln!("{}: {stats:?}", p["name"]);
    }
}

#[test]
fn smollm2_greedy_matches_numpy() {
    let Some(dir) = common::smollm2_golden_dir() else {
        return;
    };
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    for p in manifest["prompts"].as_array().unwrap() {
        let want = ids(&p["greedy"]);
        let got = llama::generate_greedy(&w, &ids(&p["ids"]), want.len(), |_| {}).unwrap();
        assert_eq!(got, want, "{}", p["text"]);
    }
}
