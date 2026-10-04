//! The CPU reference GPT-2 against outputs computed independently in float64 numpy
//! (scripts/gpt2_golden.py, D12): a committed tiny model that always runs, and GPT-2 124M when
//! its weights and goldens are present.

mod common;

use std::path::{Path, PathBuf};

use ember::compare::{self, Tol};
use ember::gpt2::{self, Weights};
use serde_json::Value;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

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

/// Logits tolerance for the tiny model (2 layers, E = 12, values O(1)).
const TINY_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};

/// Logits tolerance for GPT-2 124M against float64 (D12). Measured 2026-10-05: worst absolute
/// error 6.4e-4 on logits of magnitude ~100, worst relative 1.0e-5, so the relative term is the
/// one doing the work, with 10x headroom.
const GPT2_TOL: Tol = Tol {
    abs: 1e-4,
    rel: 1e-4,
};

#[test]
fn tiny_model_matches_numpy() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_gpt2");
    let w = Weights::load(&dir).unwrap();
    let golden = json(&dir.join("golden.json"));

    let logits = gpt2::forward(&w, &ids(&golden["ids"])).unwrap();
    let want = read_f32(&dir.join("logits.f32"));
    let stats = compare::check(logits.data(), &want, TINY_TOL).unwrap();
    eprintln!("tiny logits: {stats:?}");

    let out = gpt2::generate_greedy(&w, &ids(&golden["greedy_prompt"]), 100, |_| {}).unwrap();
    assert_eq!(out, ids(&golden["greedy"]));
}

#[test]
fn gpt2_logits_match_numpy() {
    let Some(dir) = common::gpt2_golden_dir() else {
        return;
    };
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    for p in manifest["prompts"].as_array().unwrap() {
        let logits = gpt2::forward(&w, &ids(&p["ids"])).unwrap();
        let want = read_f32(&dir.join(p["logits"].as_str().unwrap()));
        let stats = compare::check(logits.data(), &want, GPT2_TOL)
            .unwrap_or_else(|e| panic!("{}: {e}", p["name"]));
        eprintln!("{}: {stats:?}", p["name"]);
    }
}

/// 52 greedy steps, each recomputing the whole sequence (no KV cache until M4): the slowest
/// test, ~20 s with the threaded `linear` (D13).
#[test]
fn gpt2_greedy_matches_numpy() {
    let Some(dir) = common::gpt2_golden_dir() else {
        return;
    };
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    let mut cases: Vec<&Value> = manifest["prompts"].as_array().unwrap().iter().collect();
    // Also pinned by Hugging Face's published output, independently of the numpy script.
    cases.push(&manifest["published"]);
    for p in cases {
        let want = ids(&p["greedy"]);
        let got = gpt2::generate_greedy(&w, &ids(&p["ids"]), want.len(), |_| {}).unwrap();
        assert_eq!(got, want, "{}", p["text"]);
    }
}
