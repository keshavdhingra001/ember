//! GPT-2 on the GPU against the CPU reference (D3, D27): the committed tiny model always, and
//! GPT-2 124M when its weights and goldens are in data/gpt2/.

mod common;

use common::{gpu, ids, json, tiny_dir};
use ember::compare::{Tol, check};
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights, KvCache};
use ember::gpt2::{self, Config, Weights};
use serde_json::Value;

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

#[test]
fn tiny_model_matches_cpu() {
    let g = gpu();
    let w = Weights::load(&tiny_dir()).unwrap();
    let gw = GpuWeights::upload(g, &w).unwrap();
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
    let gw = GpuWeights::upload(g, &Weights::random(config, 7)).unwrap();
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

fn tiny_random() -> Weights {
    let config = Config {
        n_layer: 2,
        n_head: 3,
        n_embd: 12,
        n_ctx: 16,
        vocab_size: 37,
        ln_eps: 1e-5,
    };
    Weights::random(config, 7)
}

#[test]
fn cached_decode_is_bitwise_full_recompute() {
    // D33: after every step, the cached logits have exactly the bits of recomputing the whole
    // prefix. Any position or cache-row mistake shows up here, however small its effect.
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_random()).unwrap();
    let ids = [5, 0, 36, 17, 17, 2, 30, 11, 8, 23, 1, 4, 9, 33, 12, 6]; // fills n_ctx = 16
    let mut cache = KvCache::new(g, &gw.config);

    // Prefill 4 tokens, then decode one at a time to the end of the context.
    let got = gpt2_gpu::extend(g, &gw, &mut cache, &ids[..4]).unwrap();
    assert_eq!(
        bits(&got),
        bits(&gpt2_gpu::next_logits(g, &gw, &ids[..4]).unwrap())
    );
    for t in 4..ids.len() {
        let got = gpt2_gpu::extend(g, &gw, &mut cache, &ids[t..t + 1]).unwrap();
        let want = gpt2_gpu::next_logits(g, &gw, &ids[..=t]).unwrap();
        assert_eq!(bits(&got), bits(&want), "step at position {t}");
    }
    assert_eq!(cache.len(), 16);
    // The context is full: one more token must fail and leave the cache as it was.
    assert!(gpt2_gpu::extend(g, &gw, &mut cache, &[0]).is_err());
    assert_eq!(cache.len(), 16);

    // Reuse after clear(), with the prefill split into uneven chunks: stale rows from the
    // first sequence must not leak into the second.
    cache.clear();
    let other = [9, 9, 1, 2, 3, 30, 31];
    gpt2_gpu::extend(g, &gw, &mut cache, &other[..2]).unwrap();
    gpt2_gpu::extend(g, &gw, &mut cache, &other[2..5]).unwrap();
    let got = gpt2_gpu::extend(g, &gw, &mut cache, &other[5..]).unwrap();
    assert_eq!(
        bits(&got),
        bits(&gpt2_gpu::next_logits(g, &gw, &other).unwrap())
    );
}

#[test]
fn truncate_rewinds_to_an_exact_prefix() {
    // Decode past position 6, rewind to 6, take a different branch: the logits must be exactly
    // those of recomputing the new sequence, so no row past the cut leaks in.
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_random()).unwrap();
    let mut cache = KvCache::new(g, &gw.config);
    let ids = [5, 0, 36, 17, 17, 2, 30, 11, 8];
    gpt2_gpu::extend(g, &gw, &mut cache, &ids).unwrap();
    cache.truncate(6).unwrap();
    assert_eq!(cache.len(), 6);
    let got = gpt2_gpu::extend(g, &gw, &mut cache, &[4]).unwrap();
    let want = gpt2_gpu::next_logits(g, &gw, &[5, 0, 36, 17, 17, 2, 4]).unwrap();
    assert_eq!(bits(&got), bits(&want));
    assert!(cache.truncate(8).is_err());
    cache.truncate(7).unwrap(); // to the current length: a no-op, not an error
    assert_eq!(cache.len(), 7);
}

#[test]
fn a_cache_for_another_model_is_rejected() {
    // Same width and context, one layer fewer: every per-layer shape check passes, so only the
    // layer count can tell. The cache must stay empty afterwards.
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_random()).unwrap();
    let one_layer = Config {
        n_layer: 1,
        ..gw.config.clone()
    };
    let mut cache = KvCache::new(g, &one_layer);
    let e = gpt2_gpu::extend(g, &gw, &mut cache, &[1, 2]).unwrap_err();
    assert!(e.to_string().contains("1 layers, the model 2"), "{e}");
    assert!(cache.is_empty());
    // Narrower rows or a shorter context are caught by the ops' own shape checks.
    for config in [
        Config {
            n_embd: 6,
            ..gw.config.clone()
        },
        Config {
            n_ctx: 1,
            ..gw.config.clone()
        },
    ] {
        let mut cache = KvCache::new(g, &config);
        assert!(gpt2_gpu::extend(g, &gw, &mut cache, &[1, 2]).is_err());
        assert!(cache.is_empty());
    }
}

#[test]
fn gpt2_cached_decode_is_bitwise_full_recompute() {
    let Some(dir) = common::gpt2_golden_dir() else {
        return;
    };
    let g = gpu();
    let gw = GpuWeights::upload(g, &Weights::load(dir.parent().unwrap()).unwrap()).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    let p = &manifest["prompts"][0];
    let mut seq = ids(&p["ids"]);
    let mut cache = KvCache::new(g, &gw.config);
    let mut new = seq.clone();
    for step in 0..8 {
        let got = gpt2_gpu::extend(g, &gw, &mut cache, &new).unwrap();
        let want = gpt2_gpu::next_logits(g, &gw, &seq).unwrap();
        assert_eq!(bits(&got), bits(&want), "step {step}");
        let next = ember::cpu::argmax(&got).unwrap() as u32;
        seq.push(next);
        new = vec![next];
    }
}

#[test]
fn gpt2_matches_cpu() {
    let Some(dir) = common::gpt2_golden_dir() else {
        return;
    };
    let g = gpu();
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let gw = GpuWeights::upload(g, &w).unwrap();
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
