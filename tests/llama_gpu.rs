//! Llama-family models on the GPU against the CPU reference (D69): the committed tiny model
//! always, SmolLM2-135M when its weights and goldens are in data/smollm2-135m/.

mod common;

use common::{gpu, ids, json, tiny_llama_dir};
use ember::cache::CacheModel;
use ember::compare::{Tol, check};
use ember::llama::gpu::{self as llama_gpu, GpuWeights, KvCache};
use ember::llama::{self, Weights};

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// End-to-end logits tolerance on the tiny model, GPU vs CPU, as for GPT-2's tiny model.
const TINY_TOL: Tol = Tol {
    abs: 1e-5,
    rel: 1e-5,
};

/// SmolLM2-135M, GPU vs CPU. Measured 2026-10-08: worst 4.2e-4 absolute (prompts: 3.1e-4,
/// 4.2e-4, 2.3e-4, 1.6e-4), the same size as the CPU's own distance from float64 (D69), so the
/// same bound (4.7x headroom).
const SMOL_TOL: Tol = Tol {
    abs: 2e-3,
    rel: 1e-4,
};

/// The tiny fixture with its context raised to 100 positions, so decoding crosses attention's
/// 64-key chunk boundary (D63). RoPE's tables are built at upload for whatever `n_ctx` says.
fn tiny_long() -> Weights {
    let mut w = Weights::load(&tiny_llama_dir()).unwrap();
    w.config.n_ctx = 100;
    w
}

#[test]
fn tiny_model_matches_cpu() {
    let g = gpu();
    let w = Weights::load(&tiny_llama_dir()).unwrap();
    let gw = GpuWeights::upload(g, &w).unwrap();
    let golden = json(&tiny_llama_dir().join("golden.json"));
    let ids = ids(&golden["ids"]);

    let want = llama::forward(&w, &ids).unwrap();
    let got = g.read(&llama_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
    assert_eq!(got.shape(), want.shape());
    let stats = check(got.data(), want.data(), TINY_TOL).unwrap();
    eprintln!("tiny GPU vs CPU: {stats:?}");

    let prompt = self::ids(&golden["greedy_prompt"]);
    let want = self::ids(&golden["greedy"]);
    let out = llama_gpu::generate_greedy(g, &gw, &prompt, 100, |_| {}).unwrap();
    assert_eq!(out, want);
    let out = llama_gpu::generate_greedy_uncached(g, &gw, &prompt, 100, |_| {}).unwrap();
    assert_eq!(out, want);
}

#[test]
fn cached_decode_is_bitwise_full_recompute() {
    // D33 for Llama: prefill 5 tokens, then decode one at a time across the 64-key chunk
    // boundary; every step must have the bits of recomputing the whole prefix. A RoPE position
    // off by one, or a grouped head reading the wrong cache column, shows up here.
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_long()).unwrap();
    let seq: Vec<u32> = (0..70u32).map(|i| (i * 11 + 3) % 37).collect();
    let mut cache = KvCache::new(g, &gw.config);
    assert_eq!(cache.n_ctx(), 100);
    let got = llama_gpu::extend(g, &gw, &mut cache, &seq[..5]).unwrap();
    assert_eq!(
        bits(&got),
        bits(&llama_gpu::next_logits(g, &gw, &seq[..5]).unwrap())
    );
    for t in 5..seq.len() {
        let got = llama_gpu::extend(g, &gw, &mut cache, &seq[t..t + 1]).unwrap();
        let want = llama_gpu::next_logits(g, &gw, &seq[..=t]).unwrap();
        assert_eq!(bits(&got), bits(&want), "step at position {t}");
    }
    // Reuse after clear(), prefill in uneven chunks.
    cache.clear();
    llama_gpu::extend(g, &gw, &mut cache, &seq[..3]).unwrap();
    let got = llama_gpu::extend(g, &gw, &mut cache, &seq[3..9]).unwrap();
    assert_eq!(
        bits(&got),
        bits(&llama_gpu::next_logits(g, &gw, &seq[..9]).unwrap())
    );
}

#[test]
fn truncate_rewinds_to_an_exact_prefix() {
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_long()).unwrap();
    let mut cache = KvCache::new(g, &gw.config);
    llama_gpu::extend(g, &gw, &mut cache, &[5, 0, 36, 17, 17, 2, 30, 11, 8]).unwrap();
    cache.truncate(6).unwrap();
    let got = llama_gpu::extend(g, &gw, &mut cache, &[4]).unwrap();
    let want = llama_gpu::next_logits(g, &gw, &[5, 0, 36, 17, 17, 2, 4]).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn the_cache_size_is_bounded() {
    let g = gpu();
    let w = Weights::load(&tiny_llama_dir()).unwrap(); // 16 positions
    let gw = GpuWeights::upload(g, &w).unwrap();
    assert_eq!(w.config.default_ctx(), 16); // min(2048, n_ctx), D76
    assert!(KvCache::with_ctx(g, &w.config, 17).is_err());
    assert!(KvCache::with_ctx(g, &w.config, 0).is_err());
    // A smaller cache than the model allows: filling it is fine, one more is refused and
    // leaves the cache as it was.
    let mut cache = KvCache::with_ctx(g, &w.config, 6).unwrap();
    llama_gpu::extend(g, &gw, &mut cache, &[1, 2, 3, 4, 5, 6]).unwrap();
    let e = llama_gpu::extend(g, &gw, &mut cache, &[7]).unwrap_err();
    assert!(e.to_string().contains("6 positions"), "{e}");
    assert_eq!(cache.len(), 6);
    // Past the model's own positions, the uncached path refuses too.
    assert!(llama_gpu::next_logits(g, &gw, &[1; 17]).is_err());
}

#[test]
fn smollm2_matches_cpu_and_numpy() {
    let Some(dir) = common::smollm2_golden_dir() else {
        return;
    };
    let g = gpu();
    let w = Weights::load(dir.parent().unwrap()).unwrap();
    let gw = GpuWeights::upload(g, &w).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    for p in manifest["prompts"].as_array().unwrap() {
        let ids = ids(&p["ids"]);
        let want = llama::forward(&w, &ids).unwrap();
        let got = g.read(&llama_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
        let stats = check(got.data(), want.data(), SMOL_TOL)
            .unwrap_or_else(|e| panic!("{}: {e}", p["name"]));
        eprintln!("{} GPU vs CPU: {stats:?}", p["name"]);
        // Greedy with the KV cache gives numpy's tokens.
        let want = self::ids(&p["greedy"]);
        let got = llama_gpu::generate_greedy(g, &gw, &ids, want.len(), |_| {}).unwrap();
        assert_eq!(got, want, "{}", p["text"]);
    }
}

#[test]
fn smollm2_cached_decode_is_bitwise_full_recompute() {
    let Some(dir) = common::smollm2_golden_dir() else {
        return;
    };
    let g = gpu();
    let gw = GpuWeights::upload(g, &Weights::load(dir.parent().unwrap()).unwrap()).unwrap();
    let manifest = json(&dir.join("manifest.json"));
    let mut seq = ids(&manifest["prompts"][0]["ids"]);
    let mut cache = KvCache::new(g, &gw.config);
    let mut new = seq.clone();
    for step in 0..8 {
        let got = llama_gpu::extend(g, &gw, &mut cache, &new).unwrap();
        let want = llama_gpu::next_logits(g, &gw, &seq).unwrap();
        assert_eq!(bits(&got), bits(&want), "step {step}");
        let next = ember::cpu::argmax(&got).unwrap() as u32;
        seq.push(next);
        new = vec![next];
    }
}

#[test]
fn smollm2_360m_runs_by_config_alone() {
    // D67's scale check: E = 960, 32 layers, 15 query heads over 5 key/value heads, with no
    // code of its own. Fetch with `scripts/fetch_smollm2.sh 360m`. GPU vs CPU only (no numpy
    // goldens): logits within the 135M bound, greedy tokens identical.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/smollm2-360m");
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIPPED: SmolLM2-360M not found. Fetch it with scripts/fetch_smollm2.sh 360m");
        return;
    }
    let g = gpu();
    let w = Weights::load(&dir).unwrap();
    assert_eq!(
        (w.config.n_embd, w.config.n_head, w.config.n_kv_head),
        (960, 15, 5)
    );
    let gw = GpuWeights::upload(g, &w).unwrap();
    let tok = ember::Tokenizer::load_hf(&dir).unwrap();
    let ids = tok.encode("I enjoy walking with my cute dog").unwrap();
    let want = llama::forward(&w, &ids).unwrap();
    let got = g.read(&llama_gpu::forward(g, &gw, &ids).unwrap()).unwrap();
    let stats = check(got.data(), want.data(), SMOL_TOL).unwrap();
    eprintln!("360M GPU vs CPU: {stats:?}");
    let cpu = llama::generate_greedy(&w, &ids, 8, |_| {}).unwrap();
    let gpu = llama_gpu::generate_greedy(g, &gw, &ids, 8, |_| {}).unwrap();
    assert_eq!(gpu, cpu);
    eprintln!("360M continues: {:?}", tok.decode(&gpu).unwrap());
}

#[test]
fn a_cache_too_big_for_the_device_is_refused() {
    // SmolLM2-135M's shape at its full 8192 positions, no weights needed. Attention's chunk
    // partials for a full-length prefill (D63) grow with positions squared: 8192 rows x 9 heads
    // x 128 chunks x 66 floats = 2.49 GB, past any buffer this device takes. The cache must say
    // so, not hand the driver a buffer it rejects.
    let g = gpu();
    let c = llama::Config {
        n_layer: 30,
        n_head: 9,
        n_kv_head: 3,
        n_embd: 576,
        n_ff: 1536,
        vocab_size: 49152,
        n_ctx: 8192,
        rope_theta: 100000.0,
        rms_eps: 1e-5,
    };
    let limit = g
        .limits
        .max_storage_buffer_binding_size
        .min(g.limits.max_buffer_size);
    assert!(
        limit < 8192 * 9 * 128 * 66 * 4,
        "this device takes the buffer"
    );
    match KvCache::with_ctx(g, &c, 8192) {
        Ok(_) => panic!("a cache of 8192 positions was accepted"),
        Err(e) => assert!(e.to_string().contains("exceeds"), "{e}"),
    }
    // The default size fits.
    assert_eq!(KvCache::new(g, &c).n_ctx(), 2048);
}

#[test]
fn a_recompute_too_big_for_the_device_is_refused() {
    // The uncached path allocates the same partials per call: the tiny model (6 heads, d = 4)
    // at 32768 positions needs 32768 x 6 x 512 x 6 floats = 2.4 GB of them.
    let g = gpu();
    let mut w = Weights::load(&tiny_llama_dir()).unwrap();
    w.config.n_ctx = 32768;
    let gw = GpuWeights::upload(g, &w).unwrap();
    let e = llama_gpu::next_logits(g, &gw, &vec![1; 32768]).unwrap_err();
    assert!(e.to_string().contains("exceeds"), "{e}");
}
