//! The cached path's workspace (D58–D61, D64): a steady-state decode step creates no GPU
//! objects, and running on fixed, reused buffers gives the same bits as composing the one-op
//! functions. The creation counter is device-wide, so these tests live in their own binary and
//! take turns.

mod common;

use std::sync::Mutex;

use common::gpu;
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights, KvCache};
use ember::gpt2::{Config, Weights};
use ember::gpu::Binds;
use ember::{Gpu, GpuTensor, Tensor, ops};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
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
fn a_decode_step_creates_no_gpu_objects() {
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_random()).unwrap();
    let mut cache = KvCache::new(g, &gw.config);
    gpt2_gpu::extend(g, &gw, &mut cache, &[1, 5, 9, 2]).unwrap();
    // The first decode step makes the T = 1 bind groups (other kernels than the prefill's).
    gpt2_gpu::extend(g, &gw, &mut cache, &[3]).unwrap();
    let (created, binds) = (g.created(), cache.bind_groups());
    for id in [4, 30, 0] {
        gpt2_gpu::extend(g, &gw, &mut cache, &[id]).unwrap();
    }
    assert_eq!(
        g.created() - created,
        0,
        "GPU objects created by 3 decode steps"
    );
    assert_eq!(cache.bind_groups(), binds);
    // The counter does count: the one-op path creates as it goes.
    gpt2_gpu::next_logits(g, &gw, &[1, 2]).unwrap();
    assert!(g.created() > created);
}

#[test]
fn one_entry_with_two_parameter_sets_in_one_recording_is_an_error() {
    // Its uniform's write would land before the whole submit, so the first dispatch would run
    // with the second one's parameters (D59).
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let g = gpu();
    let x = g.upload(&Tensor::new(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0]).unwrap());
    let ones = g.upload(&Tensor::new(&[4], vec![1.0; 4]).unwrap());
    let zeros = g.upload(&Tensor::new(&[4], vec![0.0; 4]).unwrap());
    let out = g.alloc(&[2, 4]);
    let mut binds = Binds::default();
    let mut rec = g.rec_with(&mut binds);
    let norm =
        |rec: &mut ember::gpu::Rec, eps| ops::layer_norm_into(rec, &x, &ones, &zeros, eps, &out);
    norm(&mut rec, 1e-5).unwrap();
    norm(&mut rec, 1e-5).unwrap(); // same parameters: fine
    let e = norm(&mut rec, 1e-3).unwrap_err();
    assert!(e.to_string().contains("different parameters"), "{e}");
    drop(rec);
    // In the next recording the entry takes the new parameters.
    let mut rec = g.rec_with(&mut binds);
    norm(&mut rec, 1e-3).unwrap();
    rec.submit();
    assert_eq!(binds.len(), 1);
}

/// The logits of the last position after `ids`, from the one-op functions composed by hand:
/// every op its own allocation and submit, nothing reused.
fn logits_one_op_at_a_time(g: &Gpu, w: &GpuWeights, ids: &[u32]) -> Vec<f32> {
    let c = &w.config;
    let mut x = ops::embed(g, &w.wte_t, &w.wpe, ids, 0).unwrap();
    for b in &w.blocks {
        let lin =
            |x: &GpuTensor, l: &gpt2_gpu::GpuLinear| ops::linear(g, x, &l.w, Some(&l.b)).unwrap();
        let h = ops::layer_norm(g, &x, &b.ln_1.gain, &b.ln_1.bias, c.ln_eps).unwrap();
        let a = ops::causal_attention(g, &lin(&h, &b.qkv), c.n_head).unwrap();
        x = ops::add(g, &x, &lin(&a, &b.attn_out)).unwrap();
        let h = ops::layer_norm(g, &x, &b.ln_2.gain, &b.ln_2.bias, c.ln_eps).unwrap();
        let m = ops::gelu(g, &lin(&h, &b.fc)).unwrap();
        x = ops::add(g, &x, &lin(&m, &b.fc_out)).unwrap();
    }
    let h = ops::layer_norm(g, &x, &w.ln_f.gain, &w.ln_f.bias, c.ln_eps).unwrap();
    let last = ops::row(g, &h, ids.len() - 1).unwrap();
    let logits = ops::linear(g, &last, &w.wte_t, None).unwrap();
    g.read(&logits).unwrap().data().to_vec()
}

fn check_against_one_op(g: &Gpu, w: &GpuWeights, prompt: &[u32], more: &[u32]) {
    let mut cache = KvCache::new(g, &w.config);
    let got = gpt2_gpu::extend(g, w, &mut cache, prompt).unwrap();
    assert_eq!(
        bits(&got),
        bits(&logits_one_op_at_a_time(g, w, prompt)),
        "prefill"
    );
    let mut ids = prompt.to_vec();
    for &id in more {
        ids.push(id);
        let got = gpt2_gpu::extend(g, w, &mut cache, &[id]).unwrap();
        let want = logits_one_op_at_a_time(g, w, &ids);
        assert_eq!(
            bits(&got),
            bits(&want),
            "decode at position {}",
            ids.len() - 1
        );
    }
}

#[test]
fn workspace_path_matches_one_op_at_a_time() {
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let g = gpu();
    let gw = GpuWeights::upload(g, &tiny_random()).unwrap();
    check_against_one_op(g, &gw, &[1, 5, 9, 2, 36], &[3, 0, 17]);
    // 9 rows: the prefill's linears run the tiled matmul instead of matvec_rows (D57).
    check_against_one_op(g, &gw, &[1, 5, 9, 2, 36, 8, 8, 1, 0], &[3]);
}

#[test]
fn gpt2_workspace_path_matches_one_op_at_a_time() {
    let _turn = ONE_AT_A_TIME.lock().unwrap();
    let Some(dir) = common::gpt2_dir() else {
        return;
    };
    let g = gpu();
    let gw = GpuWeights::upload(g, &Weights::load(&dir).unwrap()).unwrap();
    check_against_one_op(g, &gw, &[40, 2883, 6155, 351, 616], &[13, 314]);
}
