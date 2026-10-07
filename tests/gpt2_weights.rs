//! Loading the real GPT-2 124M checkpoint. Skips when `data/gpt2/` is missing (D15).

mod common;

use ember::gpt2::Weights;
use ember::safetensors::SafeTensors;

#[test]
fn loads_gpt2_small() {
    let Some(dir) = common::gpt2_dir() else {
        return;
    };
    let w = Weights::load(&dir).unwrap();
    let c = &w.config;
    assert_eq!((c.n_layer, c.n_head, c.n_embd), (12, 12, 768));
    assert_eq!(w.wte.shape(), &[50257, 768]);
    assert_eq!(w.blocks.len(), 12);

    // Spot-check the transpose against the raw file: raw c_fc is [in=768, out=3072], ours is
    // [out, in], so ours[o][i] must equal raw[i][o].
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let raw = st.tensor("h.11.mlp.c_fc.weight").unwrap();
    let ours = &w.blocks[11].fc.w;
    for (i, o) in [(0, 0), (1, 0), (0, 1), (767, 3071), (300, 2000)] {
        assert_eq!(ours.data()[o * 768 + i], raw.data()[i * 3072 + o]);
    }

    // Count parameters, leaving out the 12 causal-mask buffers (`attn.bias`), which are
    // constants, not learned weights. GPT-2 "124M" has 124,439,808.
    let params: usize = st
        .names()
        .filter(|n| !n.ends_with(".attn.bias"))
        .map(|n| st.entry(n).unwrap().shape.iter().product::<usize>())
        .sum();
    assert_eq!(params, 124_439_808);
    assert_eq!(w.param_count(), params);
}
