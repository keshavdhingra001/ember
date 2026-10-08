"""Reference outputs for ember's Llama-family models (D69), in the same style as gpt2_golden.py:
token ids from Hugging Face's `tokenizers`, forward passes in float64 numpy.

Setup: the venv of gpt2_golden.py (numpy, tokenizers).

Usage (from the repository root):
    target/tmp/venv/bin/python scripts/llama_golden.py tokenizer  # tests/fixtures/smollm2_tokenizer_cases.json
    target/tmp/venv/bin/python scripts/llama_golden.py tiny       # tests/fixtures/tiny_llama/
    target/tmp/venv/bin/python scripts/llama_golden.py smollm2    # data/smollm2-135m/golden/ (needs the weights)

The forward pass follows Hugging Face's LlamaForCausalLM: RMSNorm, rotary embeddings on q and
k with dimension i paired with i + d/2 ("rotate_half"), grouped-query attention, a SwiGLU MLP,
tied embeddings. Rotary angles are computed in float64 here (Hugging Face uses float32; the
difference is far below bf16 weights' precision and is the same for the CPU and GPU, D70).
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_golden import TOKENIZER_CASES, greedy_from, save_logits, softmax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMOL_DIR = ROOT / "data" / "smollm2-135m"
FIXTURES = ROOT / "tests" / "fixtures"

# ---------------------------------------------------------------- tokenizer cases

# GPT-2's tricky strings, plus what SmolLM2 adds (D75): digits split one by one (with and
# without a leading space, non-ASCII digits) and special tokens matched whole, including next
# to text, back to back, and a near miss.
SMOL_CASES = TOKENIZER_CASES + [
    "2024-10-08 at 17:49, pi = 3.14159, x10 and 10x",
    "  42  ",
    "Arabic-Indic ٣٤٥ and fullwidth １２３ and superscript ²³ and ⅷ",
    "<|endoftext|>",
    "Hello<|endoftext|>world",
    "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_end|><|im_end|>",
    "<|im_start <|endoftext| <|endoftext|",
]


def tokenizer_cases():
    from tokenizers import Tokenizer, __version__

    tok = Tokenizer.from_file(str(SMOL_DIR / "tokenizer.json"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in SMOL_CASES]
    out = FIXTURES / "smollm2_tokenizer_cases.json"
    doc = {
        "source": f"huggingface tokenizers {__version__}, HuggingFaceTB/SmolLM2-135M tokenizer.json",
        "cases": cases,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- safetensors (bf16)

def read_safetensors(path):
    """name -> float64 array, from F32 or BF16 (bf16 is the top half of an f32)."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    data = memoryview(raw)[8 + n :]
    out = {}
    for name, e in header.items():
        if name == "__metadata__":
            continue
        begin, end = e["data_offsets"]
        if e["dtype"] == "BF16":
            h = np.frombuffer(data[begin:end], dtype="<u2").astype(np.uint32) << 16
            a = h.view(np.float32)
        else:
            assert e["dtype"] == "F32", (name, e["dtype"])
            a = np.frombuffer(data[begin:end], dtype="<f4")
        out[name] = a.reshape(e["shape"]).astype(np.float64)
    return out


def to_bf16(a):
    """float32 -> bf16 bits (uint16), round to nearest even. No NaN handling: inputs are finite."""
    b = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = b + 0x7FFF + ((b >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def write_bf16_safetensors(path, tensors):
    """tensors: list of (name, float32 array), stored as BF16 in the order given."""
    header, chunks, offset = {}, [], 0
    for name, a in tensors:
        b = to_bf16(a).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        chunks.append(b)
        offset += len(b)
    header["__metadata__"] = {"format": "pt", "source": "scripts/llama_golden.py"}
    h = json.dumps(header, separators=(",", ":")).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))


# ---------------------------------------------------------------- the model

def rms_norm(x, g, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope(x, pos, theta):
    """x: [heads, T, d] at positions pos [T]. Dimension i turns with i + d/2 by pos * theta^(-2i/d)."""
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)  # [d/2]
    ang = np.outer(pos, inv)  # [T, d/2]
    cos = np.concatenate([np.cos(ang)] * 2, axis=-1)  # [T, d]
    sin = np.concatenate([np.sin(ang)] * 2, axis=-1)
    half = d // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)  # rotate_half
    return x * cos + rotated * sin


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(W, cfg, ids):
    """Logits [T, vocab] for token ids [T]. W holds float64 arrays, Hugging Face layout [out, in]."""
    T, E = len(ids), cfg["hidden_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    D, eps, theta = E // H, cfg["rms_norm_eps"], cfg["rope_theta"]
    pos = np.arange(T, dtype=np.float64)
    x = W["model.embed_tokens.weight"][ids]
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms_norm(x, W[p + "input_layernorm.weight"], eps)
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, KV, D).transpose(1, 0, 2)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, KV, D).transpose(1, 0, 2)
        q, k = rope(q, pos, theta), rope(k, pos, theta)
        k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)  # head h uses kv h // group
        att = np.where(future, -np.inf, q @ k.transpose(0, 2, 1) / np.sqrt(D))  # [H, T, T]
        y = (softmax(att) @ v).transpose(1, 0, 2).reshape(T, E)
        x = x + y @ W[p + "self_attn.o_proj.weight"].T
        h = rms_norm(x, W[p + "post_attention_layernorm.weight"], eps)
        m = silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)
        x = x + m @ W[p + "mlp.down_proj.weight"].T
    x = rms_norm(x, W["model.norm.weight"], eps)
    return x @ W["model.embed_tokens.weight"].T  # tied LM head


# ---------------------------------------------------------------- tiny model fixture

TINY = {
    "model_type": "llama",
    "hidden_act": "silu",
    "hidden_size": 24,
    "intermediate_size": 20,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "vocab_size": 37,
}


def tiny_tensors(seed):
    cfg, E, F = TINY, TINY["hidden_size"], TINY["intermediate_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    D = E // H
    rng = np.random.default_rng(seed)

    def u(shape, lo, hi):
        return rng.uniform(lo, hi, size=shape).astype(np.float32)

    def lin(n_in, n_out, gain=1.0):
        s = gain / np.sqrt(n_in)
        return u((n_out, n_in), -s, s)  # Hugging Face layout [out, in]

    # With tied embeddings a random model's residual stream is mostly the token's own
    # embedding, so it predicts the last token again and again. Small embeddings and larger
    # output projections let the blocks steer the prediction.
    tensors = [("model.embed_tokens.weight", u((cfg["vocab_size"], E), -0.3, 0.3))]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        tensors += [
            (p + "input_layernorm.weight", u((E,), 0.8, 1.2)),
            (p + "self_attn.q_proj.weight", lin(E, H * D)),
            (p + "self_attn.k_proj.weight", lin(E, KV * D)),
            (p + "self_attn.v_proj.weight", lin(E, KV * D)),
            (p + "self_attn.o_proj.weight", lin(E, E, 3.0)),
            (p + "post_attention_layernorm.weight", u((E,), 0.8, 1.2)),
            (p + "mlp.gate_proj.weight", lin(E, F)),
            (p + "mlp.up_proj.weight", lin(E, F)),
            (p + "mlp.down_proj.weight", lin(F, E, 3.0)),
        ]
    tensors.append(("model.norm.weight", u((E,), 0.8, 1.2)))
    return tensors


def tiny():
    """A 2-layer model with 6 query heads sharing 2 key/value heads (d = 4), seeded random
    weights stored as BF16 like the real checkpoint, plus its logits and a greedy continuation.
    The seed is the first from 2027 whose continuation has at least 4 distinct tokens and a
    top-2 gap of at least 0.05 at every step: a model stuck repeating one token would let a
    broken implementation pass, and a near tie could flip in float32."""
    cfg = TINY
    out = FIXTURES / "tiny_llama"
    out.mkdir(parents=True, exist_ok=True)
    ids = [5, 0, 36, 17, 17, 2, 30, 11, 8, 23]  # includes id 0, the last id, and a repeat
    prompt = ids[:4]
    n = cfg["max_position_embeddings"] - len(prompt)  # fill the whole context
    for seed in range(2027, 2127):
        write_bf16_safetensors(out / "model.safetensors", tiny_tensors(seed))
        W = read_safetensors(out / "model.safetensors")  # compute from exactly what was stored
        cont, gap = greedy_from(lambda ids: forward(W, cfg, ids)[-1], prompt, n)
        if len(set(cont)) >= 4 and gap >= 0.05:
            break
    else:
        sys.exit("no seed in 2027..2126 gives a varied continuation")
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    logits = forward(W, cfg, ids)
    save_logits(out / "logits.f32", logits)
    golden = {"seed": seed, "ids": ids, "logits": "logits.f32", "shape": list(logits.shape),
              "greedy_prompt": prompt, "greedy": cont, "greedy_min_gap": gap}
    (out / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)} (seed {seed}): logits {logits.shape}, greedy {cont} "
          f"(min top-2 gap {gap:.3g})")


# ---------------------------------------------------------------- SmolLM2-135M golden

PROMPTS = [
    ("dog", "I enjoy walking with my cute dog", 16),
    ("fox", "The quick brown fox jumps over the lazy dog.", 12),
    ("mixed", "In 1998, the café's ½-price menu — naïve  spacing\n\nand an emoji 🙂!", 12),
    ("chat", "<|im_start|>user\nWhat is 12 + 30?<|im_end|>\n<|im_start|>assistant\n", 12),
]
SMOL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"


def smollm2():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(SMOL_DIR / "tokenizer.json"))
    cfg = json.loads((SMOL_DIR / "config.json").read_text())
    W = read_safetensors(SMOL_DIR / "model.safetensors")
    out = SMOL_DIR / "golden"
    out.mkdir(exist_ok=True)
    prompts = []
    for name, prompt, n in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        logits = forward(W, cfg, ids)
        save_logits(out / f"{name}.logits.f32", logits)
        cont, gap = greedy_from(lambda ids: forward(W, cfg, ids)[-1], ids, n)
        text = tok.decode(cont)
        prompts.append({"name": name, "text": prompt, "ids": ids, "logits": f"{name}.logits.f32",
                        "shape": list(logits.shape), "greedy": cont, "greedy_text": text,
                        "greedy_min_gap": gap})
        print(f"{name}: {len(ids)} tokens -> {text!r} (min top-2 gap {gap:.3g})")
    manifest = {"model": "HuggingFaceTB/SmolLM2-135M", "sha256": SMOL_SHA256,
                "note": "logits computed in float64, stored as little-endian float32, row-major [T, vocab]",
                "prompts": prompts}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cmds = {"tokenizer": tokenizer_cases, "tiny": tiny, "smollm2": smollm2}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: {sys.argv[0]} {' | '.join(cmds)}")
    cmds[sys.argv[1]]()
