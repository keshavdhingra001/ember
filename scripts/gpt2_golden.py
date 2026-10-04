"""Reference outputs for ember's CPU GPT-2 (D12). Run once; the Rust tests compare against them.

Everything here is computed in float64 with numpy, written in the vectorized style (matrix
products over whole sequences, all heads at once) so it shares as little structure as possible
with the Rust reference's explicit loops. Token ids come from Hugging Face's `tokenizers`, an
implementation independent of ember's hand-written BPE.

Setup (from the repository root; PyPI is slow here, the two wheels are about 20 MB):
    python3 -m venv target/tmp/venv
    target/tmp/venv/bin/pip install numpy tokenizers

Usage:
    target/tmp/venv/bin/python scripts/gpt2_golden.py tokenizer  # tests/fixtures/gpt2_tokenizer_cases.json
    target/tmp/venv/bin/python scripts/gpt2_golden.py tiny       # tests/fixtures/tiny_gpt2/
    target/tmp/venv/bin/python scripts/gpt2_golden.py gpt2       # data/gpt2/golden/ (needs the weights)
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GPT2_DIR = ROOT / "data" / "gpt2"
FIXTURES = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------- safetensors

def read_safetensors(path):
    """name -> float64 array. Skips the `attn.bias` causal-mask buffers (constants, not weights)."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    data = memoryview(raw)[8 + n :]
    out = {}
    for name, e in header.items():
        if name == "__metadata__" or name.endswith(".attn.bias"):
            continue
        assert e["dtype"] == "F32", (name, e["dtype"])
        begin, end = e["data_offsets"]
        a = np.frombuffer(data[begin:end], dtype="<f4").reshape(e["shape"])
        out[name] = a.astype(np.float64)
    return out


def write_safetensors(path, tensors):
    """tensors: list of (name, float32 array), laid out in the order given."""
    header, chunks, offset = {}, [], 0
    for name, a in tensors:
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        chunks.append(b)
        offset += len(b)
    header["__metadata__"] = {"format": "pt", "source": "scripts/gpt2_golden.py"}
    h = json.dumps(header, separators=(",", ":")).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))


# ---------------------------------------------------------------- the model

def layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)  # biased variance, as in PyTorch
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    """gelu_new: the tanh approximation GPT-2 was trained with (not the exact erf form)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def forward(W, cfg, ids):
    """Logits [T, vocab] for token ids [T]. W holds float64 arrays in the checkpoint's layout."""
    T, E, H = len(ids), cfg["n_embd"], cfg["n_head"]
    D, eps = E // H, cfg["layer_norm_epsilon"]
    assert T <= cfg["n_positions"]
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    for i in range(cfg["n_layer"]):
        p = f"h.{i}."
        h = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"], eps)
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]  # Conv1D: x @ [in, out]
        q, k, v = (a.reshape(T, H, D).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=1))
        att = np.where(future, -np.inf, q @ k.transpose(0, 2, 1) / np.sqrt(D))  # [H, T, T]
        y = (softmax(att) @ v).transpose(1, 0, 2).reshape(T, E)
        x = x + y @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"], eps)
        m = gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + m @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"], eps)
    return x @ W["wte.weight"].T  # tied LM head


def greedy(W, cfg, ids, n):
    """n greedy tokens (full recompute each step). Also returns the smallest gap between the
    best and second-best logit over the steps: if it is tiny, float32 could pick differently."""
    ids, min_gap = list(ids), np.inf
    for _ in range(n):
        last = forward(W, cfg, ids)[-1]
        top2 = np.sort(last)[-2:]
        min_gap = min(min_gap, top2[1] - top2[0])
        ids.append(int(np.argmax(last)))  # argmax returns the first index on ties
    return ids[-n:], float(min_gap)


def save_logits(path, logits):
    np.ascontiguousarray(logits, dtype="<f4").tofile(path)


# ---------------------------------------------------------------- tokenizer cases

TOKENIZER_CASES = [
    "",
    "Hello, my dog is cute",
    "The quick brown fox jumps over the lazy dog.",
    "a   b",
    "  leading spaces and trailing   ",
    "multiple\n\nnewlines\n",
    "Tabs\tand\r\nCRLF \t \n mixed",
    "contractions: I'm you're they've we'll he'd it's don't O'Neil's",
    "uppercase contractions don't merge: I'M YOU'RE IT'S",
    "numbers 1234567890 3.14159 -42 1e10 1,000,000",
    "unicode: café naïve résumé Zürich Ελληνικά Кириллица",
    "CJK: 日本語のテキスト 中文 한국어",
    "emoji: 🙂👍🏽 🇮🇳 👨‍👩‍👧",
    "Devanagari combining marks: नमस्ते दुनिया",
    "math: ∑ x² ≤ ½ — “quotes” ‘single’ …",
    'code: fn main() { println!("hi"); } // ok?!',
    "odd spaces: nbsp　ideographic thin",
    "controls: \x0b\x0c vt ff \x1c\x1d\x1e\x1f seps \x85 nel \x00 nul",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "!!!??? ... --- ___ *** ### @@@",
    "trailing space ",
    "\n",
    "   ",
]


def tokenizer_cases():
    from tokenizers import Tokenizer, __version__

    tok = Tokenizer.from_file(str(GPT2_DIR / "tokenizer.json"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TOKENIZER_CASES]
    out = FIXTURES / "gpt2_tokenizer_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"source": f"huggingface tokenizers {__version__}, openai-community/gpt2 tokenizer.json", "cases": cases}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- tiny model fixture

TINY = {
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-05,
    "n_embd": 12,
    "n_head": 3,
    "n_layer": 2,
    "n_positions": 16,
    "vocab_size": 37,
}


def tiny():
    """A 2-layer, 3-head, E=12 model with seeded random weights in the checkpoint layout (odd
    sizes so index mix-ups can't cancel out), plus its logits and a greedy continuation."""
    cfg, E = TINY, TINY["n_embd"]
    rng = np.random.default_rng(2026)

    def u(shape, lo, hi):
        return rng.uniform(lo, hi, size=shape).astype(np.float32)

    def lin(n_in, n_out):
        s = 1.0 / np.sqrt(n_in)
        return u((n_in, n_out), -s, s), u((n_out,), -0.1, 0.1)  # Conv1D layout [in, out]

    T = cfg["n_positions"]
    tensors = [("wte.weight", u((cfg["vocab_size"], E), -1, 1)), ("wpe.weight", u((T, E), -0.5, 0.5))]
    for i in range(cfg["n_layer"]):
        p = f"h.{i}."
        for name, (w, b) in [("attn.c_attn", lin(E, 3 * E)), ("attn.c_proj", lin(E, E)),
                             ("mlp.c_fc", lin(E, 4 * E)), ("mlp.c_proj", lin(4 * E, E))]:
            tensors += [(p + name + ".weight", w), (p + name + ".bias", b)]
        for ln in ["ln_1", "ln_2"]:
            tensors += [(p + ln + ".weight", u((E,), 0.8, 1.2)), (p + ln + ".bias", u((E,), -0.1, 0.1))]
        # The real checkpoint carries a causal-mask buffer per layer; the loader must ignore it.
        tensors.append((p + "attn.bias", np.tril(np.ones((1, 1, T, T), dtype=np.float32))))
    tensors += [("ln_f.weight", u((E,), 0.8, 1.2)), ("ln_f.bias", u((E,), -0.1, 0.1))]

    out = FIXTURES / "tiny_gpt2"
    out.mkdir(parents=True, exist_ok=True)
    write_safetensors(out / "model.safetensors", tensors)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")

    W = read_safetensors(out / "model.safetensors")  # compute from exactly what was stored
    ids = [5, 0, 36, 17, 17, 2, 30, 11, 8, 23]  # includes id 0, the last id, and a repeat
    logits = forward(W, cfg, ids)
    save_logits(out / "logits.f32", logits)
    prompt = ids[:4]
    cont, gap = greedy(W, cfg, prompt, T - len(prompt))  # fill the whole context
    golden = {"ids": ids, "logits": "logits.f32", "shape": list(logits.shape),
              "greedy_prompt": prompt, "greedy": cont, "greedy_min_gap": gap}
    (out / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: logits {logits.shape}, greedy {cont} (min top-2 gap {gap:.3g})")


# ---------------------------------------------------------------- GPT-2 124M golden

PROMPTS = [
    ("hello", "Hello, my dog is cute", 12),
    ("fox", "The quick brown fox jumps over the lazy dog.", 12),
    ("mixed", "In 1998, the café's ½-price menu — naïve  spacing\n\nand an emoji 🙂!", 12),
]
# Hugging Face's "How to generate text" blog post (2020) shows greedy GPT-2 124M continuing this
# prompt; the first sentence is pinned here as a check that is independent of this script.
PUBLISHED_PROMPT = "I enjoy walking with my cute dog"
PUBLISHED_START = ", but I'm not sure if I'll ever be able to walk with my dog."


def gpt2():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(GPT2_DIR / "tokenizer.json"))
    cfg = json.loads((GPT2_DIR / "config.json").read_text())
    W = read_safetensors(GPT2_DIR / "model.safetensors")
    out = GPT2_DIR / "golden"
    out.mkdir(exist_ok=True)

    def text(ids):
        return tok.decode(ids)

    prompts = []
    for name, prompt, n in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        logits = forward(W, cfg, ids)
        save_logits(out / f"{name}.logits.f32", logits)
        cont, gap = greedy(W, cfg, ids, n)
        prompts.append({"name": name, "text": prompt, "ids": ids, "logits": f"{name}.logits.f32",
                        "shape": list(logits.shape), "greedy": cont, "greedy_text": text(cont),
                        "greedy_min_gap": gap})
        print(f"{name}: {len(ids)} tokens -> {text(cont)!r} (min top-2 gap {gap:.3g})")

    ids = tok.encode(PUBLISHED_PROMPT, add_special_tokens=False).ids
    n = len(tok.encode(PUBLISHED_START, add_special_tokens=False).ids)
    cont, gap = greedy(W, cfg, ids, n)
    ok = text(cont) == PUBLISHED_START
    print(f"published check: {text(cont)!r} {'MATCHES' if ok else 'DIFFERS FROM'} the blog post")
    if not ok:
        sys.exit("the numpy forward pass disagrees with the published output: fix it before using these goldens")
    published = {"text": PUBLISHED_PROMPT, "ids": ids, "greedy": cont, "greedy_text": text(cont),
                 "greedy_min_gap": gap}

    manifest = {"model": "openai-community/gpt2",
                "sha256": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
                "note": "logits computed in float64, stored as little-endian float32, row-major [T, vocab]",
                "prompts": prompts, "published": published}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cmds = {"tokenizer": tokenizer_cases, "tiny": tiny, "gpt2": gpt2}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: gpt2_golden.py [{' | '.join(cmds)}]")
    cmds[sys.argv[1]]()
