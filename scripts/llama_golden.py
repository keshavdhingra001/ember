"""Reference outputs for ember's Llama-family models (D69), in the same style as gpt2_golden.py:
token ids from Hugging Face's `tokenizers`, forward passes in float64 numpy.

Setup: the venv of gpt2_golden.py (numpy, tokenizers).

Usage (from the repository root):
    target/tmp/venv/bin/python scripts/llama_golden.py tokenizer  # tests/fixtures/smollm2_tokenizer_cases.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_golden import TOKENIZER_CASES  # noqa: E402

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


if __name__ == "__main__":
    cmds = {"tokenizer": tokenizer_cases}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: {sys.argv[0]} {' | '.join(cmds)}")
    cmds[sys.argv[1]]()
