//! ember's BPE with SmolLM2's extras (D75) against Hugging Face's `tokenizers`. The expected
//! ids are committed in tests/fixtures/ (from scripts/llama_golden.py); the vocab comes from
//! data/smollm2-135m/.

mod common;

use ember::Tokenizer;
use ember::tokenizer::byte_to_char;
use serde_json::Value;

#[test]
fn matches_hugging_face() {
    let Some(dir) = common::smollm2_dir() else {
        return;
    };
    let tok = Tokenizer::load_hf(&dir).unwrap();
    assert_eq!(tok.vocab_size(), 49152);
    assert_eq!(tok.token_id("<|endoftext|>"), Some(0));

    let fixture = include_str!("fixtures/smollm2_tokenizer_cases.json");
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text).unwrap();
        if got != want {
            failures.push(format!("{text:?}\n   got  {got:?}\n   want {want:?}"));
        }
        // Round trip, minus the bytes the vocab has no token for (dropped, as Hugging Face does).
        let kept: Vec<u8> = text
            .bytes()
            .filter(|&b| tok.token_id(&byte_to_char(b).to_string()).is_some())
            .collect();
        assert_eq!(
            tok.decode(&got).unwrap(),
            String::from_utf8_lossy(&kept),
            "round trip"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn gpt2_ids_are_unchanged_by_the_hf_loader() {
    // GPT-2's tokenizer.json has no digit split, and D11 keeps `<|endoftext|>` as plain text
    // for `load`; `load_hf` maps it to its id, as Hugging Face does, and nothing else changes.
    let Some(dir) = common::gpt2_dir() else {
        return;
    };
    let (plain, hf) = (
        Tokenizer::load(&dir).unwrap(),
        Tokenizer::load_hf(&dir).unwrap(),
    );
    let text = "numbers 1234 and words, then <|endoftext|> and more";
    let (a, b) = (plain.encode(text).unwrap(), hf.encode(text).unwrap());
    assert!(!a.contains(&50256) && b.contains(&50256));
    let before = "numbers 1234 and words, then ";
    assert_eq!(plain.encode(before).unwrap(), hf.encode(before).unwrap());
}
