//! ember's BPE against Hugging Face's `tokenizers` on tricky strings (D11). The expected ids are
//! committed in tests/fixtures/ (from scripts/gpt2_golden.py); the vocab comes from data/gpt2/.

mod common;

use ember::Tokenizer;
use serde_json::Value;

#[test]
fn matches_hugging_face() {
    let Some(dir) = common::gpt2_dir() else {
        return;
    };
    let tok = Tokenizer::load(&dir).unwrap();
    assert_eq!(tok.vocab_size(), 50257);
    assert_eq!(tok.token_id("<|endoftext|>"), Some(50256));

    let fixture = include_str!("fixtures/gpt2_tokenizer_cases.json");
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
        assert_eq!(tok.decode(&got).unwrap(), text, "round trip");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn endoftext_in_input_is_plain_text() {
    let Some(dir) = common::gpt2_dir() else {
        return;
    };
    let tok = Tokenizer::load(&dir).unwrap();
    let ids = tok.encode("<|endoftext|>").unwrap();
    assert!(ids.len() > 1 && !ids.contains(&50256), "{ids:?}");
    assert_eq!(tok.decode(&[50256]).unwrap(), "<|endoftext|>");
}
