//! GPT-2's byte-level BPE tokenizer, written out by hand (D11).
//!
//! Encoding a string takes three steps:
//!
//! 1. **Pre-split** the text with GPT-2's regex into words: contractions (`'s`, `'ll`), runs of
//!    letters, runs of digits, runs of other symbols (each optionally led by one space), and
//!    whitespace. BPE never merges across these boundaries, so " dog" and "dog!" can't become
//!    one token.
//! 2. **Bytes to symbols.** Each word's UTF-8 bytes become one symbol per byte. GPT-2's vocab
//!    stores tokens as text, so every byte gets a printable stand-in character
//!    ([`byte_to_char`]); a leading space shows up in `vocab.json` as `Ġ`.
//! 3. **Merge.** Repeatedly find the adjacent pair with the lowest rank in `merges.txt` (rank =
//!    line number, i.e. how early training learned it), and merge every occurrence of it, left
//!    to right. Stop when no adjacent pair has a rank. The remaining symbols are tokens.
//!
//! Because the base alphabet is all 256 bytes, any string encodes: there is no unknown token.
//! Decoding concatenates the tokens' bytes and reads them as UTF-8.
//!
//! Llama-family models of the same lineage (SmolLM2) add two steps, read from `tokenizer.json`
//! by [`Tokenizer::load_hf`] (D75): special tokens such as `<|im_start|>` are cut out of the
//! text whole before anything else, and every digit becomes its own piece before step 1.

use std::collections::HashMap;
use std::path::Path;

use fancy_regex::Regex;
use serde_json::Value;

use crate::error::{Error, Result};

/// GPT-2's pre-tokenizer pattern, verbatim from OpenAI's `encoder.py`. Note it is
/// case-sensitive: `'S` in "IT'S" is not a contraction match and splits differently.
///
/// The `\s+(?!\S)` alternative matches a run of whitespace *except its last character* when
/// a word follows, so the final space can attach to that word (" world" is one token). Rust's
/// `regex` crate has no look-ahead, hence `fancy-regex`.
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    /// Token id -> its bytes.
    tokens: Vec<Vec<u8>>,
    /// Token text (in the byte-stand-in alphabet) -> id, as in `vocab.json`.
    ids: HashMap<String, u32>,
    /// Single byte -> its one-symbol token id. All 256 exist in GPT-2's vocab; SmolLM2's lacks
    /// 21 (rare controls and bytes valid UTF-8 almost never uses), and such a byte is dropped
    /// from the input, as Hugging Face does without an unknown token (D75).
    byte_ids: [Option<u32>; 256],
    /// `(left, right)` -> `(rank, merged)`: lower rank merges first.
    merges: HashMap<(u32, u32), (u32, u32)>,
    pattern: Regex,
    /// Split every digit into its own piece before the regex (D75; off for GPT-2).
    split_digits: bool,
    /// Special tokens matched whole in the input (D75; none for GPT-2, D11).
    special: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Load `vocab.json` and `merges.txt` from a Hugging Face model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))
        };
        Self::from_strings(&read("vocab.json")?, &read("merges.txt")?)
            .map_err(|e| Error::Format(format!("{}: {e}", dir.display())))
    }

    /// A tokenizer from the two files' contents. Every byte must have its own token (GPT-2's
    /// promise that any string encodes).
    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> std::result::Result<Self, String> {
        let tok = Self::parse(vocab_json, merges_txt)?;
        if let Some(b) = (0..=255u8).find(|&b| tok.byte_ids[b as usize].is_none()) {
            return Err(format!("vocab has no token for byte {b:#04x}"));
        }
        Ok(tok)
    }

    fn parse(vocab_json: &str, merges_txt: &str) -> std::result::Result<Self, String> {
        let char_to_byte: HashMap<char, u8> = (0..=255u8).map(|b| (byte_to_char(b), b)).collect();

        let vocab: Value =
            serde_json::from_str(vocab_json).map_err(|e| format!("vocab.json: {e}"))?;
        let Value::Object(vocab) = vocab else {
            return Err("vocab.json is not an object".into());
        };
        let n = vocab.len();
        let mut tokens = vec![None; n];
        let mut ids = HashMap::with_capacity(vocab.len());
        for (text, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("id of `{text}` is not an integer"))?;
            let slot = tokens
                .get_mut(id as usize)
                .ok_or_else(|| format!("id {id} of `{text}`: ids must be 0..{n}"))?;
            let bytes = text
                .chars()
                .map(|c| char_to_byte.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| {
                    format!("token `{text}` has a character outside the byte alphabet")
                })?;
            if slot.replace(bytes).is_some() {
                return Err(format!("id {id} is used twice"));
            }
            ids.insert(text, id as u32);
        }
        // Every slot is filled: n entries, ids all < n, no id used twice.
        let tokens: Vec<Vec<u8>> = tokens.into_iter().map(Option::unwrap).collect();

        let mut byte_ids = [None; 256];
        for b in 0..=255u8 {
            byte_ids[b as usize] = ids.get(&byte_to_char(b).to_string()).copied();
        }

        let mut merges = HashMap::new();
        let lines = merges_txt.lines().filter(|l| !l.starts_with("#version"));
        for (rank, line) in lines.filter(|l| !l.is_empty()).enumerate() {
            let (a, b) = line
                .split_once(' ')
                .ok_or_else(|| format!("merge `{line}` is not two tokens"))?;
            let id = |t: &str| {
                ids.get(t)
                    .copied()
                    .ok_or_else(|| format!("merge `{line}`: `{t}` is not in the vocab"))
            };
            let pair = (id(a)?, id(b)?);
            let merged = id(&format!("{a}{b}"))?;
            if merges.insert(pair, (rank as u32, merged)).is_some() {
                return Err(format!("merge `{line}` appears twice"));
            }
        }

        Ok(Tokenizer {
            tokens,
            ids,
            byte_ids,
            merges,
            pattern: Regex::new(PATTERN).expect("PATTERN is a valid regex"),
            split_digits: false,
            special: Vec::new(),
        })
    }

    /// Load a byte-level BPE tokenizer with the extras its `tokenizer.json` asks for (D75):
    /// the digit split and the special tokens. Only the shapes ember implements are accepted:
    /// no normalizer, and a pre-tokenizer that is GPT-2's `ByteLevel` alone or preceded by
    /// `Digits` with `individual_digits`. Anything else is an error, not a silent mismatch.
    pub fn load_hf(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))
        };
        let in_dir = |e: String| Error::Format(format!("{}: {e}", dir.display()));
        let mut tok = Self::parse(&read("vocab.json")?, &read("merges.txt")?).map_err(in_dir)?;
        tok.apply_hf(&read("tokenizer.json")?)
            .map_err(|e| in_dir(format!("tokenizer.json: {e}")))?;
        Ok(tok)
    }

    fn apply_hf(&mut self, tokenizer_json: &str) -> std::result::Result<(), String> {
        let t: Value = serde_json::from_str(tokenizer_json).map_err(|e| e.to_string())?;
        if !t["normalizer"].is_null() {
            return Err(format!("unsupported normalizer {}", t["normalizer"]));
        }
        let byte_level = |p: &Value| {
            p["type"] == "ByteLevel" && p["add_prefix_space"] == false && p["use_regex"] != false
        };
        let pre = &t["pre_tokenizer"];
        self.split_digits = if byte_level(pre) {
            false
        } else if pre["type"] == "Sequence"
            && pre["pretokenizers"].as_array().is_some_and(|ps| {
                ps.len() == 2
                    && ps[0]["type"] == "Digits"
                    && ps[0]["individual_digits"] == true
                    && byte_level(&ps[1])
            })
        {
            true
        } else {
            return Err(format!("unsupported pre-tokenizer {pre}"));
        };
        // After BPE: a post-processor that adds tokens (a BOS, a template) would change the
        // ids, and a decoder other than byte-level would change the text. ByteLevel's only
        // touches offsets, which ember doesn't report.
        for key in ["post_processor", "decoder"] {
            if !(t[key].is_null() || t[key]["type"] == "ByteLevel") {
                return Err(format!("unsupported {key} {}", t[key]));
            }
        }
        // Plain BPE as `bpe` implements it. GPT-2's older file has no "type".
        let model = &t["model"];
        let empty = |v: &Value| v.is_null() || *v == "";
        if !(model["type"] == "BPE" || model["type"].is_null())
            || !model["unk_token"].is_null()
            || model["byte_fallback"] == true
            || model["ignore_merges"] == true
            || !empty(&model["continuing_subword_prefix"])
            || !empty(&model["end_of_word_suffix"])
        {
            return Err("model is not plain byte-level BPE".into());
        }
        let mut special = Vec::new();
        for a in t["added_tokens"].as_array().into_iter().flatten() {
            let (Some(text), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) else {
                return Err(format!("bad added token {a}"));
            };
            // `normalized` only matters with a normalizer, and there is none (checked above).
            if a["special"] != true
                || a["lstrip"] == true
                || a["rstrip"] == true
                || a["single_word"] == true
            {
                return Err(format!("unsupported added token {a}"));
            }
            if self.token_id(text) != Some(id as u32) {
                return Err(format!("added token `{text}` is not id {id} in vocab.json"));
            }
            special.push((text.to_string(), id as u32));
        }
        // No ordering needed: `encode` takes the earliest match, and the longest one there.
        self.special = special;
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Id of a token given in `vocab.json`'s spelling, e.g. `"<|endoftext|>"` (50256 in GPT-2).
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// Encode text. With [`Tokenizer::load`] (GPT-2) this is plain text: `<|endoftext|>`
    /// written in the input becomes ordinary tokens, as in OpenAI's original encoder (D11).
    /// With [`Tokenizer::load_hf`], special tokens in the input map to their ids, as Hugging
    /// Face does (D75).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in `rest`; at one position, the longest.
            let next = self
                .special
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            let (plain, special) = match next {
                Some((at, len, id)) => (&rest[..at], Some((id, at + len))),
                None => (rest, None),
            };
            for word in self.split(plain)? {
                out.extend(self.bpe(word.as_bytes()));
            }
            match special {
                Some((id, end)) => {
                    out.push(id);
                    rest = &rest[end..];
                }
                None => break,
            }
        }
        Ok(out)
    }

    /// Step 1: the pre-split (each digit on its own first, if the tokenizer asks for it, then
    /// GPT-2's regex within each piece). Public so `ember tokenize` can show it.
    pub fn split<'t>(&self, text: &'t str) -> Result<Vec<&'t str>> {
        let mut words = Vec::new();
        for piece in self.digit_pieces(text) {
            for m in self.pattern.find_iter(piece) {
                let m = m.map_err(|e| Error::Input(format!("pre-tokenizer regex failed: {e}")))?;
                words.push(m.as_str());
            }
        }
        Ok(words)
    }

    /// `text` cut so that every numeric character (`char::is_numeric`, as Hugging Face's
    /// `Digits` uses) stands alone; the whole text when the digit split is off.
    fn digit_pieces<'t>(&self, text: &'t str) -> Vec<&'t str> {
        if !self.split_digits {
            return vec![text];
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if c.is_numeric() {
                if start < i {
                    pieces.push(&text[start..i]);
                }
                pieces.push(&text[i..i + c.len_utf8()]);
                start = i + c.len_utf8();
            }
        }
        if start < text.len() {
            pieces.push(&text[start..]);
        }
        pieces
    }

    /// Steps 2 and 3 for one pre-split word. Bytes without a token are dropped before merging.
    fn bpe(&self, word: &[u8]) -> Vec<u32> {
        let mut symbols: Vec<u32> = word
            .iter()
            .filter_map(|&b| self.byte_ids[b as usize])
            .collect();
        loop {
            // The adjacent pair that was learned earliest. Ties can't happen: ranks are unique.
            let best = symbols
                .windows(2)
                .filter_map(|w| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, _)| (rank, w[0], w[1]))
                })
                .min();
            let Some((_, a, b)) = best else {
                return symbols;
            };
            let merged = self.merges[&(a, b)].1;
            // Merge every occurrence, scanning left to right. "aaa" with merge (a, a) becomes
            // [aa, a]: once a pair is consumed, its right half can't start another pair.
            let mut next = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                    next.push(merged);
                    i += 2;
                } else {
                    next.push(symbols[i]);
                    i += 1;
                }
            }
            symbols = next;
        }
    }

    /// The raw bytes of one token. A token can end in the middle of a multi-byte UTF-8
    /// character (emoji are often split across tokens), so this is bytes, not `&str`.
    pub fn token_bytes(&self, id: u32) -> Result<&[u8]> {
        self.tokens
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Input(format!("token id {id} is outside the vocab")))
    }

    /// Concatenate the tokens' bytes and decode as UTF-8, replacing invalid sequences with
    /// U+FFFD (a generated sequence can stop mid-character).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.token_bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// GPT-2's byte -> printable-character map (`bytes_to_unicode` in `encoder.py`).
///
/// The 188 bytes that are already visible, non-space Latin-1 characters (`!`..=`~`,
/// `¡`..=`¬`, `®`..=`ÿ`) stand for themselves. The other 68 (control characters, space, DEL,
/// NBSP, soft hyphen) are given code points 256, 257, ... in byte order. So space (0x20, the
/// 33rd such byte) becomes U+0120 `Ġ` and newline (0x0A) becomes U+010A `Ċ`.
pub fn byte_to_char(b: u8) -> char {
    let visible = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    if visible(b) {
        return b as char;
    }
    let n = (0..b).filter(|&x| !visible(x)).count() as u32;
    char::from_u32(256 + n).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet() {
        assert_eq!(byte_to_char(b'a'), 'a');
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
        assert_eq!(byte_to_char(0), '\u{100}');
        assert_eq!(byte_to_char(0xAD), '\u{143}'); // soft hyphen, the 68th and last
        let all: std::collections::HashSet<char> = (0..=255).map(byte_to_char).collect();
        assert_eq!(all.len(), 256, "the map must be one-to-one");
    }

    /// A toy vocab: the 256 byte tokens, then merges for "ab", "abc" and " ab".
    fn toy() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(byte_to_char(b).to_string(), b.into());
        }
        for (i, t) in ["ab", "abc", "Ġab", "aa"].iter().enumerate() {
            vocab.insert(t.to_string(), (256 + i).into());
        }
        // Ranks: "a b" 0, "ab c" 1, "Ġ ab" 2, "a a" 3.
        let merges = "#version: 0.2\na b\nab c\nĠ ab\na a\n";
        Tokenizer::from_strings(&Value::Object(vocab).to_string(), merges).unwrap()
    }

    #[test]
    fn merges_by_rank_not_by_position() {
        let t = toy();
        assert_eq!(t.encode("abc").unwrap(), [257]); // a b -> ab, then ab c -> abc
        // " ab": the pair (a, b) has rank 0 and beats (Ġ, a) which has no rank; then (Ġ, ab).
        assert_eq!(t.encode(" ab").unwrap(), [258]);
        // "aab": (a, a) is rank 3 and (a, b) rank 0, so the second a pairs with b: [a, ab].
        assert_eq!(t.encode("aab").unwrap(), [97, 256]);
        assert_eq!(t.encode("ba").unwrap(), [98, 97]);
    }

    #[test]
    fn merges_every_occurrence_left_to_right() {
        let t = toy();
        assert_eq!(t.encode("abab").unwrap(), [256, 256]);
        assert_eq!(t.encode("aaa").unwrap(), [259, 97]);
        assert_eq!(t.encode("aaaa").unwrap(), [259, 259]);
    }

    #[test]
    fn pre_split_follows_gpt2() {
        let t = toy();
        let cases: [(&str, &[&str]); 7] = [
            ("Hello world", &["Hello", " world"]),
            ("a   b", &["a", "  ", " b"]),
            ("I'm here", &["I", "'m", " here"]),
            ("IT'S", &["IT", "'", "S"]), // case-sensitive: ' is a symbol run, S a letter run
            ("x=42!", &["x", "=", "42", "!"]),
            ("end  \n", &["end", "  \n"]),
            ("café 🙂", &["café", " 🙂"]),
        ];
        for (text, want) in cases {
            assert_eq!(t.split(text).unwrap(), want, "{text:?}");
        }
    }

    #[test]
    fn round_trips_arbitrary_bytes() {
        let t = toy();
        for s in [
            "",
            "plain",
            "  spaces  ",
            "tab\tnew\nline",
            "ünïcödé 🙂 日本",
            "\0\x7f",
        ] {
            assert_eq!(t.decode(&t.encode(s).unwrap()).unwrap(), s);
        }
    }

    #[test]
    fn decode_tolerates_split_characters() {
        let t = toy();
        let ids = t.encode("é").unwrap(); // two bytes, two byte-tokens here
        assert_eq!(ids.len(), 2);
        assert_eq!(t.decode(&ids[..1]).unwrap(), "\u{FFFD}");
        assert!(t.decode(&[100_000]).is_err());
    }

    /// `toy()` with SmolLM2's extras: a digit split and the special tokens `<s>` (id 260) and
    /// `<s>x` (261), one a prefix of the other.
    fn toy_hf() -> Tokenizer {
        let mut vocab: serde_json::Map<String, Value> = (0..=255u8)
            .map(|b| (byte_to_char(b).to_string(), b.into()))
            .collect();
        for (i, t) in ["ab", "abc", "Ġab", "aa", "<s>", "<s>x"].iter().enumerate() {
            vocab.insert(t.to_string(), (256 + i).into());
        }
        let mut t =
            Tokenizer::parse(&Value::Object(vocab).to_string(), "a b\nab c\nĠ ab\na a\n").unwrap();
        t.apply_hf(&hf_json(
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "Digits", "individual_digits": true},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}]}"#,
        ))
        .unwrap();
        t
    }

    fn hf_json(pre_tokenizer: &str) -> String {
        format!(
            r#"{{"normalizer": null, "pre_tokenizer": {pre_tokenizer},
                "model": {{"type": "BPE", "unk_token": null}},
                "added_tokens": [
                  {{"id": 260, "content": "<s>", "special": true}},
                  {{"id": 261, "content": "<s>x", "special": true}}]}}"#
        )
    }

    #[test]
    fn digits_stand_alone_before_the_regex() {
        let t = toy_hf();
        let cases: [(&str, &[&str]); 4] = [
            ("x=42!", &["x", "=", "4", "2", "!"]),
            (" 42 ", &[" ", "4", "2", " "]), // the space can't lead a digit any more
            ("ab12cd", &["ab", "1", "2", "cd"]),
            ("٣²", &["٣", "²"]), // char::is_numeric: other scripts and superscripts too
        ];
        for (text, want) in cases {
            assert_eq!(t.split(text).unwrap(), want, "{text:?}");
        }
        assert_eq!(toy().split("x=42!").unwrap(), ["x", "=", "42", "!"]); // GPT-2: off
    }

    #[test]
    fn special_tokens_match_whole_earliest_then_longest() {
        let t = toy_hf();
        assert_eq!(t.encode("<s>").unwrap(), [260]);
        assert_eq!(t.encode("<s>x").unwrap(), [261]); // not <s> then x
        assert_eq!(t.encode("ab<s>ab").unwrap(), [256, 260, 256]);
        assert_eq!(t.encode("<s><s>x").unwrap(), [260, 261]);
        // A near miss is plain text, byte by byte here.
        assert_eq!(t.encode("<s").unwrap(), [u32::from(b'<'), u32::from(b's')]);
        // Without load_hf there are no special tokens (D11).
        assert_ne!(toy().encode("<s>").unwrap(), [260]);
    }

    #[test]
    fn bytes_without_a_token_are_dropped_before_merging() {
        // A vocab without byte 0x04 (ids stay dense: "!?" takes id 4). "!\x04?" is one symbol
        // run for the regex; dropping 0x04 leaves "!?", which then merges.
        let mut vocab: serde_json::Map<String, Value> = (0..=255u8)
            .filter(|&b| b != 4)
            .map(|b| (byte_to_char(b).to_string(), b.into()))
            .collect();
        vocab.insert("!?".into(), 4.into());
        let vocab = Value::Object(vocab).to_string();
        let e = Tokenizer::from_strings(&vocab, "! ?\n").err().unwrap();
        assert!(e.contains("no token for byte 0x04"), "{e}"); // GPT-2's loader insists
        let t = Tokenizer::parse(&vocab, "! ?\n").unwrap();
        assert_eq!(t.encode("!\x04?").unwrap(), [4]);
        assert_eq!(t.encode("\x04").unwrap(), [0u32; 0]);
        // Across words nothing merges: "a", "\x04" and "b" are three regex matches.
        assert_eq!(
            t.encode("a\x04b").unwrap(),
            [u32::from(b'a'), u32::from(b'b')]
        );
    }

    #[test]
    fn rejects_tokenizer_json_it_cannot_follow() {
        let byte_level = r#"{"type": "ByteLevel", "add_prefix_space": false}"#;
        let bad = [
            hf_json(byte_level)
                .replace(r#""normalizer": null"#, r#""normalizer": {"type": "NFC"}"#),
            hf_json(r#"{"type": "ByteLevel", "add_prefix_space": true}"#),
            hf_json(r#"{"type": "Whitespace"}"#),
            hf_json(&format!(
                r#"{{"type": "Sequence", "pretokenizers": [
                    {{"type": "Digits", "individual_digits": false}}, {byte_level}]}}"#
            )),
            hf_json(byte_level).replace(r#""unk_token": null"#, r#""unk_token": "<unk>""#),
            hf_json(byte_level).replace(r#""type": "BPE""#, r#""type": "WordPiece""#),
            hf_json(byte_level).replace(r#""id": 260"#, r#""id": 7"#), // not its vocab id
            hf_json(byte_level).replace(r#""special": true}]"#, r#""special": false}]"#),
            // Llama 3 style: a template that puts a BOS token in front.
            hf_json(byte_level).replace(
                r#""normalizer": null"#,
                r#""normalizer": null, "post_processor": {"type": "TemplateProcessing"}"#,
            ),
            hf_json(byte_level).replace(
                r#""normalizer": null"#,
                r#""normalizer": null, "decoder": {"type": "Metaspace"}"#,
            ),
        ];
        for json in bad {
            assert_ne!(json, hf_json(byte_level), "the case must change the file");
            let mut t = toy_hf();
            assert!(t.apply_hf(&json).is_err(), "accepted {json}");
        }
        assert!(toy_hf().apply_hf(&hf_json(byte_level)).is_ok());
        // SmolLM2's own: no post-processor, a ByteLevel decoder.
        let smol = hf_json(byte_level).replace(
            r#""normalizer": null"#,
            r#""normalizer": null, "post_processor": null, "decoder": {"type": "ByteLevel"}"#,
        );
        assert!(toy_hf().apply_hf(&smol).is_ok());
    }

    #[test]
    fn rejects_inconsistent_files() {
        let vocab = |extra: &str| {
            let mut v: Vec<String> = (0..=255u8)
                .map(|b| format!("{:?}: {b}", byte_to_char(b).to_string()))
                .collect();
            if !extra.is_empty() {
                v.push(extra.to_string());
            }
            format!("{{{}}}", v.join(","))
        };
        let bad = [
            (vocab(r#""ab": 300"#), "", "ids must be"),
            (vocab(r#""ab": 7"#), "", "used twice"),
            (vocab(""), "a b\n", "`ab` is not in the vocab"),
            (vocab(r#""ab": 256"#), "a b\na b\n", "appears twice"),
            (vocab(r#""ab": 256"#), "ab\n", "not two tokens"),
            (vocab(r#""a b": 256"#), "", "outside the byte alphabet"),
            (
                r#"{"a": 0}"#.to_string(),
                "",
                "vocab has no token for byte 0x00",
            ),
        ];
        for (v, m, want) in bad {
            let e = Tokenizer::from_strings(&v, m).err().expect("must fail");
            assert!(e.contains(want), "got `{e}`, wanted `{want}`");
        }
    }
}
