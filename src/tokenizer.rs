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
    /// Single byte -> its one-symbol token id. All 256 exist in GPT-2's vocab.
    byte_ids: [u32; 256],
    /// `(left, right)` -> `(rank, merged)`: lower rank merges first.
    merges: HashMap<(u32, u32), (u32, u32)>,
    pattern: Regex,
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

    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> std::result::Result<Self, String> {
        let char_to_byte: HashMap<char, u8> = (0..=255u8).map(|b| (byte_to_char(b), b)).collect();

        let vocab: Value =
            serde_json::from_str(vocab_json).map_err(|e| format!("vocab.json: {e}"))?;
        let Value::Object(vocab) = vocab else {
            return Err("vocab.json is not an object".into());
        };
        let mut tokens = vec![None; vocab.len()];
        let mut ids = HashMap::with_capacity(vocab.len());
        for (text, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("id of `{text}` is not an integer"))?;
            let slot = tokens
                .get_mut(id as usize)
                .ok_or_else(|| format!("id {id} of `{text}`: ids must be 0..{}", ids.len()))?;
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

        let mut byte_ids = [0; 256];
        for b in 0..=255u8 {
            byte_ids[b as usize] = *ids
                .get(&byte_to_char(b).to_string())
                .ok_or_else(|| format!("vocab has no token for byte {b:#04x}"))?;
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
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Id of a token given in `vocab.json`'s spelling, e.g. `"<|endoftext|>"` (50256 in GPT-2).
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// Encode text as plain text: `<|endoftext|>` written in the input becomes ordinary tokens,
    /// as in OpenAI's original encoder (Hugging Face maps it to the special id instead).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for word in self.split(text)? {
            out.extend(self.bpe(word.as_bytes()));
        }
        Ok(out)
    }

    /// Step 1: the regex pre-split. Public so `ember tokenize` can show it.
    pub fn split<'t>(&self, text: &'t str) -> Result<Vec<&'t str>> {
        self.pattern
            .find_iter(text)
            .map(|m| m.map(|m| m.as_str()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Input(format!("pre-tokenizer regex failed: {e}")))
    }

    /// Steps 2 and 3 for one pre-split word.
    fn bpe(&self, word: &[u8]) -> Vec<u32> {
        let mut symbols: Vec<u32> = word.iter().map(|&b| self.byte_ids[b as usize]).collect();
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
