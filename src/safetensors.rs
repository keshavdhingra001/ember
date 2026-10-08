//! A minimal safetensors reader (D10).
//!
//! The format, in full:
//!
//! ```text
//! [ u64 little-endian N ][ N bytes of JSON header ][ data section ]
//! header = { "<name>": { "dtype": "F32", "shape": [r, c], "data_offsets": [begin, end] }, ...,
//!            "__metadata__": { "<key>": "<value>", ... } }        (metadata optional)
//! ```
//!
//! Offsets are relative to the start of the data section, `end` is exclusive, and the tensors
//! must tile the data section exactly: no gaps, no overlaps, nothing left over. Data is
//! row-major and little-endian. That is the whole format, so we parse it by hand and validate
//! every claim the header makes before trusting it.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// The format caps the header at 100 MB; a bigger length field means a corrupt or hostile file.
const MAX_HEADER: u64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        use Dtype::*;
        Some(match s {
            "BOOL" => Bool,
            "U8" => U8,
            "I8" => I8,
            "F8_E5M2" => F8E5M2,
            "F8_E4M3" => F8E4M3,
            "I16" => I16,
            "U16" => U16,
            "F16" => F16,
            "BF16" => BF16,
            "I32" => I32,
            "U32" => U32,
            "F32" => F32,
            "F64" => F64,
            "I64" => I64,
            "U64" => U64,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        use Dtype::*;
        match self {
            Bool | U8 | I8 | F8E5M2 | F8E4M3 => 1,
            I16 | U16 | F16 | BF16 => 2,
            I32 | U32 | F32 => 4,
            F64 | I64 | U64 => 8,
        }
    }
}

/// One header entry. `begin..end` is the byte range inside the data section.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

enum Bytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

pub struct SafeTensors {
    bytes: Bytes,
    data_start: usize,
    entries: BTreeMap<String, Entry>,
}

impl SafeTensors {
    /// Memory-map `path` and validate its header. Tensor bytes are only touched (paged in) when
    /// a tensor is read.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: the mapping is read-only, and the only hazard is another process truncating or
        // rewriting the file while it is mapped (reads could then fault or see torn data). Weight
        // files under data/ are written once by the fetch step and never modified in place.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        let (data_start, entries) =
            parse(&mmap).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        Ok(SafeTensors {
            bytes: Bytes::Mapped(mmap),
            data_start,
            entries,
        })
    }

    /// Parse a file already in memory (tests, and files built on the fly).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (data_start, entries) = parse(&bytes).map_err(Error::Format)?;
        Ok(SafeTensors {
            bytes: Bytes::Owned(bytes),
            data_start,
            entries,
        })
    }

    /// Tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    /// Copy one F32 or BF16 tensor out of the file, as f32. Each group is decoded with `from_le_bytes`:
    /// the data section has no alignment guarantee (it starts at 8 + N), so we can't
    /// reinterpret the bytes as `&[f32]` in place.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Format(format!("tensor `{name}` not in file")))?;
        let bytes = &self.bytes.as_slice()[self.data_start + e.begin..self.data_start + e.end];
        // `as_chunks` splits off whole groups; the remainder is empty because the header check
        // guaranteed end - begin == numel * the dtype's size.
        let data = match e.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            // bf16 is the top half of an f32 (same sign and exponent, 7 mantissa bits), so the
            // widening is exact (D68).
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
            other => {
                return Err(Error::Format(format!(
                    "tensor `{name}` is {other:?}; only F32 and BF16 are supported"
                )));
            }
        };
        Tensor::new(&e.shape, data)
    }
}

/// Validate the header and return where the data section starts plus every entry. Errors are
/// plain strings: the caller adds the file name.
fn parse(file: &[u8]) -> std::result::Result<(usize, BTreeMap<String, Entry>), String> {
    let Some(len_bytes) = file.get(..8) else {
        return Err(format!(
            "{} bytes is too short for the 8-byte header length",
            file.len()
        ));
    };
    let n = u64::from_le_bytes(len_bytes.try_into().unwrap());
    if n > MAX_HEADER || 8 + n > file.len() as u64 {
        return Err(format!(
            "header length {n} doesn't fit a {}-byte file",
            file.len()
        ));
    }
    let data_start = 8 + n as usize;
    let header: Value = serde_json::from_slice(&file[8..data_start])
        .map_err(|e| format!("header is not valid JSON: {e}"))?;
    let Value::Object(map) = header else {
        return Err("header is not a JSON object".into());
    };
    let data_len = file.len() - data_start;

    let mut entries = BTreeMap::new();
    for (name, v) in map {
        if name == "__metadata__" {
            continue;
        }
        let entry = parse_entry(&v).map_err(|e| format!("tensor `{name}`: {e}"))?;
        if entry.end > data_len {
            return Err(format!(
                "tensor `{name}` ends at byte {} but the data section has {data_len}",
                entry.end
            ));
        }
        entries.insert(name, entry);
    }

    // The tensors must tile the data section: sorted by start, each begins where the previous
    // ended. A gap means unindexed bytes, a start before the previous end means two tensors
    // share bytes, and a short total means trailing garbage.
    let mut spans: Vec<(usize, usize, &str)> = entries
        .iter()
        .map(|(k, e)| (e.begin, e.end, k.as_str()))
        .collect();
    spans.sort();
    let mut expected = 0;
    for (begin, end, name) in spans {
        if begin != expected {
            let what = if begin > expected { "gap" } else { "overlap" };
            return Err(format!(
                "{what} before tensor `{name}`: it starts at byte {begin}, expected {expected}"
            ));
        }
        expected = end;
    }
    if expected != data_len {
        return Err(format!(
            "tensors cover {expected} bytes but the data section has {data_len}"
        ));
    }
    Ok((data_start, entries))
}

fn parse_entry(v: &Value) -> std::result::Result<Entry, String> {
    let dtype = v
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or("missing dtype")?;
    let dtype = Dtype::parse(dtype).ok_or_else(|| format!("unknown dtype {dtype}"))?;
    let shape = v
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("missing shape")?
        .iter()
        .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad dimension"))
        .collect::<std::result::Result<Vec<usize>, _>>()?;
    let offsets = v
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("missing data_offsets")?;
    let [begin, end] = offsets.as_slice() else {
        return Err("data_offsets must have two elements".into());
    };
    let (Some(begin), Some(end)) = (begin.as_u64(), end.as_u64()) else {
        return Err("data_offsets must be non-negative integers".into());
    };
    let (begin, end) = (begin as usize, end as usize);
    if begin > end {
        return Err(format!("data_offsets [{begin}, {end}] run backwards"));
    }
    // checked: a hostile shape like [2^40, 2^40] must not wrap around to a small size.
    let bytes = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
        .ok_or("shape overflows")?;
    if end - begin != bytes {
        return Err(format!(
            "shape {shape:?} of {dtype:?} needs {bytes} bytes, offsets give {}",
            end - begin
        ));
    }
    Ok(Entry {
        dtype,
        shape,
        begin,
        end,
    })
}

/// Serialize F32 tensors (tests only: real files come from the fetch step). Tensors are laid
/// out in the order given.
#[cfg(test)]
pub(crate) fn serialize(tensors: &[(&str, &Tensor)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, t) in tensors {
        let begin = data.len();
        data.extend(t.data().iter().flat_map(|x| x.to_le_bytes()));
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": t.shape(), "data_offsets": [begin, data.len()]}),
        );
    }
    let header = Value::Object(header).to_string();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a file from a header string and raw data bytes.
    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    fn f32_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn err(bytes: Vec<u8>) -> String {
        match SafeTensors::from_bytes(bytes) {
            Ok(_) => panic!("expected the file to be rejected"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn reads_tensors_in_any_header_order() {
        // `b` is listed first in the header but stored second: lookups go by offsets, not order.
        let header = r#"{"b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]},
                         "__metadata__":{"format":"pt"},
                         "a":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}}"#;
        let data = f32_bytes(&[1.0, -2.0, 0.5, 1e-3, f32::MAX]);
        let st = SafeTensors::from_bytes(file(header, &data)).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["a", "b"]);
        let a = st.tensor("a").unwrap();
        assert_eq!((a.shape(), a.data()), (&[2, 1][..], &[1.0, -2.0][..]));
        assert_eq!(st.tensor("b").unwrap().data(), &[0.5, 1e-3, f32::MAX]);
        assert!(st.tensor("c").is_err());
    }

    #[test]
    fn serialize_round_trips() {
        let a = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(&[1], vec![-0.0]).unwrap();
        let st = SafeTensors::from_bytes(serialize(&[("a", &a), ("b", &b)])).unwrap();
        assert_eq!(st.tensor("a").unwrap(), a);
        assert_eq!(
            st.tensor("b").unwrap().data()[0].to_bits(),
            (-0.0f32).to_bits()
        );
    }

    #[test]
    fn data_section_need_not_be_aligned() {
        // Pad the header (trailing spaces are legal JSON) so the data starts at an odd offset:
        // decoding must not assume 4-byte alignment.
        let mut header = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_string();
        while (8 + header.len()) % 4 != 1 {
            header.push(' ');
        }
        let st = SafeTensors::from_bytes(file(&header, &f32_bytes(&[3.25]))).unwrap();
        assert_eq!(st.tensor("x").unwrap().data(), &[3.25]);
    }

    #[test]
    fn scalars_and_empty_tensors() {
        let header = r#"{"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
                         "e":{"dtype":"F32","shape":[0,5],"data_offsets":[4,4]}}"#;
        let st = SafeTensors::from_bytes(file(header, &f32_bytes(&[7.0]))).unwrap();
        assert_eq!(st.tensor("s").unwrap().data(), &[7.0]);
        assert!(st.tensor("e").unwrap().is_empty());
    }

    #[test]
    fn bf16_widens_exactly_and_other_dtypes_are_refused() {
        // bf16 bit patterns: 1.0, -2.5, -0.0, the smallest subnormal, +inf, a quiet NaN. Each
        // must become the f32 with the same top 16 bits and zeros below (D68).
        let bf16: [u16; 6] = [0x3F80, 0xC020, 0x8000, 0x0001, 0x7F80, 0x7FC0];
        let data: Vec<u8> = bf16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let header = r#"{"w":{"dtype":"BF16","shape":[6],"data_offsets":[0,12]},
                         "h":{"dtype":"F16","shape":[2],"data_offsets":[12,16]}}"#;
        let mut bytes = data;
        bytes.extend_from_slice(&[0; 4]);
        let st = SafeTensors::from_bytes(file(header, &bytes)).unwrap();
        let w = st.tensor("w").unwrap();
        let bits: Vec<u32> = w.data().iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = bf16.iter().map(|&h| u32::from(h) << 16).collect();
        assert_eq!(bits, want);
        assert_eq!(&w.data()[..2], &[1.0, -2.5]);
        assert_eq!(st.entry("h").unwrap().dtype, Dtype::F16);
        assert!(
            st.tensor("h")
                .unwrap_err()
                .to_string()
                .contains("only F32 and BF16")
        );
    }

    #[test]
    fn empty_header_is_a_valid_empty_file() {
        let st = SafeTensors::from_bytes(file("{}", &[])).unwrap();
        assert_eq!(st.names().count(), 0);
    }

    #[test]
    fn rejects_truncated_files() {
        assert!(err(vec![1, 2, 3]).contains("too short"));
        // The length field claims a 1000-byte header; the file has 10 bytes in total.
        let mut short = 1000u64.to_le_bytes().to_vec();
        short.extend_from_slice(b"{}");
        assert!(err(short).contains("doesn't fit"));
        // Off by one: the header runs exactly one byte past the end of the file.
        let mut one_over = file("{}", &[]);
        one_over[0] = 3;
        assert!(err(one_over).contains("doesn't fit"));
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(err(huge).contains("doesn't fit"));
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(err(file("{not json", &[])).contains("not valid JSON"));
        assert!(err(file("[1,2]", &[])).contains("not a JSON object"));
        let cases = [
            (
                r#"{"x":{"shape":[1],"data_offsets":[0,4]}}"#,
                "missing dtype",
            ),
            (
                r#"{"x":{"dtype":"F31","shape":[1],"data_offsets":[0,4]}}"#,
                "unknown dtype",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
                "bad dimension",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
                "two elements",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
                "backwards",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                "needs 8 bytes",
            ),
            // Too many bytes for the shape is as wrong as too few.
            (
                r#"{"x":{"dtype":"F32","shape":[],"data_offsets":[0,8]}}"#,
                "needs 4 bytes",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}"#,
                "overflows",
            ),
        ];
        for (header, want) in cases {
            let e = err(file(header, &[0; 4]));
            assert!(e.contains(want), "{header}: got `{e}`, wanted `{want}`");
        }
    }

    #[test]
    fn tensors_must_tile_the_data_section() {
        let one = |a: u32, b: u32| format!(r#""dtype":"F32","shape":[1],"data_offsets":[{a},{b}]"#);
        let past_end = format!(r#"{{"x":{{{}}}}}"#, one(4, 8));
        assert!(err(file(&past_end, &[0; 4])).contains("ends at byte 8"));
        let gap = format!(r#"{{"x":{{{}}},"y":{{{}}}}}"#, one(0, 4), one(8, 12));
        assert!(err(file(&gap, &[0; 12])).contains("gap before tensor `y`"));
        let overlap = format!(
            r#"{{"x":{{"dtype":"F32","shape":[2],"data_offsets":[0,8]}},"y":{{{}}}}}"#,
            one(4, 8)
        );
        assert!(err(file(&overlap, &[0; 8])).contains("overlap"));
        let trailing = format!(r#"{{"x":{{{}}}}}"#, one(0, 4));
        assert!(err(file(&trailing, &[0; 6])).contains("cover 4 bytes"));
    }
}
