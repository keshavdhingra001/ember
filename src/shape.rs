//! Shape and input checks shared by the CPU reference ops (`cpu.rs`) and their GPU twins
//! (`ops.rs`), so both sides accept and reject exactly the same arguments.

use crate::error::{Error, Result};

pub(crate) fn same(op: &str, a: &[usize], b: &[usize]) -> Result<()> {
    if a != b {
        return Err(Error::Shape(format!("{op}: {a:?} vs {b:?}")));
    }
    Ok(())
}

/// `x: [T, in]`, `w: [out, in]`, `b: [out]` or none -> `(T, in, out)`.
pub(crate) fn linear(
    x: &[usize],
    w: &[usize],
    b: Option<&[usize]>,
) -> Result<(usize, usize, usize)> {
    let (&[t, n_in], &[n_out, w_in]) = (x, w) else {
        return Err(Error::Shape(format!(
            "linear: x {x:?} and w {w:?} must be 2-D"
        )));
    };
    if n_in != w_in || b.is_some_and(|b| b != [n_out]) {
        return Err(Error::Shape(format!("linear: x {x:?}, w {w:?}, b {b:?}")));
    }
    Ok((t, n_in, n_out))
}

/// `x: [rows, cols]` with `gain`, `bias`: `[cols]` -> `(rows, cols)`.
pub(crate) fn layer_norm(x: &[usize], gain: &[usize], bias: &[usize]) -> Result<(usize, usize)> {
    let &[rows, cols] = x else {
        return Err(Error::Shape(format!("layer_norm: x {x:?} must be 2-D")));
    };
    if gain != [cols] || bias != [cols] {
        return Err(Error::Shape(format!(
            "layer_norm: x {x:?}, gain {gain:?}, bias {bias:?}"
        )));
    }
    Ok((rows, cols))
}

/// `qkv: [T, 3E]` that splits into 3 x `n_head` heads -> `(T, E)`.
pub(crate) fn qkv(op: &str, qkv: &[usize], n_head: usize) -> Result<(usize, usize)> {
    let &[t, three_e] = qkv else {
        return Err(Error::Shape(format!("{op}: qkv {qkv:?} must be 2-D")));
    };
    if n_head == 0 || three_e % (3 * n_head) != 0 {
        return Err(Error::Shape(format!(
            "{op}: qkv {qkv:?} doesn't split into 3 x {n_head} heads"
        )));
    }
    Ok((t, three_e / 3))
}

/// `wte: [V, E]`, `wpe: [n_ctx, E]`, and `ids` at positions `start..`: every id inside the
/// vocab, every position inside the context. Returns `E`. A kernel can't report a bad id, only
/// read the wrong row (D9), so this runs before every embed.
pub(crate) fn embed(wte: &[usize], wpe: &[usize], ids: &[u32], start: usize) -> Result<usize> {
    let (&[v, e], &[n_ctx, e2]) = (wte, wpe) else {
        return Err(Error::Shape(format!(
            "embed: wte {wte:?} and wpe {wpe:?} must be 2-D"
        )));
    };
    if e != e2 {
        return Err(Error::Shape(format!("embed: wte {wte:?} vs wpe {wpe:?}")));
    }
    if start + ids.len() > n_ctx {
        return Err(Error::Input(format!(
            "positions {start}..{} exceed the context length {n_ctx}",
            start + ids.len()
        )));
    }
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= v) {
        return Err(Error::Input(format!(
            "token id {bad} is outside the vocab (size {v})"
        )));
    }
    Ok(e)
}
