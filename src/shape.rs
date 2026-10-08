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

/// The GPU layout (D45): `x: [T, in]`, `w: [in, out]`, `b: [out]` or none -> `(T, in, out)`.
pub(crate) fn linear_in_out(
    x: &[usize],
    w: &[usize],
    b: Option<&[usize]>,
) -> Result<(usize, usize, usize)> {
    match w {
        &[n_in, n_out] => linear(x, &[n_out, n_in], b),
        _ => Err(Error::Shape(format!(
            "linear: x {x:?} and w {w:?} must be 2-D"
        ))),
    }
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

/// `qkv: [T, (H + 2 KV) d]` for H query heads sharing KV key/value heads (D71). Returns
/// `(T, d)`.
pub(crate) fn qkv_gqa(
    op: &str,
    qkv: &[usize],
    n_head: usize,
    n_kv_head: usize,
) -> Result<(usize, usize)> {
    let &[t, width] = qkv else {
        return Err(Error::Shape(format!("{op}: qkv {qkv:?} must be 2-D")));
    };
    if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
        return Err(Error::Shape(format!(
            "{op}: {n_head} query heads don't group over {n_kv_head} key/value heads"
        )));
    }
    let heads = n_head + 2 * n_kv_head;
    if !width.is_multiple_of(heads) {
        return Err(Error::Shape(format!(
            "{op}: qkv {qkv:?} doesn't split into {n_head} + 2 x {n_kv_head} heads"
        )));
    }
    Ok((t, width / heads))
}

/// `x: [T, E]` and `gain: [E]`. Returns `(T, E)`.
pub(crate) fn rms_norm(x: &[usize], gain: &[usize]) -> Result<(usize, usize)> {
    match (x, gain) {
        (&[t, e], &[g]) if g == e => Ok((t, e)),
        _ => Err(Error::Shape(format!("rms_norm: x {x:?}, gain {gain:?}"))),
    }
}

/// `x: [T, W]` whose first `n_rot` heads of width d (even) are rotated at positions
/// `start..start + T`, with tables `[n_pos, d/2]` covering them. Returns `(T, W)`.
pub(crate) fn rope(
    x: &[usize],
    n_rot: usize,
    d: usize,
    start: usize,
    cos: &[usize],
    sin: &[usize],
) -> Result<(usize, usize)> {
    let &[t, w] = x else {
        return Err(Error::Shape(format!("rope: x {x:?} must be 2-D")));
    };
    if d == 0 || !d.is_multiple_of(2) || n_rot * d > w {
        return Err(Error::Shape(format!(
            "rope: {n_rot} heads of width {d} (must be even) in rows of {w}"
        )));
    }
    match (cos, sin) {
        (&[n, h], s) if h == d / 2 && s == cos && start + t <= n => Ok((t, w)),
        _ => Err(Error::Shape(format!(
            "rope: tables {cos:?} / {sin:?} for d = {d} and positions {start}..{}",
            start + t
        ))),
    }
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
