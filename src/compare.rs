//! Float comparison with explicit tolerances (D5). CPU and GPU results legitimately differ in
//! the last bits (fused multiply-add, summation order, approximate `exp`), so kernels are
//! checked against the CPU reference with a per-op tolerance instead of bitwise equality.

use std::fmt;

/// Element `i` passes when `|got - want| <= abs + rel * |want|`.
#[derive(Debug, Clone, Copy)]
pub struct Tol {
    pub abs: f32,
    pub rel: f32,
}

impl Tol {
    /// For ops that must be exact: a single IEEE operation per element (add, copy).
    pub const EXACT: Tol = Tol { abs: 0.0, rel: 0.0 };
}

/// How close a passing comparison was, so tests can report it and tolerances can be tightened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub max_abs: f32,
    pub max_rel: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch {
    Length {
        got: usize,
        want: usize,
    },
    /// `index` is the worst failing element (largest error beyond its allowance);
    /// `failures` counts every failing element.
    Value {
        index: usize,
        got: f32,
        want: f32,
        failures: usize,
    },
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mismatch::Length { got, want } => write!(f, "length {got}, expected {want}"),
            Mismatch::Value {
                index,
                got,
                want,
                failures,
            } => write!(
                f,
                "{failures} element(s) out of tolerance; worst at [{index}]: got {got:e}, expected {want:e} (error {:e})",
                (got - want).abs()
            ),
        }
    }
}

impl std::error::Error for Mismatch {}

/// Compare `got` (GPU) against `want` (CPU reference).
pub fn check(got: &[f32], want: &[f32], tol: Tol) -> Result<Stats, Mismatch> {
    if got.len() != want.len() {
        return Err(Mismatch::Length {
            got: got.len(),
            want: want.len(),
        });
    }
    let mut stats = Stats {
        max_abs: 0.0,
        max_rel: 0.0,
    };
    let mut worst: Option<(usize, f32)> = None;
    let mut failures = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        match excess(g, w, tol) {
            None => {
                if g.is_finite() && w.is_finite() {
                    let err = (g - w).abs();
                    stats.max_abs = stats.max_abs.max(err);
                    if w != 0.0 {
                        stats.max_rel = stats.max_rel.max(err / w.abs());
                    }
                }
            }
            Some(over) => {
                failures += 1;
                if worst.is_none_or(|(_, o)| over > o) {
                    worst = Some((i, over));
                }
            }
        }
    }
    match worst {
        None => Ok(stats),
        Some((index, _)) => Err(Mismatch::Value {
            index,
            got: got[index],
            want: want[index],
            failures,
        }),
    }
}

/// `None` if the element passes, otherwise how far past its allowance it is (infinite for
/// NaN / infinity mismatches, so those always rank as the worst).
fn excess(g: f32, w: f32, tol: Tol) -> Option<f32> {
    if g.is_nan() || w.is_nan() {
        return if g.is_nan() && w.is_nan() {
            None
        } else {
            Some(f32::INFINITY)
        };
    }
    if g.is_infinite() || w.is_infinite() {
        return if g == w { None } else { Some(f32::INFINITY) };
    }
    let over = (g - w).abs() - (tol.abs + tol.rel * w.abs());
    if over <= 0.0 { None } else { Some(over) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_passes() {
        let s = check(&[1.0, -2.0, 0.0], &[1.0, -2.0, -0.0], Tol::EXACT).unwrap();
        assert_eq!(s.max_abs, 0.0);
    }

    #[test]
    fn one_ulp_fails_exact_but_passes_rel() {
        let w = 1.0f32;
        let g = f32::from_bits(w.to_bits() + 1);
        assert!(check(&[g], &[w], Tol::EXACT).is_err());
        let tol = Tol {
            abs: 0.0,
            rel: 2.0 * f32::EPSILON,
        };
        assert!(check(&[g], &[w], tol).is_ok());
    }

    #[test]
    fn relative_error_is_measured_against_the_reference() {
        // |2 - 1| = 1 > 0.6 * |want| = 0.6, so this fails. Scaling by |got| (1.2) would pass:
        // a wrong GPU value must not loosen its own tolerance.
        let tol = Tol { abs: 0.0, rel: 0.6 };
        assert!(check(&[2.0], &[1.0], tol).is_err());
        assert!(check(&[1.5], &[1.0], tol).is_ok());
    }

    #[test]
    fn reports_worst_element_and_count() {
        let want = [0.0, 0.0, 0.0, 0.0];
        let got = [0.0, 0.5, 0.0, 3.0];
        let tol = Tol { abs: 0.1, rel: 0.0 };
        match check(&got, &want, tol) {
            Err(Mismatch::Value {
                index, failures, ..
            }) => {
                assert_eq!(index, 3);
                assert_eq!(failures, 2);
            }
            other => panic!("expected a value mismatch, got {other:?}"),
        }
    }

    #[test]
    fn nan_and_inf_must_match_exactly() {
        let loose = Tol {
            abs: 1e30,
            rel: 1e30,
        };
        assert!(check(&[f32::NAN], &[f32::NAN], Tol::EXACT).is_ok());
        assert!(check(&[f32::NAN], &[1.0], loose).is_err());
        assert!(check(&[1.0], &[f32::NAN], loose).is_err());
        assert!(check(&[f32::INFINITY], &[f32::INFINITY], Tol::EXACT).is_ok());
        assert!(check(&[f32::INFINITY], &[f32::MAX], loose).is_err());
        assert!(check(&[f32::NEG_INFINITY], &[f32::INFINITY], loose).is_err());
    }

    #[test]
    fn length_mismatch() {
        assert_eq!(
            check(&[1.0], &[1.0, 2.0], Tol::EXACT),
            Err(Mismatch::Length { got: 1, want: 2 })
        );
    }
}
