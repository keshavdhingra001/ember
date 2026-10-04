//! A tiny seeded random number generator (SplitMix64) for test data and, later, seeded
//! sampling. Hand-written so a seed means the same sequence forever: a crate upgrade can't
//! silently change every generated input. Same generator as lob's.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`: the top 24 bits fill an f32 mantissa exactly, so every value is
    /// representable and 1.0 is never returned.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// `n` values uniform in `[lo, hi)`.
    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_first_value() {
        // SplitMix64's published first output for seed 0. If this changes, every
        // generated test input changes with it.
        assert_eq!(Rng::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn unit_is_in_range() {
        let mut rng = Rng::new(3);
        for _ in 0..10_000 {
            let x = rng.unit();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
