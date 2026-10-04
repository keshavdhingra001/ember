//! GPU `add` vs the CPU reference (D3), plus upload/readback round trips.

mod common;

use common::gpu;
use ember::compare::{Tol, check};
use ember::rng::Rng;
use ember::{Tensor, cpu, ops};

/// Run GPU add on host tensors with an optional workgroup cap; return the result on the host.
fn gpu_add(a: &Tensor, b: &Tensor, max_groups: Option<u32>) -> Tensor {
    let g = gpu();
    let (ga, gb) = (g.upload(a), g.upload(b));
    let out = match max_groups {
        Some(cap) => ops::add_with_max_groups(g, &ga, &gb, cap),
        None => ops::add(g, &ga, &gb),
    }
    .unwrap();
    g.read(&out).unwrap()
}

fn random_pair(shape: &[usize], seed: u64) -> (Tensor, Tensor) {
    let n = shape.iter().product();
    let mut rng = Rng::new(seed);
    (
        Tensor::new(shape, rng.vec(n, -100.0, 100.0)).unwrap(),
        Tensor::new(shape, rng.vec(n, -100.0, 100.0)).unwrap(),
    )
}

fn assert_add_matches(a: &Tensor, b: &Tensor, max_groups: Option<u32>) {
    let want = cpu::add(a, b).unwrap();
    let got = gpu_add(a, b, max_groups);
    assert_eq!(got.shape(), want.shape());
    // One IEEE add per element on both sides: must be bit-for-bit equal (D5).
    if let Err(m) = check(got.data(), want.data(), Tol::EXACT) {
        panic!("n={} max_groups={max_groups:?}: {m}", a.len());
    }
}

#[test]
fn sizes_around_workgroup_boundaries() {
    // 255/256/257: one short of, exactly, and one past a 256-thread workgroup.
    for (seed, n) in [1, 2, 3, 255, 256, 257, 1000, 4096, 100_003]
        .into_iter()
        .enumerate()
    {
        let (a, b) = random_pair(&[n], seed as u64);
        assert_add_matches(&a, &b, None);
    }
}

#[test]
fn grid_stride_loop_covers_every_element() {
    // With 1, 3 or 7 workgroups each invocation must loop over many elements (D8).
    let (a, b) = random_pair(&[10_000], 11);
    for cap in [1, 3, 7] {
        assert_add_matches(&a, &b, Some(cap));
    }
}

#[test]
fn larger_than_one_dispatch_dimension() {
    // Past 65535 workgroups x 256 threads: only correct because of the grid-stride loop.
    let n = 65_535 * 256 + 1_000;
    let (a, b) = random_pair(&[n], 12);
    assert_add_matches(&a, &b, None);
}

#[test]
fn keeps_multi_dimensional_shape() {
    let (a, b) = random_pair(&[3, 5, 7], 13);
    assert_add_matches(&a, &b, None);
}

#[test]
fn empty_tensor() {
    let a = Tensor::zeros(&[0, 4]);
    let got = gpu_add(&a, &a, None);
    assert_eq!(got.shape(), &[0, 4]);
    assert!(got.is_empty());
}

#[test]
fn special_values() {
    let a = vec![
        f32::INFINITY,
        f32::INFINITY,
        f32::NAN,
        -0.0,
        f32::MAX,
        1.0,
        -1.0,
        1e-30,
    ];
    let b = vec![
        1.0,
        f32::NEG_INFINITY, // inf + -inf = NaN
        1.0,
        -0.0,
        f32::MAX,           // overflows to inf
        f32::EPSILON / 2.0, // rounds back to 1.0 (round-to-nearest-even)
        1.0,
        1e-30,
    ];
    let n = a.len();
    let (a, b) = (Tensor::new(&[n], a).unwrap(), Tensor::new(&[n], b).unwrap());
    assert_add_matches(&a, &b, None);
}

#[test]
fn shape_mismatch_is_an_error() {
    let g = gpu();
    let a = g.upload(&Tensor::zeros(&[2, 3]));
    let b = g.upload(&Tensor::zeros(&[3, 2]));
    assert!(ops::add(g, &a, &b).is_err());
}

#[test]
fn upload_read_round_trip_is_bitwise() {
    let g = gpu();
    let mut rng = Rng::new(14);
    let data: Vec<f32> = (0..5000)
        .map(|_| f32::from_bits(rng.next_u64() as u32))
        .filter(|x| !x.is_nan())
        .collect();
    let t = Tensor::new(&[data.len()], data).unwrap();
    let back = g.read(&g.upload(&t)).unwrap();
    let bits = |t: &Tensor| t.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&back), bits(&t));
}

#[test]
fn same_input_same_bits() {
    // D4: repeated runs on one device are bit-identical.
    let (a, b) = random_pair(&[50_000], 15);
    let first = gpu_add(&a, &b, None);
    for _ in 0..3 {
        assert_eq!(gpu_add(&a, &b, None).data(), first.data());
    }
}
