//! `ember` CLI: `info` (what GPU we got), `selftest` (GPU add vs the CPU reference), and
//! `tokenize` (GPT-2's BPE, step by step).

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ember::compare::{self, Tol};
use ember::rng::Rng;
use ember::{Gpu, Tensor, Tokenizer, cpu, ops};

const USAGE: &str = "usage: ember [info | selftest | tokenize <text>]";
const GPT2_DIR: &str = "data/gpt2";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("info") => info(),
        Some("selftest") => selftest(),
        Some("tokenize") if args.len() > 1 => tokenize(&args[1..].join(" ")),
        Some(other) => {
            eprintln!("unknown command or missing argument: `{other}`\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn info() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = Gpu::new()?;
    let i = &gpu.info;
    let l = &gpu.limits;
    println!(
        "adapter      {} ({:?}, {:?})",
        i.name, i.device_type, i.backend
    );
    println!("driver       {} {}", i.driver, i.driver_info);
    println!("max buffer   {} MiB", l.max_buffer_size >> 20);
    println!(
        "max storage binding  {} MiB",
        l.max_storage_buffer_binding_size >> 20
    );
    println!(
        "workgroups   {} per dim, {} invocations, {} B shared memory",
        l.max_compute_workgroups_per_dimension,
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroup_storage_size
    );
    Ok(())
}

fn selftest() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = Gpu::new()?;
    let n = 1 << 20;
    let mut rng = Rng::new(42);
    let a = Tensor::new(&[n], rng.vec(n, -1.0, 1.0))?;
    let b = Tensor::new(&[n], rng.vec(n, -1.0, 1.0))?;
    let want = cpu::add(&a, &b)?;

    let start = Instant::now();
    let (ga, gb) = (gpu.upload(&a), gpu.upload(&b));
    let got = gpu.read(&ops::add(&gpu, &ga, &gb)?)?;
    let elapsed = start.elapsed();

    let stats = compare::check(got.data(), want.data(), Tol::EXACT)?;
    println!(
        "add  n={n}  ok  max_abs={:e}  ({:.1} ms incl. upload + readback, first run)",
        stats.max_abs,
        elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}

/// Print each pre-split word and the tokens BPE made of it.
fn tokenize(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tok = Tokenizer::load(Path::new(GPT2_DIR))?;
    let mut total = 0;
    for word in tok.split(text)? {
        let ids = tok.encode(word)?;
        let pieces: Vec<String> = ids
            .iter()
            .map(|&id| {
                format!(
                    "{:?}={id}",
                    String::from_utf8_lossy(tok.token_bytes(id).unwrap())
                )
            })
            .collect();
        println!("{word:?} -> {}", pieces.join(" "));
        total += ids.len();
    }
    println!("{total} tokens");
    Ok(())
}
