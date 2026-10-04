//! `ember` CLI: `info` (what GPU we got), `selftest` (GPU add vs the CPU reference),
//! `tokenize` (GPT-2's BPE, step by step), `generate` (greedy GPT-2 on the GPU with a KV cache,
//! or without one, or on the CPU reference) and `bench` (prefill and decode timings, D35).
//! Wall-clock timing lives here, in the CLI, never in the engine (D4).

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ember::compare::{self, Tol};
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights, KvCache};
use ember::gpt2::{self, Weights};
use ember::rng::Rng;
use ember::{Gpu, Tensor, Tokenizer, cpu, ops};

const USAGE: &str = "usage: ember [info | selftest | tokenize <text>\n              | generate [--cpu | --no-cache] [-n <tokens>] <prompt>\n              | bench [-n <tokens>] <prompt>]";
const GPT2_DIR: &str = "data/gpt2";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("info") => info(),
        Some("selftest") => selftest(),
        Some("tokenize") if args.len() > 1 => tokenize(&args[1..].join(" ")),
        Some("generate") if args.len() > 1 => generate(&args[1..]),
        Some("bench") if args.len() > 1 => bench(&args[1..]),
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

/// Greedy GPT-2, streaming tokens as they come: on the GPU with a KV cache (D29), on the GPU
/// recomputing everything with `--no-cache` (M3), or on the CPU reference with `--cpu`.
fn generate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args;
    let (mut cpu_ref, mut no_cache) = (false, false);
    while let [flag, rest @ ..] = args {
        match flag.as_str() {
            "--cpu" => cpu_ref = true,
            "--no-cache" => no_cache = true,
            _ => break,
        }
        args = rest;
    }
    let (n, prompt) = parse_n(args, 20)?;
    let dir = Path::new(GPT2_DIR);
    let tok = Tokenizer::load(dir)?;
    let w = Weights::load(dir)?;
    let ids = tok.encode(&prompt)?;

    let mut pending = Vec::new();
    let on_token = |id: u32| {
        pending.extend_from_slice(tok.token_bytes(id).unwrap());
        print!("{}", take_utf8(&mut pending));
        let _ = std::io::stdout().flush();
    };
    let (out, secs, device) = if cpu_ref {
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start = Instant::now();
        let out = gpt2::generate_greedy(&w, &ids, n, on_token)?;
        let secs = start.elapsed().as_secs_f64();
        (out, secs, "CPU reference, no KV cache".to_string())
    } else {
        let gpu = Gpu::new()?;
        let gw = GpuWeights::upload(&gpu, &w);
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start = Instant::now();
        let out = if no_cache {
            gpt2_gpu::generate_greedy_uncached(&gpu, &gw, &ids, n, on_token)?
        } else {
            gpt2_gpu::generate_greedy(&gpu, &gw, &ids, n, on_token)?
        };
        let secs = start.elapsed().as_secs_f64();
        let mode = if no_cache { "no KV cache" } else { "KV cache" };
        (out, secs, format!("{}, {mode}", gpu.info.name))
    };
    println!("{}", String::from_utf8_lossy(&pending));
    eprintln!(
        "[{} prompt + {} generated tokens in {secs:.2} s, {:.2} tokens/s incl. prefill; {device}]",
        ids.len(),
        out.len(),
        out.len() as f64 / secs
    );
    Ok(())
}

/// `[-n N] <words...>` -> (N, the words joined by spaces).
fn parse_n(args: &[String], default: usize) -> Result<(usize, String), Box<dyn std::error::Error>> {
    Ok(match args {
        [flag, n, rest @ ..] if flag == "-n" && !rest.is_empty() => (n.parse()?, rest.join(" ")),
        _ => (default, args.join(" ")),
    })
}

/// `ember bench [-n N] <prompt>` (D35): prefill latency and decode rate with the KV cache, and
/// the uncached rate for comparison. 1 warm-up run, then the median of 5, wall clock.
fn bench(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const RUNS: usize = 5;
    let (n, prompt) = parse_n(args, 32)?;
    let dir = Path::new(GPT2_DIR);
    let tok = Tokenizer::load(dir)?;
    let w = Weights::load(dir)?;
    let ids = tok.encode(&prompt)?;
    let gpu = Gpu::new()?;
    let gw = GpuWeights::upload(&gpu, &w);
    let mut cache = KvCache::new(&gpu, &w.config);
    let argmax = |l: &[f32]| cpu::argmax(l).expect("logits contain NaN") as u32;

    // One cached run: (prefill seconds, decode seconds for n tokens). The decode clock covers
    // everything a real step does: the model, the 201 KB readback and the argmax.
    let mut cached = || -> Result<(f64, f64), Box<dyn std::error::Error>> {
        cache.clear();
        let t0 = Instant::now();
        let mut next = argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &ids)?);
        let prefill = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..n {
            next = argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &[next])?);
        }
        Ok((prefill, t1.elapsed().as_secs_f64()))
    };
    cached()?; // warm-up
    let runs = (0..RUNS).map(|_| cached()).collect::<Result<Vec<_>, _>>()?;
    let prefill = median(runs.iter().map(|r| r.0).collect());
    let decode = median(runs.iter().map(|r| r.1).collect());

    let uncached = || -> Result<f64, Box<dyn std::error::Error>> {
        let t = Instant::now();
        gpt2_gpu::generate_greedy_uncached(&gpu, &gw, &ids, n + 1, |_| {})?;
        Ok(t.elapsed().as_secs_f64())
    };
    uncached()?;
    let full = median((0..RUNS).map(|_| uncached()).collect::<Result<_, _>>()?);

    println!(
        "{} ({:?}); prompt {} tokens, {n} decode steps; median of {RUNS} after 1 warm-up, wall clock",
        gpu.info.name,
        gpu.info.backend,
        ids.len()
    );
    println!("prefill            {:8.1} ms", prefill * 1e3);
    println!(
        "decode (KV cache)  {:8.1} ms/token  {:6.2} tokens/s",
        decode * 1e3 / n as f64,
        n as f64 / decode
    );
    println!(
        "no cache (M3)      {:8.1} ms/token  {:6.2} tokens/s  (prompt + {} tokens, all recomputed)",
        full * 1e3 / (n + 1) as f64,
        (n + 1) as f64 / full,
        n + 1
    );
    Ok(())
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// Remove and return the longest printable prefix of `buf`. A token can end halfway through a
/// multi-byte character; those bytes wait in `buf` for the next token. Invalid bytes (which
/// can't become valid later) are printed as U+FFFD.
fn take_utf8(buf: &mut Vec<u8>) -> String {
    let mut out = String::new();
    loop {
        let n = match std::str::from_utf8(buf) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to() + e.error_len().unwrap_or(0),
        };
        if n == 0 {
            return out; // empty, or only the start of an incomplete character
        }
        out.push_str(&String::from_utf8_lossy(&buf[..n]));
        buf.drain(..n);
    }
}

#[cfg(test)]
mod tests {
    use super::take_utf8;

    #[test]
    fn take_utf8_holds_back_split_characters() {
        let pizza = "🍕".as_bytes(); // 4 bytes
        let mut buf = b"a".to_vec();
        buf.extend_from_slice(&pizza[..2]);
        assert_eq!(take_utf8(&mut buf), "a");
        assert_eq!(buf, &pizza[..2]);
        buf.extend_from_slice(&pizza[2..]);
        assert_eq!(take_utf8(&mut buf), "🍕");
        assert!(buf.is_empty());
        // A lone continuation byte can never become valid: print it as U+FFFD and move on.
        buf.extend_from_slice(&[0x80, b'b']);
        assert_eq!(take_utf8(&mut buf), "\u{FFFD}b");
        buf.extend_from_slice(&[0xFF, 0xFE, b'c', pizza[0]]);
        assert_eq!(take_utf8(&mut buf), "\u{FFFD}\u{FFFD}c");
        assert_eq!(buf, &pizza[..1]);
    }
}
