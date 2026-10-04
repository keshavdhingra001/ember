//! `ember` CLI: `info` (what GPU we got), `selftest` (GPU add vs the CPU reference),
//! `tokenize` (GPT-2's BPE, step by step) and `generate` (greedy GPT-2 on the CPU reference).

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ember::compare::{self, Tol};
use ember::gpt2::{self, Weights};
use ember::rng::Rng;
use ember::{Gpu, Tensor, Tokenizer, cpu, ops};

const USAGE: &str =
    "usage: ember [info | selftest | tokenize <text> | generate [-n <tokens>] <prompt>]";
const GPT2_DIR: &str = "data/gpt2";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("info") => info(),
        Some("selftest") => selftest(),
        Some("tokenize") if args.len() > 1 => tokenize(&args[1..].join(" ")),
        Some("generate") if args.len() > 1 => generate(&args[1..]),
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

/// Greedy GPT-2 on the CPU reference, streaming tokens as they come. Slow by design (no KV
/// cache, naive matmul): it is the oracle, not the engine.
fn generate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (n, prompt) = match args {
        [flag, n, rest @ ..] if flag == "-n" && !rest.is_empty() => (n.parse()?, rest.join(" ")),
        _ => (20, args.join(" ")),
    };
    let dir = Path::new(GPT2_DIR);
    let tok = Tokenizer::load(dir)?;
    let w = Weights::load(dir)?;
    let ids = tok.encode(&prompt)?;

    print!("{prompt}");
    std::io::stdout().flush()?;
    let start = Instant::now();
    let mut pending = Vec::new();
    let out = gpt2::generate_greedy(&w, &ids, n, |id| {
        pending.extend_from_slice(tok.token_bytes(id).unwrap());
        print!("{}", take_utf8(&mut pending));
        let _ = std::io::stdout().flush();
    })?;
    println!("{}", String::from_utf8_lossy(&pending));
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "[{} prompt + {} generated tokens in {secs:.1} s, {:.2} tokens/s; CPU reference, no KV cache]",
        ids.len(),
        out.len(),
        out.len() as f64 / secs
    );
    Ok(())
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
