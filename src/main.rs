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
        Some("generate") => generate(&args[1..]),
        Some("bench") => bench(&args[1..]),
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
    let Opts {
        n,
        cpu: cpu_ref,
        no_cache,
        prompt,
    } = parse_opts(args, 20, &["--cpu", "--no-cache"])?;
    let (tok, w, ids) = load_gpt2(&prompt)?;

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

/// The tokenizer and weights from `data/gpt2/`, and the prompt's token ids.
fn load_gpt2(prompt: &str) -> Result<(Tokenizer, Weights, Vec<u32>), Box<dyn std::error::Error>> {
    let dir = Path::new(GPT2_DIR);
    let tok = Tokenizer::load(dir)?;
    let w = Weights::load(dir)?;
    let ids = tok.encode(prompt)?;
    Ok((tok, w, ids))
}

/// Options before the prompt. `-n` and the flags a command allows come first, in any order;
/// everything from the first other word on is the prompt.
struct Opts {
    n: usize,
    cpu: bool,
    no_cache: bool,
    prompt: String,
}

fn parse_opts(
    args: &[String],
    default_n: usize,
    flags: &[&str],
) -> Result<Opts, Box<dyn std::error::Error>> {
    let mut o = Opts {
        n: default_n,
        cpu: false,
        no_cache: false,
        prompt: String::new(),
    };
    let mut rest = args;
    loop {
        match rest {
            [flag, v, tail @ ..] if flag == "-n" => {
                o.n = v.parse().map_err(|_| format!("-n: `{v}` is not a count"))?;
                rest = tail;
            }
            [flag] if flag == "-n" => return Err("-n needs a count".into()),
            [flag, tail @ ..] if flags.contains(&flag.as_str()) => {
                match flag.as_str() {
                    "--cpu" => o.cpu = true,
                    _ => o.no_cache = true,
                }
                rest = tail;
            }
            [flag, ..] if flag.starts_with("--") => {
                return Err(format!("unknown flag `{flag}`\n\n{USAGE}").into());
            }
            _ => break,
        }
    }
    o.prompt = rest.join(" ");
    if o.prompt.is_empty() {
        return Err(format!("missing prompt\n\n{USAGE}").into());
    }
    Ok(o)
}

/// `ember bench [-n N] <prompt>` (D35): prefill latency and decode rate with the KV cache, and
/// the uncached rate for comparison. 1 warm-up run, then the median of 5, wall clock.
fn bench(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const RUNS: usize = 5;
    let Opts { n, prompt, .. } = parse_opts(args, 32, &[])?;
    let (_, w, ids) = load_gpt2(&prompt)?;
    // The uncached run generates n + 1 tokens after the prompt. The greedy loop would quietly
    // stop at the context length and the rates below would then divide by the wrong count.
    if n == 0 || ids.len() + n + 1 > w.config.n_ctx {
        return Err(format!(
            "bench needs 1 <= n and prompt + n + 1 <= {} tokens (prompt is {}, n is {n})",
            w.config.n_ctx,
            ids.len()
        )
        .into());
    }
    let gpu = Gpu::new()?;
    let gw = GpuWeights::upload(&gpu, &w);
    let mut cache = KvCache::new(&gpu, &w.config);
    let argmax = |l: &[f32]| -> Result<u32, Box<dyn std::error::Error>> {
        Ok(cpu::argmax(l).ok_or("logits contain NaN")? as u32)
    };

    // One cached run: (prefill seconds, decode seconds for n tokens). The decode clock covers
    // everything a real step does: the model, the 201 KB readback and the argmax.
    let mut cached = || -> Result<(f64, f64), Box<dyn std::error::Error>> {
        cache.clear();
        let t0 = Instant::now();
        let mut next = argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &ids)?)?;
        let prefill = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..n {
            next = argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &[next])?)?;
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
    use super::{parse_opts, take_utf8};

    fn args(s: &str) -> Vec<String> {
        s.split(' ')
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    }

    #[test]
    fn options_come_before_the_prompt_in_any_order() {
        let o = parse_opts(
            &args("-n 5 --cpu hello world"),
            20,
            &["--cpu", "--no-cache"],
        )
        .unwrap();
        assert_eq!((o.n, o.cpu, o.no_cache), (5, true, false));
        assert_eq!(o.prompt, "hello world");
        let o = parse_opts(&args("--no-cache -n 3 hi"), 20, &["--cpu", "--no-cache"]).unwrap();
        assert_eq!(
            (o.n, o.cpu, o.no_cache, o.prompt.as_str()),
            (3, false, true, "hi")
        );
        // After the first prompt word, `-n` is text. A lone `-` word is text too.
        let o = parse_opts(&args("say -n - twice"), 20, &[]).unwrap();
        assert_eq!((o.n, o.prompt.as_str()), (20, "say -n - twice"));
    }

    #[test]
    fn bad_options_are_errors_not_prompts() {
        let flags = ["--cpu", "--no-cache"];
        for bad in ["", "-n 5", "-n", "-n five hi", "--cpu", "--fast hi"] {
            assert!(parse_opts(&args(bad), 20, &flags).is_err(), "`{bad}`");
        }
        // bench takes no flags: `--cpu` there is unknown.
        assert!(parse_opts(&args("--cpu hi"), 32, &[]).is_err());
    }

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
