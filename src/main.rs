//! `ember` CLI: `info` (what GPU we got), `selftest` (GPU add vs the CPU reference),
//! `tokenize` (GPT-2's BPE, step by step), `generate` (greedy GPT-2 on the GPU with a KV cache,
//! or without one, or on the CPU reference), `bench` (prefill and decode timings, D35),
//! `profile` (per-kernel GPU times, D36-D41) and `matmul` (the linear kernels vs the roofs, D47).
//! Wall-clock timing lives here, in the CLI, never in the engine (D4).

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ember::compare::{self, Tol};
use ember::gpt2::gpu::{self as gpt2_gpu, GpuWeights, KvCache};
use ember::gpt2::{self, Weights};
use ember::profile::{self, KernelTime};
use ember::rng::Rng;
use ember::{Gpu, GpuTensor, Tensor, Tokenizer, cpu, ops};

const USAGE: &str = "usage: ember [info | selftest | tokenize <text>\n              | generate [--cpu | --no-cache] [-n <tokens>] <prompt>\n              | bench [-n <tokens>] <prompt>\n              | profile [-n <runs>] <prompt>\n              | matmul [-n <runs>]]";
const GPT2_DIR: &str = "data/gpt2";

type CliResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("info") => info(),
        Some("selftest") => selftest(),
        Some("tokenize") if args.len() > 1 => tokenize(&args[1..].join(" ")),
        Some("generate") => generate(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("profile") => profile_cmd(&args[1..]),
        Some("matmul") => matmul_cmd(&args[1..]),
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

fn info() -> CliResult {
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
    if gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        println!(
            "timestamps   yes, one tick = {} ns",
            gpu.queue.get_timestamp_period()
        );
    } else {
        println!("timestamps   no (TIMESTAMP_QUERY unsupported: `ember profile` can't run)");
    }
    Ok(())
}

fn selftest() -> CliResult {
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
fn tokenize(text: &str) -> CliResult {
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
fn generate(args: &[String]) -> CliResult {
    let Opts {
        n,
        cpu: cpu_ref,
        no_cache,
        prompt,
    } = prompt_opts(args, 20, &["--cpu", "--no-cache"])?;
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
        let gw = GpuWeights::upload(&gpu, &w)?;
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
fn load_gpt2(prompt: &str) -> CliResult<(Tokenizer, Weights, Vec<u32>)> {
    let dir = Path::new(GPT2_DIR);
    let tok = Tokenizer::load(dir)?;
    let w = Weights::load(dir)?;
    let ids = tok.encode(prompt)?;
    Ok((tok, w, ids))
}

/// Options before the prompt. `-n` and the flags a command allows come first, in any order;
/// everything from the first other word on is the prompt (possibly empty; see `prompt_opts`).
struct Opts {
    n: usize,
    cpu: bool,
    no_cache: bool,
    prompt: String,
}

fn parse_opts(args: &[String], default_n: usize, flags: &[&str]) -> CliResult<Opts> {
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
    Ok(o)
}

/// `parse_opts` for a command that needs a prompt.
fn prompt_opts(args: &[String], default_n: usize, flags: &[&str]) -> CliResult<Opts> {
    let o = parse_opts(args, default_n, flags)?;
    if o.prompt.is_empty() {
        return Err(format!("missing prompt\n\n{USAGE}").into());
    }
    Ok(o)
}

/// `-n` as a count of measured runs: at least one.
fn measured_runs(n: usize) -> CliResult<usize> {
    if n == 0 {
        return Err("-n: at least one measured run".into());
    }
    Ok(n)
}

/// The adapter and driver, for the first line of every report.
fn device(gpu: &Gpu) -> String {
    let i = &gpu.info;
    format!(
        "{} ({:?}), {} {}",
        i.name, i.backend, i.driver, i.driver_info
    )
}

/// `ember bench [-n N] <prompt>` (D35): prefill latency and decode rate with the KV cache, and
/// the uncached rate for comparison. 1 warm-up run, then the median of 5, wall clock.
fn bench(args: &[String]) -> CliResult {
    const RUNS: usize = 5;
    let Opts { n, prompt, .. } = prompt_opts(args, 32, &[])?;
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
    let gw = GpuWeights::upload(&gpu, &w)?;
    let mut cache = KvCache::new(&gpu, &w.config);
    let argmax = gpt2::argmax_token;

    // One cached run: (prefill seconds, decode seconds for n tokens). The decode clock covers
    // everything a real step does: the model, the 201 KB readback and the argmax.
    let mut cached = || -> CliResult<(f64, f64)> {
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
    // The uncached path runs without a cache, as `generate --no-cache` does. With the cache's
    // 155 MiB still allocated, its temporaries stop fitting the allocator's blocks from T = 20
    // on, and each step then creates and frees a 256 MiB block: +60 ms per token (D63).
    drop(cache);

    let uncached = || -> CliResult<f64> {
        let t = Instant::now();
        gpt2_gpu::generate_greedy_uncached(&gpu, &gw, &ids, n + 1, |_| {})?;
        Ok(t.elapsed().as_secs_f64())
    };
    uncached()?;
    let full = median((0..RUNS).map(|_| uncached()).collect::<Result<_, _>>()?);

    println!(
        "{}; prompt {} tokens, {n} decode steps; median of {RUNS} after 1 warm-up, wall clock",
        device(&gpu),
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

/// The median of a non-empty set of runs.
fn median(xs: Vec<f64>) -> f64 {
    profile::median(xs).expect("at least one measured run")
}

/// One measured step: every kernel's GPU time (D36) and the step's wall-clock time.
struct Sample {
    kernels: Vec<KernelTime>,
    wall_ns: f64,
}

/// Run `step` with the profiler on. The wall clock stops when `step` returns, which for the
/// model means after its logits were read back, so the GPU has finished.
fn measure<T>(gpu: &Gpu, step: impl FnOnce() -> ember::Result<T>) -> CliResult<(T, Sample)> {
    gpu.profile_start()?;
    let t = Instant::now();
    let out = step()?;
    let wall_ns = t.elapsed().as_secs_f64() * 1e9;
    let kernels = gpu.profile_finish()?;
    Ok((out, Sample { kernels, wall_ns }))
}

/// `runs` samples of `step`, after one warm-up run that is thrown away.
fn sample_runs<T>(
    gpu: &Gpu,
    runs: usize,
    mut step: impl FnMut() -> ember::Result<T>,
) -> CliResult<Vec<Sample>> {
    let mut samples = Vec::with_capacity(runs);
    for run in 0..=runs {
        let (_, s) = measure(gpu, &mut step)?;
        if run > 0 {
            samples.push(s);
        }
    }
    Ok(samples)
}

/// Median GPU ms of the kernels one call of `step` dispatches.
fn gpu_ms<T>(gpu: &Gpu, runs: usize, step: impl FnMut() -> ember::Result<T>) -> CliResult<f64> {
    Ok(step_ms(&sample_runs(gpu, runs, step)?).0)
}

/// Elements in the bandwidth probes' buffer: 256 MiB of f32.
const PROBE_LEN: usize = 64 << 20;

/// `x` per millisecond, in units of 1e9 per second (GB/s for bytes, GFLOP/s for flops).
fn giga_per_s(x: f64, ms: f64) -> f64 {
    x / (ms * 1e6)
}

/// Median per-step GPU ms of the named kernels together across samples (0 if none ran).
fn kernel_ms(samples: &[Sample], kernels: &[&str]) -> f64 {
    let per_run = samples.iter().map(|s| {
        s.kernels
            .iter()
            .filter(|t| kernels.contains(&t.kernel))
            .map(|t| t.ns)
            .sum::<f64>()
    });
    median(per_run.collect()) / 1e6
}

/// Median total kernel ms and wall ms per step.
fn step_ms(samples: &[Sample]) -> (f64, f64) {
    let gpu = samples.iter().map(|s| s.kernels.iter().map(|t| t.ns).sum());
    let wall = samples.iter().map(|s| s.wall_ns);
    (median(gpu.collect()) / 1e6, median(wall.collect()) / 1e6)
}

/// The per-kernel table of D38: calls, median GPU ms and share of the step's wall clock.
fn print_table(title: &str, samples: &[Sample]) {
    let (gpu_ms, wall_ms) = step_ms(samples);
    println!("\n{title}");
    println!(
        "  {:<12} {:>6} {:>10} {:>8}",
        "kernel", "calls", "GPU ms", "% step"
    );
    let mut rows: Vec<(&str, usize, f64)> = profile::by_kernel(&samples[0].kernels)
        .into_iter()
        .map(|(k, calls, _)| (k, calls, kernel_ms(samples, &[k])))
        .collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    let calls: usize = rows.iter().map(|r| r.1).sum();
    for (k, c, ms) in rows {
        println!("  {k:<12} {c:>6} {ms:>10.3} {:>7.1}%", 100.0 * ms / wall_ms);
    }
    println!(
        "  {:<12} {calls:>6} {gpu_ms:>10.3} {:>7.1}%",
        "all kernels",
        100.0 * gpu_ms / wall_ms
    );
    println!(
        "  {:<12} {:>6} {wall_ms:>10.3}   (not in kernels: {:.3} ms of submits, copies, readback, CPU)",
        "wall clock",
        "",
        wall_ms - gpu_ms
    );
}

/// `ember profile [-n RUNS] <prompt>` (D36-D41): per-kernel tables for the prefill and one
/// decode step, a decode sweep over context positions, and the measured copy bandwidth. Every
/// number is the median of RUNS (default 5) after 1 warm-up.
fn profile_cmd(args: &[String]) -> CliResult {
    let Opts {
        n: runs, prompt, ..
    } = prompt_opts(args, 5, &[])?;
    let runs = measured_runs(runs)?;
    let (_, w, ids) = load_gpt2(&prompt)?;
    if ids.len() + 1 > w.config.n_ctx {
        return Err(format!("the prompt needs at most {} tokens", w.config.n_ctx - 1).into());
    }
    let gpu = Gpu::new()?;
    let gw = GpuWeights::upload(&gpu, &w)?;
    let mut cache = KvCache::new(&gpu, &w.config);
    let argmax = gpt2::argmax_token;
    println!(
        "{}; timestamp tick {} ns",
        device(&gpu),
        gpu.queue.get_timestamp_period()
    );
    println!(
        "prompt {:?} = {} tokens; median of {runs} after 1 warm-up; GPU time from timestamp queries, step time from the wall clock",
        prompt,
        ids.len()
    );

    // Prefill, then the first decode step right after it.
    let (mut prefill, mut decode) = (Vec::new(), Vec::new());
    for run in 0..=runs {
        cache.clear();
        let (next, p) = measure(&gpu, || {
            argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &ids)?)
        })?;
        let (_, d) = measure(&gpu, || gpt2_gpu::extend(&gpu, &gw, &mut cache, &[next]))?;
        if run > 0 {
            prefill.push(p);
            decode.push(d);
        }
    }
    print_table(&format!("prefill ({} tokens)", ids.len()), &prefill);
    print_table(&format!("decode step at position {}", ids.len()), &decode);

    // Context sweep (D40): fill the cache to position p once, then time one decode step there,
    // rewinding the cache to p at the start of each run (a length reset, no GPU work).
    println!("\ndecode vs context position");
    println!(
        "  {:>8} {:>10} {:>10} {:>12} {:>8}",
        "position", "wall ms", "GPU ms", "attention ms", "% attn"
    );
    let filler: Vec<u32> = ids.iter().copied().cycle().take(w.config.n_ctx).collect();
    for pos in [8, 128, 512, 1000] {
        if pos >= w.config.n_ctx {
            continue;
        }
        cache.clear();
        let next = argmax(&gpt2_gpu::extend(&gpu, &gw, &mut cache, &filler[..pos])?)?;
        let samples = sample_runs(&gpu, runs, || {
            cache.truncate(pos)?;
            gpt2_gpu::extend(&gpu, &gw, &mut cache, &[next])
        })?;
        let (gpu_ms, wall_ms) = step_ms(&samples);
        // Both passes of D63.
        let attn = kernel_ms(&samples, &["attention", "attention_combine"]);
        println!(
            "  {pos:>8} {wall_ms:>10.3} {gpu_ms:>10.3} {attn:>12.3} {:>7.1}%",
            100.0 * attn / gpu_ms
        );
    }

    // Bandwidth roofline (D41): a 256 MiB copy kernel reads and writes every byte once.
    let src = probe_buffer(&gpu)?;
    let copy_ms = gpu_ms(&gpu, runs, || ops::copy(&gpu, &src))?;
    let copy_gbs = giga_per_s(2.0 * (PROBE_LEN * 4) as f64, copy_ms);
    let (decode_gpu_ms, decode_wall_ms) = step_ms(&decode);
    // Bytes of weights a decode step reads: every tensor once, except `wpe`, of which it reads
    // one row. (`wte` is read whole by the tied LM head.)
    let bytes = 4.0 * (w.param_count() - w.wpe.len()) as f64;
    println!("\nbandwidth");
    println!(
        "  copy kernel     256 MiB read + 256 MiB written in {copy_ms:.3} ms = {copy_gbs:.1} GB/s (measured roofline)"
    );
    for (what, ms) in [
        ("decode kernels", decode_gpu_ms),
        ("decode wall", decode_wall_ms),
    ] {
        let gbs = giga_per_s(bytes, ms);
        println!(
            "  {what:<16}{:.0} MB of weights in {ms:.3} ms = {gbs:.1} GB/s = {:.0}% of the roofline",
            bytes / 1e6,
            100.0 * gbs / copy_gbs
        );
    }
    Ok(())
}

/// A probe buffer of `PROBE_LEN` distinct values.
fn probe_buffer(gpu: &Gpu) -> CliResult<GpuTensor> {
    let data = (0..PROBE_LEN).map(|i| i as f32).collect();
    Ok(gpu.upload(&Tensor::new(&[PROBE_LEN], data)?))
}

/// `ember matmul [-n RUNS]` (D47): GPU time of each linear kernel on GPT-2's shapes, median of
/// RUNS (default 5) after 1 warm-up. Prefill shapes report GFLOP/s against a measured compute
/// roof (`fma_peak`); one-row shapes report weight GB/s against the copy roofline (D41). Random
/// weights: no checkpoint needed.
fn matmul_cmd(args: &[String]) -> CliResult {
    let Opts { n, prompt, .. } = parse_opts(args, 5, &[])?;
    if !prompt.is_empty() {
        return Err(format!("matmul takes no prompt\n\n{USAGE}").into());
    }
    let runs = measured_runs(n)?;
    let gpu = Gpu::new()?;
    println!(
        "{}; median of {runs} after 1 warm-up; GPU time from timestamp queries",
        device(&gpu)
    );
    let time = |op: &dyn Fn() -> ember::Result<GpuTensor>| gpu_ms(&gpu, runs, op);

    // Compute roof: 4096 x 256 invocations, 32 chains x 2048 steps each = 137 GFLOP.
    let (groups, iters) = (4096u32, 2048u32);
    let peak_ms = time(&|| ops::fma_peak(&gpu, groups, iters))?;
    let peak_flops = 2.0 * (ops::FMA_PEAK_CHAINS * iters as usize * 256 * groups as usize) as f64;
    let peak = giga_per_s(peak_flops, peak_ms);
    // Bandwidth roofs (D41, D47).
    let src = probe_buffer(&gpu)?;
    let probe_bytes = (PROBE_LEN * 4) as f64;
    let copy_gbs = giga_per_s(2.0 * probe_bytes, time(&|| ops::copy(&gpu, &src))?);
    let read_gbs = giga_per_s(probe_bytes, time(&|| ops::read_peak(&gpu, &src))?);
    drop(src);
    println!("compute roof (fma_peak)  {peak:7.1} GFLOP/s");
    println!("copy (D41)               {copy_gbs:7.1} GB/s read + written");
    println!("read roof (read_peak)    {read_gbs:7.1} GB/s");

    // GPT-2's block matrices (in, out), and the LM head.
    let mats = [
        ("qkv", 768, 2304),
        ("attn_out", 768, 768),
        ("fc", 768, 3072),
        ("fc_out", 3072, 768),
        ("lm_head", 768, 50257),
    ];
    let mut rng = Rng::new(1);
    let mut random = |shape: &[usize]| -> CliResult<GpuTensor> {
        let n = shape.iter().product();
        Ok(gpu.upload(&Tensor::new(shape, rng.vec(n, -0.1, 0.1))?))
    };

    println!("\nprefill: GFLOP/s (% of the compute roof)");
    println!(
        "  {:<9} {:>5} {:>18} {:>18} {:>8}",
        "matrix", "T", "naive", "tiled", "speedup"
    );
    for &(name, n_in, n_out) in &mats {
        let w_out_in = random(&[n_out, n_in])?;
        let w_in_out = random(&[n_in, n_out])?;
        for t in [7, 128, 512] {
            if name == "lm_head" && t > 7 {
                continue; // generation runs the head on one row (D26)
            }
            let x = random(&[t, n_in])?;
            let flops = 2.0 * (t * n_in * n_out) as f64;
            let naive = time(&|| ops::linear_naive(&gpu, &x, &w_out_in, None))?;
            let tiled = time(&|| ops::linear(&gpu, &x, &w_in_out, None))?;
            let gf = |ms: f64| giga_per_s(flops, ms);
            println!(
                "  {name:<9} {t:>5} {:>8.1} ({:>4.1}%) {:>8.1} ({:>4.1}%) {:>7.1}x",
                gf(naive),
                100.0 * gf(naive) / peak,
                gf(tiled),
                100.0 * gf(tiled) / peak,
                naive / tiled
            );
        }
    }

    // Decode: one GPT-2 step's matrices, 12 distinct copies of each block matrix plus the head,
    // run in model order. Distinct copies matter: one 2-9 MB matrix timed over and over stays
    // in the last-level cache (shared with the CPU on this chip) and "beats" DRAM bandwidth.
    println!(
        "\ndecode (T = 1), one step's matrices in model order: weight GB/s (% of the read roof)"
    );
    println!(
        "  {:<9} {:>9} {:>18} {:>18} {:>8}",
        "matrix", "MB", "naive", "matvec", "speedup"
    );
    let n_layer = 12;
    let order: Vec<usize> = (0..n_layer)
        .flat_map(|_| 0..4)
        .chain(std::iter::once(4))
        .collect();
    let xs = [random(&[1, 768])?, random(&[1, 3072])?];
    let x_for = |n_in: usize| if n_in == 768 { &xs[0] } else { &xs[1] };
    // Median per-kind ms of a whole step, for one kernel and weight layout.
    let mut step = |naive: bool| -> CliResult<Vec<f64>> {
        let ws = order
            .iter()
            .map(|&m| {
                let (_, n_in, n_out) = mats[m];
                random(&if naive { [n_out, n_in] } else { [n_in, n_out] })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let samples = sample_runs(&gpu, runs, || {
            for (&m, w) in order.iter().zip(&ws) {
                let x = x_for(mats[m].1);
                if naive {
                    ops::linear_naive(&gpu, x, w, None)?;
                } else {
                    ops::linear(&gpu, x, w, None)?;
                }
            }
            Ok(())
        })?;
        let mut per_kind: Vec<Vec<f64>> = vec![Vec::new(); mats.len()];
        for s in &samples {
            let mut sums = vec![0.0; mats.len()];
            for (&m, k) in order.iter().zip(&s.kernels) {
                sums[m] += k.ns / 1e6;
            }
            for (v, x) in per_kind.iter_mut().zip(sums) {
                v.push(x);
            }
        }
        Ok(per_kind.into_iter().map(median).collect())
    };
    let naive = step(true)?;
    let matvec = step(false)?;
    let (mut naive_total, mut matvec_total, mut bytes_total) = (0.0, 0.0, 0.0);
    for (m, &(name, n_in, n_out)) in mats.iter().enumerate() {
        let reps = if name == "lm_head" { 1 } else { n_layer };
        let bytes = 4.0 * (reps * n_in * n_out) as f64;
        let gbs = |ms: f64| giga_per_s(bytes, ms);
        naive_total += naive[m];
        matvec_total += matvec[m];
        bytes_total += bytes;
        println!(
            "  {name:<9} {:>9.1} {:>8.1} ({:>4.0}%) {:>8.1} ({:>4.0}%) {:>7.1}x",
            bytes / 1e6,
            gbs(naive[m]),
            100.0 * gbs(naive[m]) / read_gbs,
            gbs(matvec[m]),
            100.0 * gbs(matvec[m]) / read_gbs,
            naive[m] / matvec[m]
        );
    }
    println!(
        "  step      {:>9.1} {:>8.1} ({:>4.0}%) {:>8.1} ({:>4.0}%) {:>7.1}x   ({naive_total:.2} ms vs {matvec_total:.2} ms)",
        bytes_total / 1e6,
        giga_per_s(bytes_total, naive_total),
        100.0 * giga_per_s(bytes_total, naive_total) / read_gbs,
        giga_per_s(bytes_total, matvec_total),
        100.0 * giga_per_s(bytes_total, matvec_total) / read_gbs,
        naive_total / matvec_total
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
    use super::{parse_opts, prompt_opts, take_utf8};

    fn args(s: &str) -> Vec<String> {
        s.split(' ')
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    }

    #[test]
    fn options_come_before_the_prompt_in_any_order() {
        let o = prompt_opts(
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
            assert!(prompt_opts(&args(bad), 20, &flags).is_err(), "`{bad}`");
        }
        // bench takes no flags: `--cpu` there is unknown.
        assert!(prompt_opts(&args("--cpu hi"), 32, &[]).is_err());
        // matmul takes options but no prompt.
        let o = parse_opts(&args("-n 3"), 5, &[]).unwrap();
        assert_eq!((o.n, o.prompt.as_str()), (3, ""));
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
