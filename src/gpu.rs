//! The GPU context: one headless device and queue (D6), compiled kernels, upload and readback.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};

use wgpu::util::DeviceExt;

use crate::error::{Error, Result};
use crate::profile::{KernelTime, Profiler};
use crate::tensor::{GpuTensor, Tensor, numel};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// Optional features we asked for and got. Only `TIMESTAMP_QUERY` so far (the profiler, D36);
    /// nothing in the engine's results depends on it.
    pub features: wgpu::Features,
    pub(crate) kernels: Kernels,
    /// `Some` while profiling (D37). A `Mutex` because `Gpu` is shared (`&Gpu`, across test
    /// threads); uncontended, it costs nanoseconds per dispatch next to microseconds of work.
    profiler: Mutex<Option<Profiler>>,
    /// Buffers and bind groups created so far (D60). Only ever read as a difference.
    created: AtomicU64,
}

/// A compiled pipeline and the name the profiler reports it under.
pub(crate) struct Kernel {
    pub name: &'static str,
    pub pipeline: wgpu::ComputePipeline,
}

/// Every compute pipeline, compiled once at startup. Compiling WGSL to the driver's ISA takes
/// milliseconds; doing it per call would dwarf the kernels themselves.
const MATVEC: &str = include_str!("shaders/matvec.wgsl");
const MATVEC_ROWS: &str = include_str!("shaders/matvec_rows.wgsl");

pub(crate) struct Kernels {
    pub add: Kernel,
    pub gelu: Kernel,
    pub embed: Kernel,
    pub softmax: Kernel,
    pub layer_norm: Kernel,
    pub linear_naive: Kernel,
    // The linear kernels, each compiled once per epilogue (D62), indexed by `Epilogue::index`.
    pub matmul: [Kernel; 3],
    pub matvec: [Kernel; 3],
    pub matvec_wide: [Kernel; 3],
    pub matvec_split: [Kernel; 3],
    pub matvec_rows: [Kernel; 3],
    pub matvec_rows_wide: [Kernel; 3],
    pub matvec_rows_split: [Kernel; 3],
    pub attention: Kernel,
    pub kv_write: Kernel,
    pub copy: Kernel,
    pub fma_peak: Kernel,
    pub read_peak: Kernel,
}

impl Gpu {
    /// Blocking constructor for native code. Picks the high-performance adapter (a discrete
    /// GPU if there is one) and requests the adapter's real limits, not WebGPU's defaults (D6).
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                // Limit bucketing hides exact limits from web pages (anti-fingerprinting).
                // We are the trusted app and want the real limits (D6).
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| Error::NoAdapter(e.to_string()))?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ember"),
                required_limits: adapter.limits(),
                // Requested when the adapter has it, so profiling is possible; not required.
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Device(e.to_string()))?;
        let limits = device.limits();
        let features = device.features();
        let k = |name, src| compute_pipeline(&device, name, src, &[]);
        // The row kernels share reduce.wgsl's trees: same source text, prepended.
        let with_reduce = |src: &str| [include_str!("shaders/reduce.wgsl"), src].concat();
        // A linear kernel once per epilogue (D62): epilogue.wgsl and the GELU it uses prepended,
        // plus the kernel's own override constants. `names` are the profiler's labels.
        let gelu_fn = include_str!("shaders/gelu_fn.wgsl");
        let linear_src = |src: &str| [gelu_fn, include_str!("shaders/epilogue.wgsl"), src].concat();
        let linear = |names: [&'static str; 3], src: &str, constants: &[(&str, f64)]| {
            let src = linear_src(src);
            names.map(|name| {
                let epilogue = match name.split_once('+') {
                    None => 0.0,
                    Some((_, "gelu")) => 1.0,
                    Some(_) => 2.0,
                };
                let mut c = constants.to_vec();
                c.push(("EPILOGUE", epilogue));
                compute_pipeline(&device, name, &src, &c)
            })
        };
        // matvec.wgsl or matvec_rows.wgsl with their two override constants.
        let matvec = |names, src, slices: u32, lookahead: bool| {
            linear(
                names,
                src,
                &[
                    ("SLICES", slices as f64),
                    ("LOOKAHEAD", lookahead as u32 as f64),
                ],
            )
        };
        let kernels = Kernels {
            add: k("add", include_str!("shaders/add.wgsl")),
            gelu: k(
                "gelu",
                &[gelu_fn, include_str!("shaders/gelu.wgsl")].concat(),
            ),
            embed: k("embed", include_str!("shaders/embed.wgsl")),
            softmax: k(
                "softmax",
                &with_reduce(include_str!("shaders/softmax.wgsl")),
            ),
            layer_norm: k(
                "layer_norm",
                &with_reduce(include_str!("shaders/layer_norm.wgsl")),
            ),
            linear_naive: k("linear_naive", include_str!("shaders/linear_naive.wgsl")),
            matmul: linear(
                ["matmul", "matmul+gelu", "matmul+res"],
                include_str!("shaders/matmul.wgsl"),
                &[],
            ),
            // One source, three configurations (D48, D56): (slices, lookahead).
            matvec: matvec(["matvec", "matvec+gelu", "matvec+res"], MATVEC, 1, true),
            matvec_wide: matvec(
                ["matvec_wide", "matvec_wide+gelu", "matvec_wide+res"],
                MATVEC,
                1,
                false,
            ),
            matvec_split: matvec(
                ["matvec_split", "matvec_split+gelu", "matvec_split+res"],
                MATVEC,
                4,
                true,
            ),
            // The same three for 2-8 rows (D57).
            matvec_rows: matvec(
                ["matvec_rows", "matvec_rows+gelu", "matvec_rows+res"],
                MATVEC_ROWS,
                1,
                true,
            ),
            matvec_rows_wide: matvec(
                [
                    "matvec_rows_wide",
                    "matvec_rows_wide+gelu",
                    "matvec_rows_wide+res",
                ],
                MATVEC_ROWS,
                1,
                false,
            ),
            matvec_rows_split: matvec(
                [
                    "matvec_rows_split",
                    "matvec_rows_split+gelu",
                    "matvec_rows_split+res",
                ],
                MATVEC_ROWS,
                4,
                true,
            ),
            attention: k(
                "attention",
                &with_reduce(include_str!("shaders/attention.wgsl")),
            ),
            kv_write: k("kv_write", include_str!("shaders/kv_write.wgsl")),
            copy: k("copy", include_str!("shaders/copy.wgsl")),
            fma_peak: k("fma_peak", include_str!("shaders/fma_peak.wgsl")),
            read_peak: k("read_peak", include_str!("shaders/read_peak.wgsl")),
        };
        Ok(Gpu {
            device,
            queue,
            info,
            limits,
            features,
            kernels,
            profiler: Mutex::new(None),
            created: AtomicU64::new(0),
        })
    }

    /// Copy a host tensor into a new storage buffer.
    pub fn upload(&self, t: &Tensor) -> GpuTensor {
        self.count();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: nonempty_bytes(t.data()),
                usage: TENSOR_USAGE,
            });
        GpuTensor {
            shape: t.shape().to_vec(),
            buffer,
        }
    }

    /// A storage buffer of u32s (token ids). Not a `GpuTensor`: tensors are f32 (D7).
    pub fn upload_u32(&self, data: &[u32]) -> wgpu::Buffer {
        self.count();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("u32"),
                contents: nonempty_bytes(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Workgroups one dispatch dimension can hold.
    pub fn max_groups(&self) -> u32 {
        self.limits.max_compute_workgroups_per_dimension
    }

    /// An uninitialised-by-us (wgpu zero-fills it) tensor for a kernel to write into.
    pub fn alloc(&self, shape: &[usize]) -> GpuTensor {
        self.count();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_size(numel(shape)),
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        });
        GpuTensor {
            shape: shape.to_vec(),
            buffer,
        }
    }

    /// Copy a GPU tensor back to the host. Blocks until every queued kernel that writes it
    /// has finished, so this is also the synchronisation point.
    pub fn read(&self, t: &GpuTensor) -> Result<Tensor> {
        self.rec().read(t)
    }

    /// A MAP_READ buffer for `n` f32s, the last stop of a readback: storage buffers can't be
    /// mapped directly, so data goes storage -> this buffer -> host.
    pub(crate) fn staging(&self, n: usize) -> wgpu::Buffer {
        self.count();
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_size(n),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Buffers and bind groups this `Gpu` has created so far (D60). A step that creates
    /// nothing leaves it unchanged.
    pub fn created(&self) -> u64 {
        self.created.load(Ordering::Relaxed)
    }

    fn count(&self) {
        self.created.fetch_add(1, Ordering::Relaxed);
    }

    /// A recording whose bind groups and uniforms are made fresh and dropped after the submit:
    /// the one-op path.
    pub fn rec(&self) -> Rec<'_> {
        Rec::new(self, None)
    }

    /// A recording that takes bind groups and uniforms from `binds` and leaves new ones there
    /// (D59): the workspace path.
    pub fn rec_with<'a>(&'a self, binds: &'a mut Binds) -> Rec<'a> {
        Rec::new(self, Some(binds))
    }

    /// Start timing every dispatch (D36, D37), discarding anything recorded but not finished.
    /// Each window gets a fresh query set (two small buffers): negligible next to a model step.
    /// Fails if the device has no `TIMESTAMP_QUERY`.
    pub fn profile_start(&self) -> Result<()> {
        if !self.features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(Error::Device(
                "this adapter has no TIMESTAMP_QUERY; profiling needs it".into(),
            ));
        }
        *self.profiler.lock().unwrap() = Some(Profiler::new(&self.device));
        Ok(())
    }

    /// Stop timing and return each dispatch since `profile_start`, in order. Waits for the GPU.
    pub fn profile_finish(&self) -> Result<Vec<KernelTime>> {
        let profiler = self.profiler.lock().unwrap().take();
        match profiler {
            Some(p) => p.finish(&self.device, &self.queue),
            None => Err(Error::Input("profile_finish without profile_start".into())),
        }
    }
}

const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Buffer size for `n` f32s. Never 0: a zero-sized buffer can't be bound, and an empty tensor
/// still needs a valid binding (the kernel just never touches it).
fn byte_size(n: usize) -> u64 {
    (n.max(1) * 4) as u64
}

/// The bytes of `data`, or 4 zero bytes for an empty slice (for the same reason as `byte_size`).
fn nonempty_bytes<T: bytemuck::Pod>(data: &[T]) -> &[u8] {
    if data.is_empty() {
        &[0; 4]
    } else {
        bytemuck::cast_slice(data)
    }
}

/// Map the first `bytes` of a MAP_READ buffer and copy them out as `T`s. Waits for the GPU to
/// finish everything submitted so far, including the copy that filled `buf`.
pub(crate) fn map_read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buf: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>> {
    let (tx, rx) = mpsc::channel();
    buf.map_async(wgpu::MapMode::Read, ..bytes, move |r| {
        // The receiver outlives this callback (we block on it below).
        let _ = tx.send(r);
    });
    let readback = |e: &dyn std::fmt::Display| Error::Readback(e.to_string());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| readback(&e))?;
    rx.recv()
        .map_err(|e| readback(&e))?
        .map_err(|e| readback(&e))?;
    let data = {
        let view = buf.get_mapped_range(..bytes).map_err(|e| readback(&e))?;
        bytemuck::cast_slice::<u8, T>(&view).to_vec()
    };
    buf.unmap();
    Ok(data)
}

/// Storage buffers one kernel binds at most (the uniform comes after them).
const MAX_BUFFERS: usize = 6;

/// One recording (D58): dispatches and copies go into a single command encoder, submitted
/// once by `submit` or `read`. Each dispatch still gets its own compute pass, so the profiler
/// times it as before; wgpu puts the barriers between dependent dispatches.
pub struct Rec<'a> {
    pub(crate) gpu: &'a Gpu,
    enc: wgpu::CommandEncoder,
    binds: Option<&'a mut Binds>,
}

impl<'a> Rec<'a> {
    fn new(gpu: &'a Gpu, mut binds: Option<&'a mut Binds>) -> Self {
        if let Some(b) = binds.as_deref_mut() {
            b.recording += 1;
        }
        Rec {
            gpu,
            enc: gpu.device.create_command_encoder(&Default::default()),
            binds,
        }
    }

    /// Record one dispatch of `(x, y, z)` workgroups. `buffers` are bindings 0.., and `params`
    /// (the bytes of the kernel's uniform struct) the binding after them.
    pub(crate) fn dispatch(
        &mut self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        params: &[u8],
        (x, y, z): (u32, u32, u32),
    ) -> Result<()> {
        let group = match self.binds.as_deref_mut() {
            Some(binds) => binds.get(self.gpu, kernel, buffers, params)?,
            None => {
                let uniform = self.gpu.uniform(params);
                self.gpu.bind_group(kernel, buffers, &uniform)
            }
        };
        let mut profiler = self.gpu.profiler.lock().unwrap();
        let mut pass = self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kernel.name),
            timestamp_writes: profiler.as_mut().and_then(|p| p.next(kernel.name)),
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    /// Record a copy of `size` bytes from `src` at byte `offset` to the start of `dst`.
    /// Offsets and sizes must be multiples of 4.
    pub(crate) fn copy(&mut self, src: &wgpu::Buffer, offset: u64, dst: &wgpu::Buffer, size: u64) {
        self.enc.copy_buffer_to_buffer(src, offset, dst, 0, size);
    }

    /// Write `data` to the start of `buffer`. The write lands when this recording is submitted,
    /// before any of its commands: it is a queue operation, not a recorded one.
    pub(crate) fn write<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, data: &[T]) {
        self.gpu
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn submit(self) {
        self.gpu.queue.submit([self.enc.finish()]);
    }

    /// Submit, then copy `t` back to the host through a new staging buffer.
    pub fn read(self, t: &GpuTensor) -> Result<Tensor> {
        let staging = self.gpu.staging(t.len());
        self.read_via(t, &staging)
    }

    /// `read` through a staging buffer the caller keeps (MAP_READ, at least `t`'s size).
    pub(crate) fn read_via(mut self, t: &GpuTensor, staging: &wgpu::Buffer) -> Result<Tensor> {
        let n = t.len();
        if n == 0 {
            self.submit();
            return Tensor::new(&t.shape, Vec::new());
        }
        let size = (n * 4) as u64;
        self.copy(&t.buffer, 0, staging, size);
        let gpu = self.gpu;
        self.submit();
        let data = map_read(&gpu.device, staging, size)?;
        Tensor::new(&t.shape, data)
    }
}

/// Bind groups and their uniform buffers, kept across recordings (D59). An entry belongs to
/// one kernel and one list of bound buffers; a new token only rewrites the uniforms whose
/// contents changed. Fixed buffers make every entry reusable, so a steady-state step creates
/// nothing (D60).
#[derive(Default)]
pub struct Binds {
    entries: HashMap<BindKey, Bound>,
    /// Counts recordings, to catch an entry reused with different contents in one of them.
    recording: u64,
}

type BindKey = (wgpu::ComputePipeline, [Option<wgpu::Buffer>; MAX_BUFFERS]);

struct Bound {
    uniform: wgpu::Buffer,
    params: Vec<u8>,
    group: wgpu::BindGroup,
    /// The last recording that used this entry.
    recording: u64,
}

impl Binds {
    /// Entries made so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get(
        &mut self,
        gpu: &Gpu,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        params: &[u8],
    ) -> Result<wgpu::BindGroup> {
        if buffers.len() > MAX_BUFFERS {
            return Err(Error::Shape(format!(
                "{}: {} buffers, at most {MAX_BUFFERS}",
                kernel.name,
                buffers.len()
            )));
        }
        let mut bound: [Option<wgpu::Buffer>; MAX_BUFFERS] = Default::default();
        for (slot, b) in bound.iter_mut().zip(buffers) {
            *slot = Some((*b).clone());
        }
        let key = (kernel.pipeline.clone(), bound);
        if let Some(e) = self.entries.get_mut(&key) {
            if e.params != params {
                // The write would land before the whole submit, so the earlier dispatch in
                // this recording would see the new contents too.
                if e.recording == self.recording {
                    return Err(Error::Input(format!(
                        "{}: same buffers bound twice in one recording with different parameters",
                        kernel.name
                    )));
                }
                gpu.queue.write_buffer(&e.uniform, 0, params);
                e.params.clear();
                e.params.extend_from_slice(params);
            }
            e.recording = self.recording;
            return Ok(e.group.clone());
        }
        let uniform = gpu.uniform(params);
        let group = gpu.bind_group(kernel, buffers, &uniform);
        self.entries.insert(
            key,
            Bound {
                uniform,
                params: params.to_vec(),
                group: group.clone(),
                recording: self.recording,
            },
        );
        Ok(group)
    }
}

impl Gpu {
    /// A uniform buffer holding `params` (the bytes of a `#[repr(C)]` struct matching the WGSL).
    fn uniform(&self, params: &[u8]) -> wgpu::Buffer {
        self.count();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: params,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// `buffers` at bindings 0.., then `uniform`.
    fn bind_group(
        &self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.count();
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .copied()
            .chain([uniform])
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }
}

/// Compile `wgsl`'s `main` with values for its `override` constants (D48).
fn compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    constants: &[(&str, f64)],
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    });
    Kernel {
        name: label,
        pipeline,
    }
}
