//! The GPU context: one headless device and queue (D6), compiled kernels, upload and readback.

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
    pub matmul: Kernel,
    pub matvec: Kernel,
    pub matvec_wide: Kernel,
    pub matvec_split: Kernel,
    pub matvec_rows: Kernel,
    pub matvec_rows_wide: Kernel,
    pub matvec_rows_split: Kernel,
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
        // matvec.wgsl or matvec_rows.wgsl with their two override constants.
        let matvec = |name, src, slices: u32, lookahead: bool| {
            compute_pipeline(
                &device,
                name,
                src,
                &[
                    ("SLICES", slices as f64),
                    ("LOOKAHEAD", lookahead as u32 as f64),
                ],
            )
        };
        let kernels = Kernels {
            add: k("add", include_str!("shaders/add.wgsl")),
            gelu: k("gelu", include_str!("shaders/gelu.wgsl")),
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
            matmul: k("matmul", include_str!("shaders/matmul.wgsl")),
            // One source, three configurations (D48, D56): (slices, lookahead).
            matvec: matvec("matvec", MATVEC, 1, true),
            matvec_wide: matvec("matvec_wide", MATVEC, 1, false),
            matvec_split: matvec("matvec_split", MATVEC, 4, true),
            // The same three for 2-8 rows (D57).
            matvec_rows: matvec("matvec_rows", MATVEC_ROWS, 1, true),
            matvec_rows_wide: matvec("matvec_rows_wide", MATVEC_ROWS, 1, false),
            matvec_rows_split: matvec("matvec_rows_split", MATVEC_ROWS, 4, true),
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
        })
    }

    /// Copy a host tensor into a new storage buffer.
    pub fn upload(&self, t: &Tensor) -> GpuTensor {
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
    pub(crate) fn upload_u32(&self, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("u32"),
                contents: nonempty_bytes(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Workgroups one dispatch dimension can hold.
    pub fn max_groups(&self) -> u32 {
        self.limits.max_compute_workgroups_per_dimension
    }

    /// An uninitialised-by-us (wgpu zero-fills it) tensor for a kernel to write into.
    pub fn alloc(&self, shape: &[usize]) -> GpuTensor {
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
    ///
    /// Storage buffers can't be mapped directly; the data goes storage -> staging buffer
    /// (MAP_READ) -> host.
    pub fn read(&self, t: &GpuTensor) -> Result<Tensor> {
        let n = t.len();
        if n == 0 {
            return Tensor::new(&t.shape, Vec::new());
        }
        let size = (n * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let data = map_read(&self.device, &staging, size)?;
        Tensor::new(&t.shape, data)
    }

    /// Copy `size` bytes from `src` (starting at byte `offset`) to the start of `dst`, in queue
    /// order after every earlier dispatch. Offsets and sizes must be multiples of 4.
    pub(crate) fn copy(&self, src: &wgpu::Buffer, offset: u64, dst: &wgpu::Buffer, size: u64) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, offset, dst, 0, size);
        self.queue.submit([enc.finish()]);
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

    /// Record and submit one compute dispatch of `(x, y, z)` workgroups. Bindings are buffers
    /// in binding order.
    pub(crate) fn dispatch(
        &self,
        kernel: &Kernel,
        bindings: &[&wgpu::Buffer],
        (x, y, z): (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut profiler = self.profiler.lock().unwrap();
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(kernel.name),
                timestamp_writes: profiler.as_mut().and_then(|p| p.next(kernel.name)),
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        self.queue.submit([enc.finish()]);
    }

    /// A small uniform buffer holding `params` (a `#[repr(C)]` Pod struct matching the WGSL).
    pub(crate) fn uniform<T: bytemuck::Pod>(&self, params: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
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
