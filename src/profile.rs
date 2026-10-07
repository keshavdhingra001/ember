//! GPU kernel timing with timestamp queries (D36, D37). Opt-in: `Gpu::profile_start` turns it
//! on, every dispatch then brackets its compute pass with two timestamps, and
//! `Gpu::profile_finish` returns each kernel's GPU time. Nothing the engine computes depends on
//! it (D4): it reads the GPU's clock, never the CPU's, and only for the report.

use crate::error::{Error, Result};
use crate::gpu::map_read;

/// Dispatches one profiling window can hold: WebGPU caps a query set at 4096 queries, two per
/// dispatch. A GPT-2 124M step is 135 dispatches.
pub const MAX_DISPATCHES: u32 = wgpu::QUERY_SET_MAX_QUERIES / 2;

/// One timed dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelTime {
    pub kernel: &'static str,
    pub ns: f64,
}

/// One profiling window: the query set and the buffers its timestamps travel through.
pub(crate) struct Profiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Kernel name of each recorded dispatch; dispatch i owns queries 2i and 2i + 1.
    labels: Vec<&'static str>,
    /// Dispatches that didn't fit. A non-zero count fails `finish` instead of silently
    /// under-reporting.
    dropped: usize,
}

impl Profiler {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let bytes = 2 * MAX_DISPATCHES as u64 * wgpu::QUERY_SIZE as u64;
        Profiler {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * MAX_DISPATCHES,
            }),
            // Query results can only be resolved into a QUERY_RESOLVE buffer, which can't be
            // mapped; they are copied to a MAP_READ buffer from there.
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profile resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profile readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            labels: Vec::new(),
            dropped: 0,
        }
    }

    /// Timestamp writes for the next dispatch, or `None` if the window is full.
    pub(crate) fn next(
        &mut self,
        kernel: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let i = self.labels.len() as u32;
        if i >= MAX_DISPATCHES {
            self.dropped += 1;
            return None;
        }
        self.labels.push(kernel);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(2 * i),
            end_of_pass_write_index: Some(2 * i + 1),
        })
    }

    /// Resolve every recorded pair, read it back and convert ticks to ns. Waits for the GPU.
    pub(crate) fn finish(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<KernelTime>> {
        if self.dropped > 0 {
            return Err(Error::Input(format!(
                "profile window overflowed: {} dispatches past the {MAX_DISPATCHES} limit",
                self.dropped
            )));
        }
        let n = self.labels.len() as u32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = 2 * n as u64 * wgpu::QUERY_SIZE as u64;
        let mut enc = device.create_command_encoder(&Default::default());
        enc.resolve_query_set(&self.queries, 0..2 * n, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, bytes);
        queue.submit([enc.finish()]);

        let ticks: Vec<u64> = map_read(device, &self.staging, bytes)?;

        let period = queue.get_timestamp_period() as f64;
        let times = self
            .labels
            .iter()
            .zip(ticks.as_chunks::<2>().0)
            .map(|(&kernel, t)| KernelTime {
                kernel,
                // wrapping: Vulkan timestamps may have fewer than 64 valid bits and wrap.
                ns: t[1].wrapping_sub(t[0]) as f64 * period,
            })
            .collect();
        Ok(times)
    }
}

/// Per-kernel totals for one run, in first-seen order: `(kernel, calls, ns)`.
pub fn by_kernel(times: &[KernelTime]) -> Vec<(&'static str, usize, f64)> {
    let mut out: Vec<(&'static str, usize, f64)> = Vec::new();
    for t in times {
        match out.iter_mut().find(|(k, _, _)| *k == t.kernel) {
            Some((_, calls, ns)) => {
                *calls += 1;
                *ns += t.ns;
            }
            None => out.push((t.kernel, 1, t.ns)),
        }
    }
    out
}

/// The median (upper middle for an even count). `None` for no values.
pub fn median(mut xs: Vec<f64>) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    Some(xs[xs.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(kernel: &'static str, ns: f64) -> KernelTime {
        KernelTime { kernel, ns }
    }

    #[test]
    fn groups_by_kernel_in_first_seen_order() {
        let times = [t("ln", 1.0), t("linear", 10.0), t("ln", 2.0), t("add", 0.5)];
        assert_eq!(
            by_kernel(&times),
            vec![("ln", 2, 3.0), ("linear", 1, 10.0), ("add", 1, 0.5)]
        );
        assert!(by_kernel(&[]).is_empty());
    }

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(vec![5.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(3.0));
        assert_eq!(median(vec![]), None);
    }
}
