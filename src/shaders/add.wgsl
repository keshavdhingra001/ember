// out[i] = a[i] + b[i] for i in 0..n.
//
// Grid-stride loop (D8): the host launches at most max_compute_workgroups_per_dimension
// workgroups, so each invocation may handle several elements, `stride` apart. Consecutive
// invocations touch consecutive addresses, so loads coalesce on every iteration.

struct Params {
    n: u32,
    // Padding to 16 bytes: uniform buffers are laid out in 16-byte units.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < params.n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
