// Rotary position embedding (D70), in place: the first n_rot heads (width d) of every row of
// x: [T, w] turn by their position. In a fused q|k|v row the query heads and then the key heads
// come first, so n_rot = n_head + n_kv_head rotates exactly q and k. Dimension i of a head pairs
// with i + d/2 ("rotate half"); row t sits at position start + t:
//   (a, b) -> (a cos - b sin, b cos + a sin),  cos/sin = table[start + t, i]
// The tables [n_pos, d/2] are computed once on the CPU in f64 (cpu::rope_tables), so CPU and GPU
// turn by the same f32 numbers; WGSL's own sin and cos are not accurate enough for large angles.
// One invocation per pair, grid-stride (D8). Each invocation reads both halves of its pair
// before writing either, and no two invocations share a pair, so in place is safe.

struct Params {
    n: u32,      // pairs: T * n_rot * d/2
    w: u32,      // row width of x
    half: u32,   // d / 2
    start: u32,  // absolute position of row 0
    n_rot: u32,  // heads to rotate per row
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let half = params.half;
    let per_row = params.n_rot * half;
    for (var p = gid.x; p < params.n; p += stride) {
        let t = p / per_row;
        let r = p % per_row;
        let head = r / half;
        let i = r % half;
        let a_at = t * params.w + head * 2u * half + i;
        let b_at = a_at + half;
        let tab = (params.start + t) * half + i;
        let c = cos_t[tab];
        let s = sin_t[tab];
        let a = x[a_at];
        let b = x[b_at];
        x[a_at] = a * c - b * s;
        x[b_at] = b * c + a * s;
    }
}
