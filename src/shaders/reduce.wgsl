// Workgroup reductions shared by the row kernels (softmax, layer_norm, attention), prepended to
// each of them at compile time (WGSL has no #include). The including shader defines `WG`, its
// workgroup size, a power of two.
//
// Every thread passes its partial result `v`; all threads get the total. A fixed halving tree
// (WG/2, WG/4, ..., 1) combines the partials, so the order, and the bits, never change (D4,
// D18). Must be called from uniform control flow: every thread reaches every barrier.

var<workgroup> partial: array<f32, WG>;

fn tree_max(l: u32, v: f32) -> f32 {
    partial[l] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] = max(partial[l], partial[l + s]);
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();  // everyone has read partial[0] before the next tree overwrites it
    return r;
}

fn tree_sum(l: u32, v: f32) -> f32 {
    partial[l] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (l < s) {
            partial[l] += partial[l + s];
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}
