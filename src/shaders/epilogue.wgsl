// The end of every linear kernel (D62), prepended to matmul.wgsl, matvec.wgsl and
// matvec_rows.wgsl after gelu_fn.wgsl. A kernel passes each output's chunked-sum total here:
// the bias is added, then EPILOGUE applies GELU or adds the residual input. All three kernels
// finish the same way, so a row keeps the bits it has in any batch (D46, D33).
//
// The kernel declares `params` (with `has_bias`) and `b`; WGSL declarations may come in any
// order.

// 0: nothing, 1: GELU (fc), 2: add `res` (attn_out and fc_out). Must match ops::Epilogue.
override EPILOGUE: u32 = 0u;

// Read only when EPILOGUE is 2; otherwise the host binds some other read-only buffer.
@group(0) @binding(4) var<storage, read> res: array<f32>;

// Output element `idx` (flat), column `col`, from its total.
fn finish(idx: u32, col: u32, total: f32) -> f32 {
    var v = total;
    if (params.has_bias != 0u) {
        v += b[col];
    }
    if (EPILOGUE == 1u) {
        v = gelu(v);
    } else if (EPILOGUE == 2u) {
        // The order of the separate add kernel: residual + branch.
        v = res[idx] + v;
    }
    return v;
}
