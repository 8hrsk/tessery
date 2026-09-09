// SPDX-License-Identifier: Apache-2.0
// Experimental only. Append to the production shader source in memory.
// Complete 16x32 output tiles, uint4/BF16 group64, unchanged K32 partial sums.
kernel void fused_mlp_serial(device const float *x [[buffer(0)]],
    device const uint *w [[buffer(1)]], device const ushort *s [[buffer(2)]],
    device const ushort *b [[buffer(3)]], device const uint *uw [[buffer(4)]],
    device const ushort *us [[buffer(5)]], device const ushort *ub [[buffer(6)]],
    device float *out [[buffer(7)]], constant Params &p [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*16 + (sg/4)*8;
    uint channel = (tile % (p.cols/32))*32;
    threadgroup float weights[32*64];
    simdgroup_float8x8 accum(0.0f), up(0.0f), left, right;
    for (uint base = 0; base < p.k; base += 64) {
        uint c = channel+tid/8, col = base+(tid%8)*8;
        uint packed = w[c*(p.k/8)+col/8], g = c*(p.k/64)+col/64;
        float scale = bf16(s[g]), bias = bf16(b[g]);
        for (uint j = 0; j < 8; ++j)
            weights[tid*8+j] = float((packed >> (j*4)) & 15)*scale+bias;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint part = 0; part < 64; part += 32) {
            simdgroup_float8x8 partial(0.0f);
            for (uint j = 0; j < 32; j += 8) {
                simdgroup_load(left, x+row*p.k+base+part+j, p.k);
                simdgroup_load(right, weights+(sg%4)*8*64+part+j, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(partial, left, right, partial);
            }
            for (uint e = 0; e < 2; ++e)
                accum.thread_elements()[e] += partial.thread_elements()[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint base = 0; base < p.k; base += 64) {
        uint c = channel+tid/8, col = base+(tid%8)*8;
        uint packed = uw[c*(p.k/8)+col/8], g = c*(p.k/64)+col/64;
        float scale = bf16(us[g]), bias = bf16(ub[g]);
        for (uint j = 0; j < 8; ++j)
            weights[tid*8+j] = float((packed >> (j*4)) & 15)*scale+bias;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint part = 0; part < 64; part += 32) {
            simdgroup_float8x8 partial(0.0f);
            for (uint j = 0; j < 32; j += 8) {
                simdgroup_load(left, x+row*p.k+base+part+j, p.k);
                simdgroup_load(right, weights+(sg%4)*8*64+part+j, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(partial, left, right, partial);
            }
            for (uint e = 0; e < 2; ++e)
                up.thread_elements()[e] += partial.thread_elements()[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = 0; e < 2; ++e) {
        float g = accum.thread_elements()[e];
        accum.thread_elements()[e] = (g / (1.0f+exp(-g))) * up.thread_elements()[e];
    }
    simdgroup_store(accum, out+row*p.cols+channel+(sg%4)*8, p.cols);
}

kernel void fused_mlp_parallel(device const float *x [[buffer(0)]],
    device const uint *w [[buffer(1)]], device const ushort *s [[buffer(2)]],
    device const ushort *b [[buffer(3)]], device const uint *uw [[buffer(4)]],
    device const ushort *us [[buffer(5)]], device const ushort *ub [[buffer(6)]],
    device float *out [[buffer(7)]], constant Params &p [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*16 + (sg/4)*8;
    uint channel = (tile % (p.cols/32))*32;
    threadgroup float weights[32*64], up_weights[32*64];
    simdgroup_float8x8 accum(0.0f), up(0.0f), left, right;
    for (uint base = 0; base < p.k; base += 64) {
        uint c = channel+tid/8, col = base+(tid%8)*8;
        uint packed = w[c*(p.k/8)+col/8], g = c*(p.k/64)+col/64;
        uint up_packed = uw[c*(p.k/8)+col/8];
        float scale = bf16(s[g]), bias = bf16(b[g]);
        float up_scale = bf16(us[g]), up_bias = bf16(ub[g]);
        for (uint j = 0; j < 8; ++j) {
            weights[tid*8+j] = float((packed >> (j*4)) & 15)*scale+bias;
            up_weights[tid*8+j] = float((up_packed >> (j*4)) & 15)*up_scale+up_bias;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint part = 0; part < 64; part += 32) {
            simdgroup_float8x8 partial(0.0f), up_partial(0.0f);
            for (uint j = 0; j < 32; j += 8) {
                simdgroup_load(left, x+row*p.k+base+part+j, p.k);
                simdgroup_load(right, weights+(sg%4)*8*64+part+j, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(partial, left, right, partial);
                simdgroup_load(right, up_weights+(sg%4)*8*64+part+j, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(up_partial, left, right, up_partial);
            }
            for (uint e = 0; e < 2; ++e) {
                accum.thread_elements()[e] += partial.thread_elements()[e];
                up.thread_elements()[e] += up_partial.thread_elements()[e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = 0; e < 2; ++e) {
        float g = accum.thread_elements()[e];
        accum.thread_elements()[e] = (g / (1.0f+exp(-g))) * up.thread_elements()[e];
    }
    simdgroup_store(accum, out+row*p.cols+channel+(sg%4)*8, p.cols);
}
