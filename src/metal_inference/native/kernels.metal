// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n, rows, cols, k, heads, kv_heads, seq, batch, dim, group, u0, u1;
    float eps, theta, scale, reserved;
};

inline float bf16(ushort value) { return as_type<float>(uint(value) << 16); }
inline float weight4(device const uint *w, device const ushort *scales,
                     device const ushort *biases, uint row, uint col, uint k, uint group) {
    uint packed = w[row * (k / 8) + col / 8];
    uint code = (packed >> ((col % 8) * 4)) & 15;
    // The registered uint4 kernel uses group size 64. Constant shifts avoid
    // integer division in the innermost multiply loop.
    uint g = row * (k >> 6) + (col >> 6);
    return float(code) * bf16(scales[g]) + bf16(biases[g]);
}

kernel void embedding4(device const uint *ids [[buffer(0)]],
                       device const uint *w [[buffer(1)]],
                       device const ushort *s [[buffer(2)]],
                       device const ushort *b [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = weight4(w, s, b, ids[i / p.cols], i % p.cols, p.cols, p.group);
}

// A SIMD group reuses each decoded weight across four token rows. Accumulation
// order for each dot product is unchanged from the single-row reference kernel.
kernel void linear4(device const float *x [[buffer(0)]],
                    device const uint *w [[buffer(1)]],
                    device const ushort *s [[buffer(2)]],
                    device const ushort *b [[buffer(3)]],
                    device float *out [[buffer(4)]],
                    constant Params &p [[buffer(8)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_threadgroup]]) {
    uint token = p.n + (row / p.cols) * 4, channel = row % p.cols;
    float total[4] = {0.0f};
    for (uint col = lane; col < p.k; col += 32) {
        float weight = weight4(w, s, b, channel, col, p.k, p.group);
        for (uint t = 0; t < 4; ++t)
            if (token+t < p.rows) total[t] += x[(token+t)*p.k+col]*weight;
    }
    for (uint t = 0; t < 4; ++t) {
        float sum = simd_sum(total[t]);
        if (lane == 0 && token+t < p.rows) out[(token+t)*p.cols+channel] = sum;
    }
}

kernel void rms_norm(device const float *x [[buffer(0)]],
                     device const ushort *w [[buffer(1)]], device float *out [[buffer(2)]],
                     constant Params &p [[buffer(8)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
    float squares = 0.0f;
    for (uint c = lane; c < p.cols; c += 32) { float v = x[row*p.cols+c]; squares += v*v; }
    float inv = rsqrt(simd_sum(squares) / float(p.cols) + p.eps);
    for (uint c = lane; c < p.cols; c += 32)
        out[row*p.cols+c] = x[row*p.cols+c] * inv * bf16(w[c]);
}

kernel void rope(device float *x [[buffer(0)]], constant Params &p [[buffer(8)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint halfdim = p.dim / 2, row = i / halfdim, c = i % halfdim;
    uint pos = (row / p.heads) % p.seq;
    float angle = float(pos) * pow(p.theta, -2.0f * float(c) / float(p.dim));
    float a = x[row*p.dim+c], b = x[row*p.dim+c+halfdim];
    x[row*p.dim+c] = a*cos(angle) - b*sin(angle);
    x[row*p.dim+c+halfdim] = a*sin(angle) + b*cos(angle);
}

// Causal or bidirectional grouped-query attention with online softmax. No SxS scores
// allocation. Each SIMD group owns one query/head and up to 256 output channels.
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *k [[buffer(1)]],
                      device const float *v [[buffer(2)]],
                      device const uint *lengths [[buffer(3)]],
                      device float *out [[buffer(4)]], constant Params &p [[buffer(8)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]]) {
    uint head = row % p.heads, token = row / p.heads;
    uint batch = token / p.seq, position = token % p.seq;
    uint kvhead = head / (p.heads / p.kv_heads);
    float accum[8] = {0.0f}, maximum = -INFINITY, denominator = 0.0f;
    uint end = p.u0 ? lengths[batch] : min(position + 1, lengths[batch]);
    for (uint key = 0; key < end; ++key) {
        uint kvrow = (batch*p.seq + key)*p.kv_heads + kvhead;
        float dot = 0.0f;
        for (uint c = lane; c < p.dim; c += 32) dot += q[row*p.dim+c] * k[kvrow*p.dim+c];
        float score = simd_sum(dot) * p.scale;
        float next = max(maximum, score);
        float correction = exp(maximum-next), prob = exp(score-next);
        denominator = denominator*correction + prob;
        for (uint c = lane; c < p.dim; c += 32)
            accum[c/32] = accum[c/32]*correction + prob*v[kvrow*p.dim+c];
        maximum = next;
    }
    for (uint c = lane; c < p.dim; c += 32) out[row*p.dim+c] = accum[c/32]/denominator;
}

// Eight queries by 32 keys, F32 SIMD matrix products and online softmax.
// Dispatcher requires sequence alignment to 32 and head dimension 32 or 128.
// Arrays total 9312 bytes before compiler alignment; no sequence-squared allocation.
kernel void attention_tiled(device const float *q [[buffer(0)]],
                            device const float *k [[buffer(1)]],
                            device const float *v [[buffer(2)]],
                            device const uint *lengths [[buffer(3)]],
                            device float *out [[buffer(4)]], constant Params &p [[buffer(8)]],
                            uint tile [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[8*32], state[8*128], partial[8*128];
    threadgroup float maxima[8], denominators[8], corrections[8];
    uint head = tile % p.heads, block = tile / p.heads;
    uint batch = block / (p.seq/8), position = (block % (p.seq/8))*8;
    uint kvhead = head / (p.heads/p.kv_heads);
    for (uint i = tid; i < 8*p.dim; i += 128) state[i] = 0.0f;
    if (tid < 8) { maxima[tid] = -INFINITY; denominators[tid] = 0.0f; }
    uint end = p.u0 ? lengths[batch] : min(position+8, lengths[batch]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < end; base += 32) {
        simdgroup_float8x8 product(0.0f), left, right;
        for (uint c = 0; c < p.dim; c += 8) {
            simdgroup_load(left, q+((batch*p.seq+position)*p.heads+head)*p.dim+c,
                           p.heads*p.dim);
            simdgroup_load(right, k+((batch*p.seq+base+sg*8)*p.kv_heads+kvhead)*p.dim+c,
                           p.kv_heads*p.dim, ulong2(0), true);
            simdgroup_multiply_accumulate(product, left, right, product);
        }
        simdgroup_store(product, scores+sg*8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = sg; row < 8; row += 4) {
            uint limit = p.u0 ? lengths[batch] : min(position+row+1, lengths[batch]);
            float score = base+lane < limit ? scores[row*32+lane]*p.scale : -INFINITY;
            float maximum = max(maxima[row], simd_max(score));
            float correction = exp(maxima[row]-maximum);
            float probability = exp(score-maximum);
            float denominator = denominators[row]*correction+simd_sum(probability);
            scores[row*32+lane] = probability;
            if (lane == 0) {
                maxima[row] = maximum;
                denominators[row] = denominator;
                corrections[row] = correction;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = sg*8; c < p.dim; c += 32) {
            simdgroup_float8x8 value(0.0f);
            for (uint j = 0; j < 32; j += 8) {
                simdgroup_load(left, scores+j, 32);
                simdgroup_load(right, v+((batch*p.seq+base+j)*p.kv_heads+kvhead)*p.dim+c,
                               p.kv_heads*p.dim);
                simdgroup_multiply_accumulate(value, left, right, value);
            }
            simdgroup_store(value, partial+c, p.dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8*p.dim; i += 128)
            state[i] = state[i]*corrections[i/p.dim]+partial[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 8*p.dim; i += 128)
        out[((batch*p.seq+position+i/p.dim)*p.heads+head)*p.dim+i%p.dim] =
            state[i]/denominators[i/p.dim];
}

kernel void add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                device float *out [[buffer(2)]], constant Params &p [[buffer(8)]],
                uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = a[i] + b[i];
}

kernel void matmul_f32(device const float *a [[buffer(0)]],
                       device const float *bt [[buffer(1)]], device float *out [[buffer(2)]],
                       constant Params &p [[buffer(8)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    uint token = p.n + (row/p.cols)*4, channel = row%p.cols;
    float total[4] = {0.0f};
    for (uint col = lane; col < p.k; col += 32) {
        float value = bt[channel*p.k+col];
        for (uint t = 0; t < 4; ++t)
            if (token+t < p.rows) total[t] += a[(token+t)*p.k+col]*value;
    }
    for (uint t = 0; t < 4; ++t) {
        float sum = simd_sum(total[t]);
        if (lane == 0 && token+t < p.rows) out[(token+t)*p.cols+channel] = sum;
    }
}

kernel void transpose_f32(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                          constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.rows*p.cols) out[(i%p.cols)*p.rows+i/p.cols] = x[i];
}

kernel void silu_f32(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                     constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = x[i] / (1.0f+exp(-x[i]));
}

kernel void silu_gate(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                      constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) { float g = gate[i]; gate[i] = (g / (1.0f+exp(-g))) * up[i]; }
}

kernel void pool_project(device const float *x [[buffer(0)]],
                         device const uint *lengths [[buffer(1)]],
                         device float *out [[buffer(2)]], constant Params &p [[buffer(8)]],
                         uint batch [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    uint start = (batch*p.seq + (p.u1 ? 0 : lengths[batch]-1))*p.cols;
    float squares = 0.0f;
    for (uint c = lane; c < p.dim; c += 32) { float v = x[start+c]; squares += v*v; }
    float inv = rsqrt(simd_sum(squares));
    for (uint c = lane; c < p.dim; c += 32) out[batch*p.dim+c] = x[start+c]*inv;
}

kernel void embedding_position(device const uint *ids [[buffer(0)]],
                               device const float *words [[buffer(1)]],
                               device const float *positions [[buffer(2)]],
                               device const float *types [[buffer(3)]],
                               device float *out [[buffer(4)]],
                               constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint token = i/p.cols, c = i%p.cols;
    out[i] = (words[ids[token]*p.cols+c] + types[c]) + positions[(token%p.seq)*p.cols+c];
}

kernel void layer_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                       device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],
                       constant Params &p [[buffer(8)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    float total = 0.0f;
    for (uint c = lane; c < p.cols; c += 32) total += x[row*p.cols+c];
    float mean = simd_sum(total)/float(p.cols), squares = 0.0f;
    for (uint c = lane; c < p.cols; c += 32) { float d = x[row*p.cols+c]-mean; squares += d*d; }
    float inv = rsqrt(simd_sum(squares)/float(p.cols)+p.eps);
    for (uint c = lane; c < p.cols; c += 32)
        out[row*p.cols+c] = (x[row*p.cols+c]-mean)*inv*w[c]+bias[c];
}

kernel void add_bias(device float *x [[buffer(0)]], device const float *bias [[buffer(1)]],
                     constant Params &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) x[i] += bias[i%p.cols];
}

kernel void gelu_f32(device float *x [[buffer(0)]], constant Params &p [[buffer(8)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    // erf-form GELU, using the A&S 7.1.26 approximation because Metal has no erf.
    // Coefficients are mathematical constants; numerical error is checked against math.erf.
    float value = x[i], z = fabs(value)*M_SQRT1_2_F;
    float t = 1.0f/(1.0f+0.3275911f*z);
    float poly = ((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
    float tail = poly*exp(-z*z);
    x[i] = 0.5f*value*(value < 0.0f ? tail : 2.0f-tail);
}

// Original 8x32 output tile: four SIMDgroups each own one 8x8 fragment.
// F32 inputs and accumulators; weights remain [N,K]. Only aligned shapes are
// routed here, so all collective loads/stores are complete and in bounds.
kernel void matmul_f32_tiled(device const float *a [[buffer(0)]],
                             device const float *bt [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant Params &p [[buffer(8)]],
                             uint tile [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*8;
    uint col = (tile % (p.cols/32))*32 + sg*8;
    simdgroup_float8x8 accum(0.0f), left, right;
    for (uint k = 0; k < p.k; k += 8) {
        simdgroup_load(left, a + row*p.k + k, p.k);
        simdgroup_load(right, bt + col*p.k + k, p.k, ulong2(0), true);
        simdgroup_multiply_accumulate(accum, left, right, accum);
    }
    simdgroup_store(accum, out + row*p.cols + col, p.cols);
}

// Full 8x32 tile, F32 partial sums reset every 32 products to bound long-K
// accumulation error. Host restricts this kernel to verified BGE projection
// shapes and complete row tiles. Remaining rows use the scalar reduction.
kernel void matmul_f32_chunk32(device const float *a [[buffer(0)]],
                             device const float *bt [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant Params &p [[buffer(8)]],
                             uint tile [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*8;
    uint col = (tile % (p.cols/32))*32 + sg*8;
    simdgroup_float8x8 accum(0.0f), left, right;
    for (uint base = 0; base < p.k; base += 32) {
        simdgroup_float8x8 partial(0.0f);
        for (uint j = 0; j < 32; j += 8) {
            simdgroup_load(left, a + row*p.k + base+j, p.k);
            simdgroup_load(right, bt + col*p.k + base+j, p.k, ulong2(0), true);
            simdgroup_multiply_accumulate(partial, left, right, partial);
        }
        for (uint e = 0; e < 2; ++e) accum.thread_elements()[e] += partial.thread_elements()[e];
    }
    simdgroup_store(accum, out + row*p.cols + col, p.cols);
}

// Original 8x32 output tile. Decode 32x32 weights into 4 KiB shared memory;
// never materialize the model's full F32 weights. All shapes must be aligned.
kernel void linear4_tiled(device const float *x [[buffer(0)]],
                          device const uint *w [[buffer(1)]],
                          device const ushort *s [[buffer(2)]],
                          device const ushort *b [[buffer(3)]],
                          device float *out [[buffer(4)]],
                          constant Params &p [[buffer(8)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*8, channel = (tile % (p.cols/32))*32;
    threadgroup float weights[32*32];
    simdgroup_float8x8 accum(0.0f), left, right;
    for (uint base = 0; base < p.k; base += 32) {
        uint c = channel + tid/4, col = base + (tid%4)*8;
        uint packed = w[c*(p.k/8)+col/8], g = c*(p.k/64)+col/64;
        float scale = bf16(s[g]), bias = bf16(b[g]);
        for (uint j = 0; j < 8; ++j)
            weights[tid*8+j] = float((packed >> (j*4)) & 15)*scale+bias;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 partial(0.0f);
        for (uint j = 0; j < 32; j += 8) {
            simdgroup_load(left, x + row*p.k + base+j, p.k);
            simdgroup_load(right, weights + sg*8*32+j, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(partial, left, right, partial);
        }
        for (uint e = 0; e < 2; ++e) accum.thread_elements()[e] += partial.thread_elements()[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum, out + row*p.cols + channel+sg*8, p.cols);
}

// Full 16x32 output tile: eight SIMD groups share 8 KiB of decoded weights.
// Load 64 K values per step, but sum independent 32-value F32 partials in the
// same order as linear4_tiled. Host dispatch requires complete aligned tiles.
kernel void linear4_16x32_k64(device const float *x [[buffer(0)]],
                             device const uint *w [[buffer(1)]],
                             device const ushort *s [[buffer(2)]],
                             device const ushort *b [[buffer(3)]],
                             device float *out [[buffer(4)]],
                             constant Params &p [[buffer(8)]],
                             uint tile [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = (tile / (p.cols/32))*16 + (sg/4)*8;
    uint channel = (tile % (p.cols/32))*32;
    threadgroup float weights[32*64];
    simdgroup_float8x8 accum(0.0f), left, right;
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
    simdgroup_store(accum, out+row*p.cols+channel+(sg%4)*8, p.cols);
}

// Partial row tiles use zero-filled shared input and bounded output stores.
// Channels and K retain the same alignment contract as linear4_tiled.
kernel void linear4_tail(device const float *x [[buffer(0)]],
                          device const uint *w [[buffer(1)]],
                          device const ushort *s [[buffer(2)]],
                          device const ushort *b [[buffer(3)]],
                          device float *out [[buffer(4)]],
                          constant Params &p [[buffer(8)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = p.n + (tile / (p.cols/32))*8, channel = (tile % (p.cols/32))*32;
    threadgroup float weights[32*32];
    threadgroup float inputs[8*32], result[8*32];
    bool tail = row + 8 > p.rows;
    simdgroup_float8x8 accum(0.0f), left, right;
    for (uint base = 0; base < p.k; base += 32) {
        uint c = channel + tid/4, col = base + (tid%4)*8;
        uint packed = w[c*(p.k/8)+col/8], g = c*(p.k/64)+col/64;
        float scale = bf16(s[g]), bias = bf16(b[g]);
        for (uint j = 0; j < 8; ++j)
            weights[tid*8+j] = float((packed >> (j*4)) & 15)*scale+bias;
        if (tail) {
            for (uint i = tid; i < 8*32; i += 128)
                inputs[i] = row + i/32 < p.rows ? x[(row+i/32)*p.k+base+i%32] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 partial(0.0f);
        for (uint j = 0; j < 32; j += 8) {
            if (tail) simdgroup_load(left, inputs+j, 32);
            else simdgroup_load(left, x + row*p.k + base+j, p.k);
            simdgroup_load(right, weights + sg*8*32+j, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(partial, left, right, partial);
        }
        for (uint e = 0; e < 2; ++e) accum.thread_elements()[e] += partial.thread_elements()[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tail) {
        simdgroup_store(accum, result+sg*8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8*32; i += 128)
            if (row+i/32 < p.rows) out[(row+i/32)*p.cols+channel+i%32] = result[i];
    } else {
        simdgroup_store(accum, out + row*p.cols + channel+sg*8, p.cols);
    }
}
