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
    uint token = (row / p.cols) * 4, channel = row % p.cols;
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
    uint token = (row/p.cols)*4, channel = row%p.cols;
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
