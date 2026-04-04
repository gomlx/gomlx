#include "gomlx_erf.h"

// ─── Fused Softmax ──────────────────────────────────────────────────────────
// One threadgroup per row. Two-pass: max then exp-sum-normalize.

constant uint SOFTMAX_TG_SIZE = 256;

kernel void fused_softmax_f32(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;

    device const float* row = src + tg_id * axis_size;
    device float* out_row = dst + tg_id * axis_size;

    threadgroup float shared[SOFTMAX_TG_SIZE];

    // Pass 1: find max
    float local_max = -HUGE_VALF;
    for (uint i = tid; i < axis_size; i += tg_size) {
        local_max = max(local_max, row[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < axis_size; i += tg_size) {
        float e = exp(row[i] - row_max);
        out_row[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];

    // Pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < axis_size; i += tg_size) {
        out_row[i] *= inv_sum;
    }
}

kernel void fused_softmax_f16(
    device const half* src    [[buffer(0)]],
    device half* dst          [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;

    device const half* row = src + tg_id * axis_size;
    device half* out_row = dst + tg_id * axis_size;
    threadgroup float shared[SOFTMAX_TG_SIZE];

    float local_max = -HUGE_VALF;
    for (uint i = tid; i < axis_size; i += tg_size) {
        local_max = max(local_max, float(row[i]));
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < axis_size; i += tg_size) {
        float e = exp(float(row[i]) - row_max);
        out_row[i] = half(e);
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    for (uint i = tid; i < axis_size; i += tg_size) {
        out_row[i] = half(float(out_row[i]) * inv_sum);
    }
}

// ─── Fused GELU ─────────────────────────────────────────────────────────────

// Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
kernel void fused_gelu_exact_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = src[id];
    dst[id] = x * 0.5f * (1.0f + gomlx_erf(x * M_SQRT1_2_F));
}

// Tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void fused_gelu_approx_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = src[id];
    float c = 0.7978845608f; // sqrt(2/pi)
    dst[id] = x * 0.5f * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

kernel void fused_gelu_exact_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(src[id]);
    dst[id] = half(x * 0.5f * (1.0f + gomlx_erf(x * M_SQRT1_2_F)));
}

kernel void fused_gelu_approx_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(src[id]);
    float c = 0.7978845608f;
    dst[id] = half(x * 0.5f * (1.0f + tanh(c * (x + 0.044715f * x * x * x))));
}

// ─── Fused LayerNorm ────────────────────────────────────────────────────────
// One threadgroup per batch element. Normalizes over norm_size elements.
// y = gamma * (x - mean) / sqrt(var + eps) + beta

constant uint NORM_TG_SIZE = 256;

kernel void fused_layernorm_f32(
    device const float* x         [[buffer(0)]],
    device const float* gamma     [[buffer(1)]],
    device const float* beta      [[buffer(2)]],
    device float* dst             [[buffer(3)]],
    constant uint& batch_size     [[buffer(4)]],
    constant uint& norm_size      [[buffer(5)]],
    constant float& epsilon       [[buffer(6)]],
    constant int& has_gamma       [[buffer(7)]],
    constant int& has_beta        [[buffer(8)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= batch_size) return;

    device const float* row = x + tg_id * norm_size;
    device float* out = dst + tg_id * norm_size;

    threadgroup float shared[NORM_TG_SIZE];
    threadgroup float shared2[NORM_TG_SIZE];

    // Compute mean
    float sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        sum += row[i];
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(norm_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float d = row[i] - mean;
        var_sum += d * d;
    }
    shared2[tid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared2[tid] += shared2[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared2[0] / float(norm_size) + epsilon);

    // Normalize + scale + shift
    for (uint i = tid; i < norm_size; i += tg_size) {
        float normed = (row[i] - mean) * inv_std;
        if (has_gamma) normed *= gamma[i];
        if (has_beta) normed += beta[i];
        out[i] = normed;
    }
}

kernel void fused_layernorm_f16(
    device const half* x         [[buffer(0)]],
    device const half* gamma     [[buffer(1)]],
    device const half* beta      [[buffer(2)]],
    device half* dst             [[buffer(3)]],
    constant uint& batch_size    [[buffer(4)]],
    constant uint& norm_size     [[buffer(5)]],
    constant float& epsilon      [[buffer(6)]],
    constant int& has_gamma      [[buffer(7)]],
    constant int& has_beta       [[buffer(8)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= batch_size) return;

    device const half* row = x + tg_id * norm_size;
    device half* out = dst + tg_id * norm_size;

    threadgroup float shared[NORM_TG_SIZE];
    threadgroup float shared2[NORM_TG_SIZE];

    float sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size)
        sum += float(row[i]);
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(norm_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var_sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float d = float(row[i]) - mean;
        var_sum += d * d;
    }
    shared2[tid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared2[tid] += shared2[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared2[0] / float(norm_size) + epsilon);

    for (uint i = tid; i < norm_size; i += tg_size) {
        float normed = (float(row[i]) - mean) * inv_std;
        if (has_gamma) normed *= float(gamma[i]);
        if (has_beta) normed += float(beta[i]);
        out[i] = half(normed);
    }
}

// ─── Fused RMSNorm ──────────────────────────────────────────────────────────
// y = weight * x / sqrt(mean(x^2) + eps)

kernel void fused_rmsnorm_f32(
    device const float* x         [[buffer(0)]],
    device const float* weight    [[buffer(1)]],
    device float* dst             [[buffer(2)]],
    constant uint& batch_size     [[buffer(3)]],
    constant uint& norm_size      [[buffer(4)]],
    constant float& epsilon       [[buffer(5)]],
    constant int& has_weight      [[buffer(6)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= batch_size) return;

    device const float* row = x + tg_id * norm_size;
    device float* out = dst + tg_id * norm_size;

    threadgroup float shared[NORM_TG_SIZE];

    float sq_sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float v = row[i];
        sq_sum += v * v;
    }
    shared[tid] = sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(norm_size) + epsilon);

    for (uint i = tid; i < norm_size; i += tg_size) {
        float normed = row[i] * inv_rms;
        if (has_weight) normed *= weight[i];
        out[i] = normed;
    }
}

kernel void fused_rmsnorm_f16(
    device const half* x         [[buffer(0)]],
    device const half* weight    [[buffer(1)]],
    device half* dst             [[buffer(2)]],
    constant uint& batch_size    [[buffer(3)]],
    constant uint& norm_size     [[buffer(4)]],
    constant float& epsilon      [[buffer(5)]],
    constant int& has_weight     [[buffer(6)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= batch_size) return;

    device const half* row = x + tg_id * norm_size;
    device half* out = dst + tg_id * norm_size;
    threadgroup float shared[NORM_TG_SIZE];

    float sq_sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float v = float(row[i]);
        sq_sum += v * v;
    }
    shared[tid] = sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(norm_size) + epsilon);

    for (uint i = tid; i < norm_size; i += tg_size) {
        float normed = float(row[i]) * inv_rms;
        if (has_weight) normed *= float(weight[i]);
        out[i] = half(normed);
    }
}
