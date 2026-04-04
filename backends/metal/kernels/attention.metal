#include <metal_stdlib>
using namespace metal;

// ─── Fused Scaled Dot-Product Attention ─────────────────────────────────────
// Q[b,h,s,d] K[b,hkv,t,d] V[b,hkv,t,d] -> O[b,h,s,d]
// Supports GQA (num_heads % num_kv_heads == 0), causal masking, additive mask.
//
// Each threadgroup computes one (batch, head, query_pos) output vector.
// Block-wise numerically stable softmax (online log-sum-exp).

constant uint ATTN_TG_SIZE = 256;

// mask_type: 0 = none, 1 = boolean, 2 = additive float
kernel void fused_sdpa_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device const float* mask      [[buffer(3)]],
    device float* O               [[buffer(4)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& num_heads      [[buffer(6)]],
    constant uint& num_kv_heads   [[buffer(7)]],
    constant uint& seq_len        [[buffer(8)]],
    constant uint& kv_len         [[buffer(9)]],
    constant uint& head_dim       [[buffer(10)]],
    constant float& scale              [[buffer(11)]],
    constant int& causal               [[buffer(12)]],
    constant int& mask_type            [[buffer(13)]],
    constant uint& mask_batch_stride   [[buffer(14)]],
    constant uint& mask_head_stride    [[buffer(15)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Decode threadgroup ID into (b, h, s)
    uint total = batch * num_heads * seq_len;
    if (tg_id >= total) return;

    uint s = tg_id % seq_len;
    uint tmp = tg_id / seq_len;
    uint h = tmp % num_heads;
    uint b = tmp / num_heads;

    uint kv_h = h / (num_heads / num_kv_heads); // GQA head mapping

    // Pointers for this (b, h, s) / (b, kv_h)
    device const float* q_vec = Q + ((b * num_heads + h) * seq_len + s) * head_dim;
    device const float* k_base = K + (b * num_kv_heads + kv_h) * kv_len * head_dim;
    device const float* v_base = V + (b * num_kv_heads + kv_h) * kv_len * head_dim;
    device float* o_vec = O + ((b * num_heads + h) * seq_len + s) * head_dim;

    // Online softmax: running max and sum
    threadgroup float shared_max[ATTN_TG_SIZE];
    threadgroup float shared_sum[ATTN_TG_SIZE];

    float local_max = -HUGE_VALF;
    float local_sum = 0.0f;

    // We accumulate weighted V in registers (each thread handles a subset of kv positions)
    // For simplicity, we do two passes: first compute softmax weights, then accumulate V.

    // Pass 1: compute scores and online softmax statistics
    // Each thread processes a strided subset of kv positions
    for (uint t = tid; t < kv_len; t += tg_size) {
        // Causal: skip future positions
        if (causal && t > s) continue;

        // Dot product q . k[t]
        device const float* k_vec = k_base + t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k_vec[d];
        }
        dot *= scale;

        // Apply mask (broadcastable [batch, heads?, seq, kv] row-major)
        if (mask_type != 0) {
            uint mask_el = b * mask_batch_stride + h * mask_head_stride + s * kv_len + t;
            if (mask_type == 1) { // boolean (0/1 float/half materialized)
                if (mask[mask_el] == 0.0f) dot = -HUGE_VALF;
            } else if (mask_type == 2) { // additive
                dot += mask[mask_el];
            }
        }

        // Online softmax update
        if (dot > local_max) {
            float correction = exp(local_max - dot);
            local_sum = local_sum * correction + exp(0.0f); // exp(dot - dot) = 1
            local_max = dot;
        } else {
            local_sum += exp(dot - local_max);
        }
    }

    // Reduce max across threadgroup
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tg_size / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) shared_max[tid] = max(shared_max[tid], shared_max[tid + s2]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Correct local_sum to global max
    local_sum *= exp(local_max - global_max);

    // Reduce sum
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tg_size / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) shared_sum[tid] += shared_sum[tid + s2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Pass 2: accumulate weighted V
    // Each thread writes to its portion of the output head_dim vector.
    // We iterate over all kv positions and accumulate.
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < kv_len; t++) {
            if (causal && t > s) break;

            // Recompute score for position t
            device const float* k_vec = k_base + t * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += q_vec[dd] * k_vec[dd];
            }
            dot *= scale;

            if (mask_type != 0) {
                uint mask_el = b * mask_batch_stride + h * mask_head_stride + s * kv_len + t;
                if (mask_type == 1) {
                    if (mask[mask_el] == 0.0f) continue;
                } else if (mask_type == 2) {
                    dot += mask[mask_el];
                }
            }

            float w = exp(dot - global_max) * inv_sum;
            acc += w * v_base[t * head_dim + d];
        }
        o_vec[d] = acc;
    }
}

// FP16 version — accumulates in FP32 for numerical stability
kernel void fused_sdpa_f16(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* V         [[buffer(2)]],
    device const half* mask      [[buffer(3)]],
    device half* O               [[buffer(4)]],
    constant uint& batch          [[buffer(5)]],
    constant uint& num_heads      [[buffer(6)]],
    constant uint& num_kv_heads   [[buffer(7)]],
    constant uint& seq_len        [[buffer(8)]],
    constant uint& kv_len         [[buffer(9)]],
    constant uint& head_dim       [[buffer(10)]],
    constant float& scale              [[buffer(11)]],
    constant int& causal               [[buffer(12)]],
    constant int& mask_type            [[buffer(13)]],
    constant uint& mask_batch_stride   [[buffer(14)]],
    constant uint& mask_head_stride    [[buffer(15)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint total = batch * num_heads * seq_len;
    if (tg_id >= total) return;

    uint s = tg_id % seq_len;
    uint tmp = tg_id / seq_len;
    uint h = tmp % num_heads;
    uint b = tmp / num_heads;
    uint kv_h = h / (num_heads / num_kv_heads);

    device const half* q_vec = Q + ((b * num_heads + h) * seq_len + s) * head_dim;
    device const half* k_base = K + (b * num_kv_heads + kv_h) * kv_len * head_dim;
    device const half* v_base = V + (b * num_kv_heads + kv_h) * kv_len * head_dim;
    device half* o_vec = O + ((b * num_heads + h) * seq_len + s) * head_dim;

    threadgroup float shared_max[ATTN_TG_SIZE];
    threadgroup float shared_sum[ATTN_TG_SIZE];

    float local_max = -HUGE_VALF;
    float local_sum = 0.0f;

    for (uint t = tid; t < kv_len; t += tg_size) {
        if (causal && t > s) continue;
        device const half* k_vec = k_base + t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++)
            dot += float(q_vec[d]) * float(k_vec[d]);
        dot *= scale;

        if (mask_type != 0) {
            uint mask_el = b * mask_batch_stride + h * mask_head_stride + s * kv_len + t;
            if (mask_type == 1) {
                if (float(mask[mask_el]) == 0.0f) dot = -HUGE_VALF;
            } else if (mask_type == 2) {
                dot += float(mask[mask_el]);
            }
        }

        if (dot > local_max) {
            local_sum = local_sum * exp(local_max - dot) + 1.0f;
            local_max = dot;
        } else {
            local_sum += exp(dot - local_max);
        }
    }

    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tg_size / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) shared_max[tid] = max(shared_max[tid], shared_max[tid + s2]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    local_sum *= exp(local_max - global_max);
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tg_size / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) shared_sum[tid] += shared_sum[tid + s2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared_sum[0];
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < kv_len; t++) {
            if (causal && t > s) break;
            device const half* k_vec = k_base + t * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++)
                dot += float(q_vec[dd]) * float(k_vec[dd]);
            dot *= scale;
            if (mask_type != 0) {
                uint mask_el = b * mask_batch_stride + h * mask_head_stride + s * kv_len + t;
                if (mask_type == 1) {
                    if (float(mask[mask_el]) == 0.0f) continue;
                } else if (mask_type == 2) {
                    dot += float(mask[mask_el]);
                }
            }
            float w = exp(dot - global_max) * inv_sum;
            acc += w * float(v_base[t * head_dim + d]);
        }
        o_vec[d] = half(acc);
    }
}

// ─── Fused RoPE (Rotary Position Embedding) ─────────────────────────────────
// Half-split layout: first half and second half of each head vector are rotated.
// x[..., :rot_dim/2] and x[..., rot_dim/2:rot_dim] form (cos,sin) pairs.

kernel void fused_rope_f32(
    device const float* x    [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& batch     [[buffer(2)]],
    constant uint& seq_len   [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim  [[buffer(5)]],
    constant uint& rot_dim   [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = batch * seq_len * num_heads * head_dim;
    if (id >= total) return;

    uint d = id % head_dim;
    uint rest = id / head_dim;
    uint s_pos = (rest / num_heads) % seq_len;

    if (d >= rot_dim) {
        dst[id] = x[id]; // passthrough unrotated dims
        return;
    }

    uint half_rot = rot_dim / 2;
    float pos = float(s_pos);

    if (d < half_rot) {
        float freq = 1.0f / pow(10000.0f, float(2 * d) / float(rot_dim));
        float angle = pos * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        // x_rot = x[d] * cos - x[d + half_rot] * sin
        uint partner = id + half_rot;
        dst[id] = x[id] * cos_a - x[partner] * sin_a;
    } else {
        uint d_local = d - half_rot;
        float freq = 1.0f / pow(10000.0f, float(2 * d_local) / float(rot_dim));
        float angle = pos * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        // x_rot = x[d - half_rot] * sin + x[d] * cos
        uint partner = id - half_rot;
        dst[id] = x[partner] * sin_a + x[id] * cos_a;
    }
}

kernel void fused_rope_f16(
    device const half* x    [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant uint& batch     [[buffer(2)]],
    constant uint& seq_len   [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim  [[buffer(5)]],
    constant uint& rot_dim   [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = batch * seq_len * num_heads * head_dim;
    if (id >= total) return;

    uint d = id % head_dim;
    if (d >= rot_dim) { dst[id] = x[id]; return; }

    uint half_rot = rot_dim / 2;
    uint rest = id / head_dim;
    uint s_pos = (rest / num_heads) % seq_len;
    float pos = float(s_pos);

    if (d < half_rot) {
        float freq = 1.0f / pow(10000.0f, float(2 * d) / float(rot_dim));
        float angle = pos * freq;
        uint partner = id + half_rot;
        dst[id] = half(float(x[id]) * cos(angle) - float(x[partner]) * sin(angle));
    } else {
        uint d_local = d - half_rot;
        float freq = 1.0f / pow(10000.0f, float(2 * d_local) / float(rot_dim));
        float angle = pos * freq;
        uint partner = id - half_rot;
        dst[id] = half(float(x[partner]) * sin(angle) + float(x[id]) * cos(angle));
    }
}
