#include <metal_stdlib>
using namespace metal;

struct BnGeom {
    uint channels;
    uint inner;
    uint outer;
    uint numel;
};

inline uint channel_from_gid(uint gid, uint C, uint inner) {
    uint t = gid / inner;
    return t % C;
}

// ─── Training: mean ─────────────────────────────────────────────────────────
[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_train_reduce_mean(
    device const float* x         [[buffer(0)]],
    device float* ms_mean         [[buffer(1)]],
    constant BnGeom& geom         [[buffer(2)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    uint T = geom.outer * geom.inner;
    float acc = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        acc += x[idx];
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float invn = 1.0f / float(max(T, 1u));
        ms_mean[c] = shared[0] * invn;
    }
}

// ─── Training: variance ─────────────────────────────────────────────────────
[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_train_reduce_var(
    device const float* x         [[buffer(0)]],
    device const float* mean_in   [[buffer(1)]],
    device float* ms_var          [[buffer(2)]],
    constant BnGeom& geom         [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    uint T = geom.outer * geom.inner;
    float mu = mean_in[c];
    float acc = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        float d = x[idx] - mu;
        acc += d * d;
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float invn = 1.0f / float(max(T, 1u));
        ms_var[c] = shared[0] * invn;
    }
}

// ─── Training: normalized output ────────────────────────────────────────────
kernel void bn_train_normalize(
    device const float* x         [[buffer(0)]],
    device const float* scale     [[buffer(1)]],
    device const float* offset    [[buffer(2)]],
    device const float* mean_in   [[buffer(3)]],
    device const float* var_in    [[buffer(4)]],
    device float* out             [[buffer(5)]],
    constant BnGeom& geom         [[buffer(6)]],
    constant float& epsilon     [[buffer(7)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (gid >= geom.numel) return;
    uint c = channel_from_gid(gid, geom.channels, geom.inner);
    float mu = mean_in[c];
    float v = var_in[c] + epsilon;
    float invstd = rsqrt(v);
    float g = scale[c];
    float b = offset[c];
    out[gid] = g * (x[gid] - mu) * invstd + b;
}

// ─── Gradient: writes dgamma = sum(dy*xhat), dbeta = sum(dy) ────────────────
[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_grad_reduce(
    device const float* x         [[buffer(0)]],
    device const float* mean_in   [[buffer(1)]],
    device const float* var_in    [[buffer(2)]],
    device const float* dy        [[buffer(3)]],
    device float* dgamma          [[buffer(4)]],
    device float* dbeta           [[buffer(5)]],
    constant BnGeom& geom         [[buffer(6)]],
    constant float& epsilon       [[buffer(7)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float s1shr[256];
    threadgroup float s2shr[256];
    uint T = geom.outer * geom.inner;
    float mu = mean_in[c];
    float inv = rsqrt(var_in[c] + epsilon);
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        float d = dy[idx];
        float xh = (x[idx] - mu) * inv;
        acc1 += d;
        acc2 += d * xh;
    }
    s1shr[tid] = acc1;
    s2shr[tid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) {
            s1shr[tid] += s1shr[tid + s];
            s2shr[tid] += s2shr[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        dbeta[c] = s1shr[0];
        dgamma[c] = s2shr[0];
    }
}

kernel void bn_grad_dx(
    device const float* x         [[buffer(0)]],
    device const float* scale     [[buffer(1)]],
    device const float* mean_in   [[buffer(2)]],
    device const float* var_in    [[buffer(3)]],
    device const float* dy        [[buffer(4)]],
    device const float* dbeta     [[buffer(5)]],
    device const float* dgamma    [[buffer(6)]],
    device float* dx              [[buffer(7)]],
    constant BnGeom& geom         [[buffer(8)]],
    constant float& epsilon       [[buffer(9)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (gid >= geom.numel) return;
    uint c = channel_from_gid(gid, geom.channels, geom.inner);
    uint T = geom.outer * geom.inner;
    float N = float(T);
    float inv = rsqrt(var_in[c] + epsilon);
    float mu = mean_in[c];
    float g = scale[c];
    float xh = (x[gid] - mu) * inv;
    float d = dy[gid];
    float sd = dbeta[c];
    float sdx = dgamma[c];
    dx[gid] = inv * g / N * (N * d - sd - xh * sdx);
}

// ─── float16 (accumulate in float; params and I/O are half where f32 used float) ─

[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_train_reduce_mean_f16(
    device const half* x         [[buffer(0)]],
    device half* ms_mean         [[buffer(1)]],
    constant BnGeom& geom         [[buffer(2)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    uint T = geom.outer * geom.inner;
    float acc = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        acc += float(x[idx]);
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float invn = 1.0f / float(max(T, 1u));
        ms_mean[c] = half(shared[0] * invn);
    }
}

[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_train_reduce_var_f16(
    device const half* x         [[buffer(0)]],
    device const half* mean_in   [[buffer(1)]],
    device half* ms_var          [[buffer(2)]],
    constant BnGeom& geom         [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    uint T = geom.outer * geom.inner;
    float mu = float(mean_in[c]);
    float acc = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        float d = float(x[idx]) - mu;
        acc += d * d;
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float invn = 1.0f / float(max(T, 1u));
        ms_var[c] = half(shared[0] * invn);
    }
}

kernel void bn_train_normalize_f16(
    device const half* x         [[buffer(0)]],
    device const half* scale     [[buffer(1)]],
    device const half* offset    [[buffer(2)]],
    device const half* mean_in   [[buffer(3)]],
    device const half* var_in    [[buffer(4)]],
    device half* out             [[buffer(5)]],
    constant BnGeom& geom         [[buffer(6)]],
    constant float& epsilon     [[buffer(7)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (gid >= geom.numel) return;
    uint c = channel_from_gid(gid, geom.channels, geom.inner);
    float mu = float(mean_in[c]);
    float v = float(var_in[c]) + epsilon;
    float invstd = rsqrt(v);
    float g = float(scale[c]);
    float b = float(offset[c]);
    out[gid] = half(g * (float(x[gid]) - mu) * invstd + b);
}

[[kernel]]
[[max_total_threads_per_threadgroup(256)]]
void bn_grad_reduce_f16(
    device const half* x         [[buffer(0)]],
    device const half* mean_in   [[buffer(1)]],
    device const half* var_in    [[buffer(2)]],
    device const half* dy        [[buffer(3)]],
    device half* dgamma          [[buffer(4)]],
    device half* dbeta           [[buffer(5)]],
    constant BnGeom& geom         [[buffer(6)]],
    constant float& epsilon       [[buffer(7)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint c                        [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    threadgroup float s1shr[256];
    threadgroup float s2shr[256];
    uint T = geom.outer * geom.inner;
    float mu = float(mean_in[c]);
    float inv = rsqrt(float(var_in[c]) + epsilon);
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (uint i = tid; i < T; i += tg_size) {
        uint outer_coord = i / geom.inner;
        uint inner_coord = i % geom.inner;
        uint idx = outer_coord * (geom.channels * geom.inner) + c * geom.inner + inner_coord;
        float d = float(dy[idx]);
        float xh = (float(x[idx]) - mu) * inv;
        acc1 += d;
        acc2 += d * xh;
    }
    s1shr[tid] = acc1;
    s2shr[tid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) {
            s1shr[tid] += s1shr[tid + s];
            s2shr[tid] += s2shr[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        dbeta[c] = half(s1shr[0]);
        dgamma[c] = half(s2shr[0]);
    }
}

kernel void bn_grad_dx_f16(
    device const half* x         [[buffer(0)]],
    device const half* scale     [[buffer(1)]],
    device const half* mean_in   [[buffer(2)]],
    device const half* var_in    [[buffer(3)]],
    device const half* dy        [[buffer(4)]],
    device const half* dbeta     [[buffer(5)]],
    device const half* dgamma    [[buffer(6)]],
    device half* dx              [[buffer(7)]],
    constant BnGeom& geom         [[buffer(8)]],
    constant float& epsilon       [[buffer(9)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (gid >= geom.numel) return;
    uint c = channel_from_gid(gid, geom.channels, geom.inner);
    uint T = geom.outer * geom.inner;
    float N = float(T);
    float inv = rsqrt(float(var_in[c]) + epsilon);
    float mu = float(mean_in[c]);
    float g = float(scale[c]);
    float xh = (float(x[gid]) - mu) * inv;
    float d = float(dy[gid]);
    float sd = float(dbeta[c]);
    float sdx = float(dgamma[c]);
    dx[gid] = half(inv * g / N * (N * d - sd - xh * sdx));
}
