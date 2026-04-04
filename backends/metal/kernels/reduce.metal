#include <metal_stdlib>
using namespace metal;

// ─── Reduction kernels ──────────────────────────────────────────────────────
// Each threadgroup reduces one "row" of inner_size elements to a single scalar.
// Grid: [outer_size] threadgroups, each with up to 1024 threads.
// src: [outer_size, inner_size], dst: [outer_size]

constant uint THREADGROUP_SIZE = 256;

// Shared reduction pattern using threadgroup memory.
#define REDUCE_KERNEL(NAME, T, INIT, COMBINE, FINALIZE)                        \
kernel void NAME(                                                              \
    device const T* src       [[buffer(0)]],                                   \
    device T* dst             [[buffer(1)]],                                   \
    constant uint& outer_size [[buffer(2)]],                                   \
    constant uint& inner_size [[buffer(3)]],                                   \
    uint tg_id   [[threadgroup_position_in_grid]],                             \
    uint tid     [[thread_index_in_threadgroup]],                              \
    uint tg_size [[threads_per_threadgroup]])                                  \
{                                                                              \
    if (tg_id >= outer_size) return;                                           \
    device const T* row = src + tg_id * inner_size;                            \
                                                                               \
    /* Each thread accumulates a strided chunk */                              \
    T acc = INIT;                                                              \
    for (uint i = tid; i < inner_size; i += tg_size) {                         \
        T val = row[i];                                                        \
        acc = COMBINE;                                                         \
    }                                                                          \
                                                                               \
    /* Threadgroup reduction via SIMD shuffle */                               \
    threadgroup T shared[THREADGROUP_SIZE];                                    \
    shared[tid] = acc;                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
                                                                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                               \
        if (tid < s) {                                                         \
            T val = shared[tid + s];                                           \
            acc = COMBINE;                                                     \
            shared[tid] = acc;                                                 \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
                                                                               \
    if (tid == 0) {                                                            \
        dst[tg_id] = FINALIZE;                                                 \
    }                                                                          \
}

// `val` is the element merged into `acc` in both the strided loop and the tree reduction.

// ── Sum ──
REDUCE_KERNEL(reduce_sum_f32, float,
    0.0f,
    acc + val,
    acc)

REDUCE_KERNEL(reduce_sum_f16, half,
    half(0.0),
    acc + val,
    acc)

// ── Product ──
REDUCE_KERNEL(reduce_product_f32, float,
    1.0f,
    acc * val,
    acc)

REDUCE_KERNEL(reduce_product_f16, half,
    half(1.0),
    acc * val,
    acc)

REDUCE_KERNEL(reduce_sum_i32, int32_t,
    int32_t(0),
    acc + val,
    acc)

REDUCE_KERNEL(reduce_product_i32, int32_t,
    int32_t(1),
    acc * val,
    acc)

REDUCE_KERNEL(reduce_max_i32, int32_t,
    INT_MIN,
    max(acc, val),
    acc)

REDUCE_KERNEL(reduce_min_i32, int32_t,
    INT_MAX,
    min(acc, val),
    acc)

REDUCE_KERNEL(reduce_sum_u32, uint32_t,
    uint32_t(0),
    acc + val,
    acc)

REDUCE_KERNEL(reduce_product_u32, uint32_t,
    uint32_t(1),
    acc * val,
    acc)

REDUCE_KERNEL(reduce_max_u32, uint32_t,
    uint32_t(0),
    max(acc, val),
    acc)

REDUCE_KERNEL(reduce_min_u32, uint32_t,
    0xFFFFFFFFu,
    min(acc, val),
    acc)

// ── Max ──
REDUCE_KERNEL(reduce_max_f32, float,
    -HUGE_VALF,
    max(acc, val),
    acc)

REDUCE_KERNEL(reduce_max_f16, half,
    half(-HUGE_VALF),
    max(acc, val),
    acc)

// ── Min ──
REDUCE_KERNEL(reduce_min_f32, float,
    HUGE_VALF,
    min(acc, val),
    acc)

REDUCE_KERNEL(reduce_min_f16, half,
    half(HUGE_VALF),
    min(acc, val),
    acc)

// ── Bitwise reductions on uint32 ──
kernel void reduce_bitwise_and_u32(
    device const uint32_t* src [[buffer(0)]],
    device uint32_t* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uint32_t* row = src + tg_id * inner_size;
    uint32_t acc = 0xFFFFFFFF;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc &= row[i];
    }
    threadgroup uint32_t shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) { acc &= shared[tid + s]; shared[tid] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}

kernel void reduce_bitwise_or_u32(
    device const uint32_t* src [[buffer(0)]],
    device uint32_t* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uint32_t* row = src + tg_id * inner_size;
    uint32_t acc = 0;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc |= row[i];
    }
    threadgroup uint32_t shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) { acc |= shared[tid + s]; shared[tid] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}

kernel void reduce_bitwise_xor_u32(
    device const uint32_t* src [[buffer(0)]],
    device uint32_t* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uint32_t* row = src + tg_id * inner_size;
    uint32_t acc = 0;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc ^= row[i];
    }
    threadgroup uint32_t shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) { acc ^= shared[tid + s]; shared[tid] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}

// Bool / uchar (0–1) logical reductions (empty dtype suffix in metal.m).
kernel void reduce_logical_and(
    device const uchar* src [[buffer(0)]],
    device uchar* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uchar* row = src + tg_id * inner_size;
    uchar acc = 1;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc = acc & (row[i] != 0 ? (uchar)1 : (uchar)0);
    }
    threadgroup uchar shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acc = shared[tid] & shared[tid + s];
            shared[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}

kernel void reduce_logical_or(
    device const uchar* src [[buffer(0)]],
    device uchar* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uchar* row = src + tg_id * inner_size;
    uchar acc = 0;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc = acc | (row[i] != 0 ? (uchar)1 : (uchar)0);
    }
    threadgroup uchar shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acc = shared[tid] | shared[tid + s];
            shared[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}

kernel void reduce_logical_xor(
    device const uchar* src [[buffer(0)]],
    device uchar* dst       [[buffer(1)]],
    constant uint& outer_size  [[buffer(2)]],
    constant uint& inner_size  [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_id >= outer_size) return;
    device const uchar* row = src + tg_id * inner_size;
    uchar acc = 0;
    for (uint i = tid; i < inner_size; i += tg_size) {
        acc = acc ^ (row[i] != 0 ? (uchar)1 : (uchar)0);
    }
    threadgroup uchar shared[THREADGROUP_SIZE];
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acc = shared[tid] ^ shared[tid + s];
            shared[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dst[tg_id] = acc;
}
