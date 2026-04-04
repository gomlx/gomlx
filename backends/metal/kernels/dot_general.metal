#include <metal_stdlib>
using namespace metal;

// ─── Batched MatMul: C[b,m,n] = A[b,m,k] @ B[b,k,n] ───────────────────────
// Uses tiled shared-memory approach for cache efficiency.

constant uint TILE = 16;

kernel void dot_general_f32(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float* C           [[buffer(2)]],
    constant uint& batch      [[buffer(3)]],
    constant uint& M          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& N          [[buffer(6)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    // gid.z = batch index, gid.y = row in M, gid.x = col in N
    uint b = gid.z;
    if (b >= batch) return;

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    device const float* Ab = A + b * M * K;
    device const float* Bb = B + b * K * N;
    device float* Cb = C + b * M * N;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;

        tileA[tid.y][tid.x] = (row < M && ak < K) ? Ab[row * K + ak] : 0.0f;
        tileB[tid.y][tid.x] = (bk < K && col < N) ? Bb[bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += tileA[tid.y][i] * tileB[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

kernel void dot_general_f16(
    device const half* A     [[buffer(0)]],
    device const half* B     [[buffer(1)]],
    device half* C           [[buffer(2)]],
    constant uint& batch     [[buffer(3)]],
    constant uint& M         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& N         [[buffer(6)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint b = gid.z;
    if (b >= batch) return;

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    device const half* Ab = A + b * M * K;
    device const half* Bb = B + b * K * N;
    device half* Cb = C + b * M * N;

    threadgroup half tileA[TILE][TILE];
    threadgroup half tileB[TILE][TILE];

    float acc = 0.0f; // accumulate in fp32 for precision

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;

        tileA[tid.y][tid.x] = (row < M && ak < K) ? Ab[row * K + ak] : half(0.0);
        tileB[tid.y][tid.x] = (bk < K && col < N) ? Bb[bk * N + col] : half(0.0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += float(tileA[tid.y][i]) * float(tileB[i][tid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        Cb[row * N + col] = half(acc);
    }
}
