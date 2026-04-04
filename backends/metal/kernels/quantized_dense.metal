#include "gomlx_erf.h"

// Matches backends.NF4LookupTable (QLoRA NF4).
constant float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f,
};

// activation: backends.ActivationType — None=0, Gelu=1, Relu=2, Silu=3, HardSwish=4, Tanh=5
METAL_FUNC float qd_apply_activation(float x, uint act) {
    switch (act) {
        case 0:
            return x;
        case 1: {
            float t = x * 0.7071067811865476f;  // x / sqrt(2)
            return x * 0.5f * (1.0f + gomlx_erf(t));
        }
        case 2:
            return max(0.0f, x);
        case 3:
            return x / (1.0f + exp(-x));
        case 4: {
            const float scale = 1.0f / 6.0f;
            float sh = min(max(x * scale + 0.5f, 0.0f), 1.0f);
            return x * sh;
        }
        case 5:
            return tanh(x);
        default:
            return x;
    }
}

METAL_FUNC void qd_read_packed_nibble(device const uchar* w, uint lin, thread uint& nib) {
    uint bi = lin >> 1;
    uint lo = lin & 1u;
    nib = lo ? ((w[bi] >> 4) & 0xFu) : (w[bi] & 0xFu);
}

METAL_FUNC int qd_signed_nibble(uint nib) {
    return (nib >= 8u) ? int(nib) - 16 : int(nib);
}

// NF4: one byte per weight (nibble index 0–15), row-major [K, N]
kernel void q_dense_nf4_u8_f32(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    (void)zp;
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        uchar wi = w[k * N + n];
        float wdq = NF4_TABLE[wi & 15u];
        acc += x[m * K + k] * wdq * sc;
    }
    out[gid] = qd_apply_activation(acc, act);
}

// NF4: packed 4-bit weights (2 nibbles per byte), low nibble first within each byte
kernel void q_dense_nf4_pack_f32(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    (void)zp;
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        uint lin = k * N + n;
        uint nib;
        qd_read_packed_nibble(w, lin, nib);
        float wdq = NF4_TABLE[nib & 15u];
        acc += x[m * K + k] * wdq * sc;
    }
    out[gid] = qd_apply_activation(acc, act);
}

// Linear int8 weights, one byte per element
kernel void q_dense_lin_i8_f32(
    device const float* x [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint hasZp = p[6];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        float wv = float(w[k * N + n]);
        float zpv = (hasZp != 0u) ? zp[k * numBlocks + bidx] : 0.0f;
        acc += x[m * K + k] * (wv * sc + zpv);
    }
    out[gid] = qd_apply_activation(acc, act);
}

kernel void q_dense_lin_u8_f32(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint hasZp = p[6];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        float wv = float(w[k * N + n]);
        float zpv = (hasZp != 0u) ? zp[k * numBlocks + bidx] : 0.0f;
        acc += x[m * K + k] * (wv * sc + zpv);
    }
    out[gid] = qd_apply_activation(acc, act);
}

kernel void q_dense_lin_i4p_f32(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint hasZp = p[6];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        uint lin = k * N + n;
        uint nib;
        qd_read_packed_nibble(w, lin, nib);
        float wv = float(qd_signed_nibble(nib));
        float zpv = (hasZp != 0u) ? zp[k * numBlocks + bidx] : 0.0f;
        acc += x[m * K + k] * (wv * sc + zpv);
    }
    out[gid] = qd_apply_activation(acc, act);
}

kernel void q_dense_lin_u4p_f32(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zp [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint M = p[0], K = p[1], N = p[2], blockSize = p[3], numBlocks = p[4];
    uint hasBias = p[5];
    uint hasZp = p[6];
    uint act = p[7];
    if (gid >= M * N) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = (hasBias != 0u) ? bias[n] : 0.0f;
    for (uint k = 0; k < K; k++) {
        uint bidx = n / blockSize;
        float sc = scales[k * numBlocks + bidx];
        uint lin = k * N + n;
        uint nib;
        qd_read_packed_nibble(w, lin, nib);
        float wv = float(nib);
        float zpv = (hasZp != 0u) ? zp[k * numBlocks + bidx] : 0.0f;
        acc += x[m * K + k] * (wv * sc + zpv);
    }
    out[gid] = qd_apply_activation(acc, act);
}
