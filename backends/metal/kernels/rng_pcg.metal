#include <metal_stdlib>
using namespace metal;

// PCG128-DXSM matching Go 1.25+ math/rand/v2/pcg.go (NewPCG seed1,seed2 -> hi,lo state).

void pcg_next(thread ulong& hi, thread ulong& lo) {
    const ulong mulHi = 2549297995355413924UL;
    const ulong mulLo = 4865540595714422341UL;
    const ulong incHi = 6364136223846793005UL;
    const ulong incLo = 1442695040888963407UL;
    ulong p_hi = hi;
    ulong p_lo = lo;
    ulong lo64 = p_lo * mulLo;
    ulong hi64 = mulhi(p_lo, mulLo);
    hi64 += p_hi * mulLo + p_lo * mulHi;
    ulong new_lo = lo64 + incLo;
    ulong carry = (new_lo < lo64) ? 1UL : 0UL;
    ulong new_hi = hi64 + incHi + carry;
    lo = new_lo;
    hi = new_hi;
}

ulong pcg_uint64(thread ulong& hi, thread ulong& lo) {
    pcg_next(hi, lo);
    ulong out_hi = hi;
    ulong out_lo = lo;
    const ulong cheapMul = 0xda942042e4dd58b5UL;
    out_hi ^= out_hi >> 32;
    out_hi *= cheapMul;
    out_hi ^= out_hi >> 48;
    out_hi *= (out_lo | 1UL);
    return out_hi;
}

// Thread 0 consumes the PCG stream identically to simplego/RNGBitGenerator (fill bytes from uint64s, LE).
kernel void rng_pcg_fill_bytes(
    device const ulong* state_in  [[buffer(0)]],
    device ulong* state_out        [[buffer(1)]],
    device uchar* dst              [[buffer(2)]],
    constant uint& num_bytes       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    ulong hi = state_in[0];
    ulong lo = state_in[1];
    ulong word = 0;
    int have = 0;
    for (uint i = 0; i < num_bytes; i++) {
        if (have == 0) {
            word = pcg_uint64(hi, lo);
            have = 8;
        }
        dst[i] = uchar(word & 0xFFUL);
        word >>= 8;
        have--;
    }
    state_out[0] = hi;
    state_out[1] = lo;
    state_out[2] = state_in[2];
}
