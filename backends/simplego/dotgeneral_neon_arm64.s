//go:build !noasm && arm64

// NEON-accelerated dot product for ARM64
// Works on all ARM64 processors (Apple M1+, Linux ARM64, etc.)
// Uses 128-bit NEON vectors (4 x float32)

#include "textflag.h"

// func dotProduct_neon_asm(a, b unsafe.Pointer, n int64) float32
TEXT ·dotProduct_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD n+16(FP), R2      // R2 = n (count)

	// Initialize accumulator to zero
	WORD $0x4f000400       // movi v0.4s, #0

	// Process 4 floats at a time using NEON
	LSR $2, R2, R3         // R3 = n / 4 (vector iterations)
	AND $3, R2, R4         // R4 = n % 4 (scalar remainder)

	CBZ R3, scalarloop     // skip vector loop if < 4 elements

vectorloop:
	// Prefetch next cache line (64 bytes ahead = 4 iterations)
	WORD $0xf9800400       // prfm pldl1keep, [x0, #128]
	WORD $0xf9800420       // prfm pldl1keep, [x1, #128]

	// Load 4 floats from each array
	WORD $0x4cdf7804       // ld1 {v4.4s}, [x0], #16
	WORD $0x4cdf7828       // ld1 {v8.4s}, [x1], #16

	// Fused multiply-accumulate: v0 += v4 * v8
	WORD $0x4e28cc80       // fmla v0.4s, v4.4s, v8.4s

	SUBS $1, R3, R3
	BNE vectorloop

	// Horizontal reduction: sum all 4 lanes of v0 to get scalar result
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s

	// Handle remaining 0-3 elements
	CBZ R4, done

scalarloop:
	// Use explicit load and add to avoid opcode ambiguity
	WORD $0xbd400004       // ldr s4, [x0]
	WORD $0x91001000       // add x0, x0, #4
	WORD $0xbd400025       // ldr s5, [x1]
	WORD $0x91001021       // add x1, x1, #4
	
	WORD $0x1f050080       // fmadd s0, s4, s5, s0
	SUBS $1, R4, R4
	BNE scalarloop

done:
	FMOVS F0, ret+24(FP)
	RET

// func dotProductGroup4_neon_asm(a, b unsafe.Pointer, b_stride, n int64) (r0, r1, r2, r3 float32)
// Calculates 4 dot products sharing the same LHS (a).
// b is the pointer to the first RHS vector. b_stride is the stride *in elements* (4 bytes) between RHS vectors.
TEXT ·dotProductGroup4_neon_asm(SB), NOSPLIT, $0-48
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b0 pointer
	MOVD b_stride+16(FP), R2
	MOVD n+24(FP), R3      // R3 = n (count)

	// Calculate b1, b2, b3 pointers
	ADD  R2, R2, R2        // R2 = stride * 2
	ADD  R2, R2, R2        // R2 = stride * 4 (bytes)
	ADD  R2, R1, R4        // R4 = b1
	ADD  R2, R4, R5        // R5 = b2
	ADD  R2, R5, R6        // R6 = b3

	// Initialize 4 accumulators to zero
	WORD $0x4f000400       // movi v0.4s, #0 (acc0)
	WORD $0x4f000401       // movi v1.4s, #0 (acc1)
	WORD $0x4f000402       // movi v2.4s, #0 (acc2)
	WORD $0x4f000403       // movi v3.4s, #0 (acc3)

	LSR $2, R3, R7         // R7 = n / 4 (vector iterations)
	AND $3, R3, R8         // R8 = n % 4 (scalar remainder)

	CBZ R7, g4_scalarloop

g4_vectorloop:
	// Prefetch next cache lines for all 5 arrays (128 bytes ahead)
	WORD $0xf9800400       // prfm pldl1keep, [x0, #128]
	WORD $0xf9800420       // prfm pldl1keep, [x1, #128]
	WORD $0xf9800480       // prfm pldl1keep, [x4, #128]
	WORD $0xf98004a0       // prfm pldl1keep, [x5, #128]
	WORD $0xf98004c0       // prfm pldl1keep, [x6, #128]

	// Load LHS (shared)
	WORD $0x4cdf7804       // ld1 {v4.4s}, [x0], #16

	// Load RHS vectors
	WORD $0x4cdf7825       // ld1 {v5.4s}, [x1], #16 (b0)
	WORD $0x4cdf7886       // ld1 {v6.4s}, [x4], #16 (b1)
	WORD $0x4cdf78a7       // ld1 {v7.4s}, [x5], #16 (b2)
	WORD $0x4cdf78c8       // ld1 {v8.4s}, [x6], #16 (b3)

	// FMLA
	WORD $0x4e25cc80       // fmla v0.4s, v4.4s, v5.4s
	WORD $0x4e26cc81       // fmla v1.4s, v4.4s, v6.4s
	WORD $0x4e27cc82       // fmla v2.4s, v4.4s, v7.4s
	WORD $0x4e28cc83       // fmla v3.4s, v4.4s, v8.4s

	SUBS $1, R7, R7
	BNE g4_vectorloop

	// Reduce vectors
	WORD $0x6e20d400       // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7e30d800       // faddp s0, v0.2s
	WORD $0x6e21d421       // faddp v1.4s, v1.4s, v1.4s
	WORD $0x7e30d821       // faddp s1, v1.2s
	WORD $0x6e22d442       // faddp v2.4s, v2.4s, v2.4s
	WORD $0x7e30d842       // faddp s2, v2.2s
	WORD $0x6e23d463       // faddp v3.4s, v3.4s, v3.4s
	WORD $0x7e30d863       // faddp s3, v3.2s

	CBZ R8, g4_done

g4_scalarloop:
	// Explicit load and add
	WORD $0xbd400004       // ldr s4, [x0]
	WORD $0x91001000       // add x0, x0, #4
	
	WORD $0xbd400025       // ldr s5, [x1]
	WORD $0x91001021       // add x1, x1, #4
	WORD $0xbd400086       // ldr s6, [x4]
	WORD $0x91001084       // add x4, x4, #4
	WORD $0xbd4000a7       // ldr s7, [x5]
	WORD $0x910010a5       // add x5, x5, #4
	WORD $0xbd4000c8       // ldr s8, [x6]
	WORD $0x910010c6       // add x6, x6, #4

	WORD $0x1f050080       // fmadd s0, s4, s5, s0
	WORD $0x1f060081       // fmadd s1, s4, s6, s1
	WORD $0x1f070082       // fmadd s2, s4, s7, s2
	WORD $0x1f080083       // fmadd s3, s4, s8, s3

	SUBS $1, R8, R8
	BNE g4_scalarloop

g4_done:
	FMOVS F0, r0+32(FP)
	FMOVS F1, r1+36(FP)
	FMOVS F2, r2+40(FP)
	FMOVS F3, r3+44(FP)
	RET
