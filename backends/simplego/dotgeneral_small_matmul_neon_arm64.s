//go:build !noasm && arm64

// NEON-accelerated SmallMatMul for ARM64
// Computes 4 output columns at once using register blocking.
//
// Key insight: By processing 4 columns together, RHS access becomes contiguous:
//   - lhs[m,k] is scalar (broadcast)
//   - rhs[k, n:n+4] is 4 contiguous floats (row access)
//   - output[m, n:n+4] is 4 contiguous floats

#include "textflag.h"

// func smallMatMulRow4_neon_asm(lhs, rhs, output unsafe.Pointer, K, N int64)
//
// Computes 4 output values: output[0:4] = sum over k of lhs[k] * rhs[k*N : k*N+4]
//
// Parameters:
//   lhs (R0): pointer to lhs[m, 0] - start of LHS row (K contiguous floats)
//   rhs (R1): pointer to rhs[0, n] - start of column group
//   output (R2): pointer to output[m, n] - where to store 4 results
//   K (R3): contracting dimension (number of iterations)
//   N (R4): RHS row stride in elements
//
// Register allocation:
//   R0: lhs pointer (advances by 4 bytes per iteration)
//   R1: rhs pointer (advances by N*4 bytes per iteration)
//   R2: output pointer
//   R3: loop counter (K)
//   R5: N*4 (byte stride for RHS)
//   v0: accumulator for 4 output values
//   v4: broadcast LHS value (only s4/lane 0 used)
//   v5: 4 contiguous RHS values
//
TEXT ·smallMatMulRow4_neon_asm(SB), NOSPLIT, $0-40
	MOVD lhs+0(FP), R0      // R0 = lhs pointer
	MOVD rhs+8(FP), R1      // R1 = rhs pointer
	MOVD output+16(FP), R2  // R2 = output pointer
	MOVD K+24(FP), R3       // R3 = K (contracting dimension)
	MOVD N+32(FP), R4       // R4 = N (RHS stride in elements)

	// Calculate byte stride: N * 4 bytes per float32
	LSL $2, R4, R5          // R5 = N * 4 (byte stride for RHS rows)

	// Initialize accumulator to zero
	WORD $0x4f000400        // movi v0.4s, #0

	// Main loop: process one k iteration at a time
	CBZ R3, store_result

loop:
	// Load lhs[k] as scalar into s4
	WORD $0xbd400004        // ldr s4, [x0]
	WORD $0x91001000        // add x0, x0, #4

	// Load rhs[k, n:n+4] - 4 contiguous floats into v5.4s
	WORD $0x4c407825        // ld1 {v5.4s}, [x1]
	ADD R5, R1, R1          // advance rhs pointer by N*4 bytes

	// Fused multiply-accumulate: v0.4s += v5.4s * v4.s[0]
	// This multiplies each lane of v5 by the scalar in v4.s[0]
	WORD $0x4f8410a0        // fmla v0.4s, v5.4s, v4.s[0]

	SUBS $1, R3, R3
	BNE loop

store_result:
	// Store 4 results to output[m, n:n+4]
	WORD $0x4c007840        // st1 {v0.4s}, [x2]

	RET
