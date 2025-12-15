//go:build !noasm && arm64

// NEON-accelerated int8 dot product for ARM64
// Uses SDOT (Signed Dot Product) instruction for 4x throughput vs scalar.
// SDOT performs 4 groups of 4x int8 multiplications and accumulates into 4x int32 lanes.
// Each lane accumulates: v0.s[i] += sum(v4.b[4*i:4*i+3] * v8.b[4*i:4*i+3])
//
// Note: SMMLA (Signed Matrix Multiply-Accumulate) is a different instruction
// available on ARMv8.6-A that performs 2x8 × 8x2 matrix multiplies.

#include "textflag.h"

// func dotProductInt8_neon_asm(a, b unsafe.Pointer, n int64) int32
TEXT ·dotProductInt8_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD n+16(FP), R2      // R2 = n (count)

	// Initialize accumulator to zero (4x int32)
	WORD $0x4f000400       // movi v0.4s, #0

	// Process 16 int8s at a time using SDOT
	LSR $4, R2, R3         // R3 = n / 16 (SDOT iterations)
	AND $15, R2, R4        // R4 = n % 16 (scalar remainder)

	CBZ R3, scalarloop_init // Skip SDOT loop if < 16 elements, need to init R3

sdotloop:
	// Prefetch next cache line
	WORD $0xf9800400       // prfm pldl1keep, [x0, #128]
	WORD $0xf9800420       // prfm pldl1keep, [x1, #128]

	// Load 16x int8 from each array (into 128-bit vectors)
	WORD $0x4cdf7004       // ld1 {v4.16b}, [x0], #16
	WORD $0x4cdf7028       // ld1 {v8.16b}, [x1], #16

	// SDOT: Signed 8-bit integer dot product (4 groups of 4 elements → 4 int32 accumulators)
	// Lane 0: v0.s[0] += v4.b[0:3] · v8.b[0:3]
	// Lane 1: v0.s[1] += v4.b[4:7] · v8.b[4:7]
	// Lane 2: v0.s[2] += v4.b[8:11] · v8.b[8:11]
	// Lane 3: v0.s[3] += v4.b[12:15] · v8.b[12:15]
	WORD $0x4e889480       // sdot v0.4s, v4.16b, v8.16b

	SUBS $1, R3, R3
	BNE sdotloop

	// Horizontal reduction: sum all 4 int32 lanes of v0
	WORD $0x4eb1b800       // addv s0, v0.4s

	// Extract the final sum to w3 (FIX: was incorrectly going to w0)
	WORD $0x0e043c03       // umov w3, v0.s[0]

	// Handle remaining 0-15 elements
	CBZ R4, done
	B scalarloop           // Jump over init

scalarloop_init:
	// FIX: Initialize accumulator for scalar-only path
	MOVW $0, R3

scalarloop:
	// Load single int8 from each array (use w6/w7 to avoid conflict with R4 loop counter)
	WORD $0x38401406       // ldrb w6, [x0], #1
	WORD $0x38401427       // ldrb w7, [x1], #1

	// Sign-extend int8 → int32
	WORD $0x13001cc6       // sxtb w6, w6
	WORD $0x13001ce7       // sxtb w7, w7

	// Multiply and accumulate: w3 = w3 + w6 * w7
	WORD $0x1b070cc3       // madd w3, w6, w7, w3

	SUBS $1, R4, R4
	BNE scalarloop

done:
	MOVW R3, ret+24(FP)
	RET

// func dotProductUint8_neon_asm(a, b unsafe.Pointer, n int64) int32
TEXT ·dotProductUint8_neon_asm(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	// Initialize accumulator
	WORD $0x4f000400       // movi v0.4s, #0

	// Process 16 uint8s at a time using UDOT (unsigned version)
	LSR $4, R2, R3
	AND $15, R2, R4

	CBZ R3, uscalarloop_init // Skip UDOT loop if < 16 elements, need to init R3

udotloop:
	// Prefetch next cache line
	WORD $0xf9800400       // prfm pldl1keep, [x0, #128]
	WORD $0xf9800420       // prfm pldl1keep, [x1, #128]

	// Load 16x uint8
	WORD $0x4cdf7004       // ld1 {v4.16b}, [x0], #16
	WORD $0x4cdf7028       // ld1 {v8.16b}, [x1], #16

	// UDOT: Unsigned 8-bit integer dot product (4 groups of 4 elements → 4 int32 accumulators)
	WORD $0x6e889480       // udot v0.4s, v4.16b, v8.16b

	SUBS $1, R3, R3
	BNE udotloop

	// Horizontal reduction
	WORD $0x4eb1b800       // addv s0, v0.4s
	WORD $0x0e043c03       // umov w3, v0.s[0]

	CBZ R4, udone
	B uscalarloop          // Jump over init

uscalarloop_init:
	// FIX: Initialize accumulator for scalar-only path
	MOVW $0, R3

uscalarloop:
	// Load and multiply (zero-extended for uint8, use w6/w7 to avoid conflict with R4 loop counter)
	WORD $0x38401406       // ldrb w6, [x0], #1
	WORD $0x38401427       // ldrb w7, [x1], #1
	WORD $0x1b070cc3       // madd w3, w6, w7, w3  // w3 = w3 + w6 * w7

	SUBS $1, R4, R4
	BNE uscalarloop

udone:
	MOVW R3, ret+24(FP)
	RET
