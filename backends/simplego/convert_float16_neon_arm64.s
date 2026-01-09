//go:build !noasm && arm64

// NEON-accelerated Float16↔Float32 conversion for ARM64
// Uses FCVTL/FCVTL2 and FCVTN/FCVTN2 instructions (base ARM64 SIMD, no FP16 extension required)
//
// FCVTL: Floating-point Convert to higher precision Long (lower)
//   Converts 4 half-precision (16-bit) values to 4 single-precision (32-bit) values
//   Encoding: 0x0e217800 + Rn<<5 + Rd (4H→4S)
//   For 8H input: FCVTL processes lower 4, FCVTL2 processes upper 4
//
// FCVTN: Floating-point Convert to lower precision Narrow (lower)
//   Converts 4 single-precision (32-bit) values to 4 half-precision (16-bit) values
//   Encoding: 0x0e216800 + Rn<<5 + Rd (4S→4H)
//
// FCVTL2: Same as FCVTL but processes upper 4 elements
//   Encoding: 0x4e217800 + Rn<<5 + Rd
//
// FCVTN2: Same as FCVTN but writes to upper 4 elements
//   Encoding: 0x4e216800 + Rn<<5 + Rd

#include "textflag.h"

// func convertFP16ToFP32_neon_asm(input, output unsafe.Pointer, n int64)
// Converts n Float16 values to Float32 values using NEON
TEXT ·convertFP16ToFP32_neon_asm(SB), NOSPLIT, $0-24
	MOVD input+0(FP), R0    // R0 = input pointer (FP16 array)
	MOVD output+8(FP), R1   // R1 = output pointer (FP32 array)
	MOVD n+16(FP), R2       // R2 = n (count of elements)

	// Process 8 FP16 elements at a time (→ 8 FP32 elements)
	LSR $3, R2, R3          // R3 = n / 8 (vector iterations)
	AND $7, R2, R4          // R4 = n % 8 (scalar remainder)

	CBZ R3, fp16_to_fp32_scalar

fp16_to_fp32_vector:
	// Load 8 FP16 values (128 bits) - ld1 loads raw bytes, FCVTL interprets as FP16
	WORD $0x4cdf7400       // ld1 {v0.8h}, [x0], #16 (size=01 for .8h)

	// FCVTL: Convert lower 4 FP16 → 4 FP32 into v1.4s
	// Encoding: 0x0e217801 (Rn=0, Rd=1)
	WORD $0x0e217801

	// FCVTL2: Convert upper 4 FP16 → 4 FP32 into v2.4s
	// Encoding: 0x4e217802 (Rn=0, Rd=2)
	WORD $0x4e217802

	// Store 8 FP32 values (2x 128 bits = 256 bits total)
	// ST1 uses L=0 (bit 22): 0x4c9f prefix, not 0x4cdf (which is LD1)
	WORD $0x4c9f7821       // st1 {v1.4s}, [x1], #16
	WORD $0x4c9f7822       // st1 {v2.4s}, [x1], #16

	SUBS $1, R3, R3
	BNE fp16_to_fp32_vector

	CBZ R4, fp16_to_fp32_done

fp16_to_fp32_scalar:
	// Load single FP16
	WORD $0x7c402400       // ldr h0, [x0], #2

	// FCVT s1, h0 (convert FP16 to FP32)
	// Encoding: 0x1ee24001
	WORD $0x1ee24001

	// Store single FP32
	WORD $0xbc004421       // str s1, [x1], #4

	SUBS $1, R4, R4
	BNE fp16_to_fp32_scalar

fp16_to_fp32_done:
	RET


// func convertFP32ToFP16_neon_asm(input, output unsafe.Pointer, n int64)
// Converts n Float32 values to Float16 values using NEON
TEXT ·convertFP32ToFP16_neon_asm(SB), NOSPLIT, $0-24
	MOVD input+0(FP), R0    // R0 = input pointer (FP32 array)
	MOVD output+8(FP), R1   // R1 = output pointer (FP16 array)
	MOVD n+16(FP), R2       // R2 = n (count of elements)

	// Process 8 FP32 elements at a time (→ 8 FP16 elements)
	LSR $3, R2, R3          // R3 = n / 8 (vector iterations)
	AND $7, R2, R4          // R4 = n % 8 (scalar remainder)

	CBZ R3, fp32_to_fp16_scalar

fp32_to_fp16_vector:
	// Load 8 FP32 values (2x 128 bits)
	WORD $0x4cdf7800       // ld1 {v0.4s}, [x0], #16 (size=10 for .4s)
	WORD $0x4cdf7801       // ld1 {v1.4s}, [x0], #16

	// FCVTN: Convert first 4 FP32 → lower 4 FP16 into v2.4h
	// Encoding: 0x0e216802 (Rn=0, Rd=2)
	WORD $0x0e216802

	// FCVTN2: Convert next 4 FP32 → upper 4 FP16 into v2.8h
	// Encoding: 0x4e216822 (Rn=1, Rd=2)
	WORD $0x4e216822

	// Store 8 FP16 values (128 bits)
	// ST1 uses L=0: 0x4c9f prefix, size=01 for .8h
	WORD $0x4c9f7422       // st1 {v2.8h}, [x1], #16

	SUBS $1, R3, R3
	BNE fp32_to_fp16_vector

	CBZ R4, fp32_to_fp16_done

fp32_to_fp16_scalar:
	// Load single FP32
	WORD $0xbc404400       // ldr s0, [x0], #4

	// FCVT h1, s0 (convert FP32 to FP16)
	// Encoding: 0x1e63c001
	WORD $0x1e63c001

	// Store single FP16
	WORD $0x7c002421       // str h1, [x1], #2

	SUBS $1, R4, R4
	BNE fp32_to_fp16_scalar

fp32_to_fp16_done:
	RET
