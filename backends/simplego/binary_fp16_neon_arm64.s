//go:build !noasm && arm64

// NEON-accelerated FP16 binary operations for ARM64
// Uses FCVTL/FCVTL2 to convert FP16→FP32, performs operation in FP32,
// then uses FCVTN/FCVTN2 to convert back to FP16.
// Processes 8 FP16 elements per iteration.

#include "textflag.h"

// func binaryAddFP16_neon_asm(a, b, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] + b[i] for n FP16 elements
TEXT ·binaryAddFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD out+16(FP), R2    // R2 = out pointer
	MOVD n+24(FP), R3      // R3 = n (count)

	// Process 8 FP16 elements at a time
	LSR $3, R3, R4         // R4 = n / 8 (vector iterations)
	AND $7, R3, R5         // R5 = n % 8 (remainder)

	CBZ R4, add_scalar_loop

add_vector_loop:
	// Load 8 FP16 from a and b
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16
	WORD $0x4c407825       // ld1 {v5.8h}, [x1]
	WORD $0x91004021       // add x1, x1, #16

	// Convert lower 4 FP16 to FP32
	WORD $0x0e217880       // fcvtl v0.4s, v4.4h (lower)
	WORD $0x0e2178a1       // fcvtl v1.4s, v5.4h (lower)

	// Convert upper 4 FP16 to FP32
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h (upper)
	WORD $0x4e2178a3       // fcvtl2 v3.4s, v5.8h (upper)

	// Add in FP32
	WORD $0x4e21d400       // fadd v0.4s, v0.4s, v1.4s
	WORD $0x4e23d442       // fadd v2.4s, v2.4s, v3.4s

	// Convert back to FP16
	WORD $0x0e216804       // fcvtn v4.4h, v0.4s (lower)
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s (upper)

	// Store 8 FP16
	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE add_vector_loop

	CBZ R5, add_done

add_scalar_loop:
	// Load single FP16
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x7c402025       // ldr h5, [x1], #2

	// Convert to FP32
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1ee240a1       // fcvt s1, h5

	// Add
	WORD $0x1e212800       // fadd s0, s0, s1

	// Convert back to FP16
	WORD $0x1e624000       // fcvt h0, s0

	// Store
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE add_scalar_loop

add_done:
	RET


// func binaryMulFP16_neon_asm(a, b, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] * b[i] for n FP16 elements
TEXT ·binaryMulFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3

	LSR $3, R3, R4         // R4 = n / 8
	AND $7, R3, R5         // R5 = n % 8

	CBZ R4, mul_scalar_loop

mul_vector_loop:
	// Load 8 FP16 from a and b
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16
	WORD $0x4c407825       // ld1 {v5.8h}, [x1]
	WORD $0x91004021       // add x1, x1, #16

	// Convert lower 4 FP16 to FP32
	WORD $0x0e217880       // fcvtl v0.4s, v4.4h
	WORD $0x0e2178a1       // fcvtl v1.4s, v5.4h

	// Convert upper 4 FP16 to FP32
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h
	WORD $0x4e2178a3       // fcvtl2 v3.4s, v5.8h

	// Multiply in FP32
	WORD $0x6e21dc00       // fmul v0.4s, v0.4s, v1.4s
	WORD $0x6e23dc42       // fmul v2.4s, v2.4s, v3.4s

	// Convert back to FP16
	WORD $0x0e216804       // fcvtn v4.4h, v0.4s
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s

	// Store 8 FP16
	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE mul_vector_loop

	CBZ R5, mul_done

mul_scalar_loop:
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x7c402025       // ldr h5, [x1], #2
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1ee240a1       // fcvt s1, h5
	WORD $0x1e210800       // fmul s0, s0, s1
	WORD $0x1e624000       // fcvt h0, s0
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE mul_scalar_loop

mul_done:
	RET


// func binarySubFP16_neon_asm(a, b, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] - b[i] for n FP16 elements
TEXT ·binarySubFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3

	LSR $3, R3, R4
	AND $7, R3, R5

	CBZ R4, sub_scalar_loop

sub_vector_loop:
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16
	WORD $0x4c407825       // ld1 {v5.8h}, [x1]
	WORD $0x91004021       // add x1, x1, #16

	WORD $0x0e217880       // fcvtl v0.4s, v4.4h
	WORD $0x0e2178a1       // fcvtl v1.4s, v5.4h
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h
	WORD $0x4e2178a3       // fcvtl2 v3.4s, v5.8h

	// Subtract in FP32
	WORD $0x4ea1d400       // fsub v0.4s, v0.4s, v1.4s
	WORD $0x4ea3d442       // fsub v2.4s, v2.4s, v3.4s

	WORD $0x0e216804       // fcvtn v4.4h, v0.4s
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s

	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE sub_vector_loop

	CBZ R5, sub_done

sub_scalar_loop:
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x7c402025       // ldr h5, [x1], #2
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1ee240a1       // fcvt s1, h5
	WORD $0x1e213800       // fsub s0, s0, s1
	WORD $0x1e624000       // fcvt h0, s0
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE sub_scalar_loop

sub_done:
	RET


// func binaryDivFP16_neon_asm(a, b, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] / b[i] for n FP16 elements
TEXT ·binaryDivFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3

	LSR $3, R3, R4
	AND $7, R3, R5

	CBZ R4, div_scalar_loop

div_vector_loop:
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16
	WORD $0x4c407825       // ld1 {v5.8h}, [x1]
	WORD $0x91004021       // add x1, x1, #16

	WORD $0x0e217880       // fcvtl v0.4s, v4.4h
	WORD $0x0e2178a1       // fcvtl v1.4s, v5.4h
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h
	WORD $0x4e2178a3       // fcvtl2 v3.4s, v5.8h

	// Divide in FP32
	WORD $0x6e21fc00       // fdiv v0.4s, v0.4s, v1.4s
	WORD $0x6e23fc42       // fdiv v2.4s, v2.4s, v3.4s

	WORD $0x0e216804       // fcvtn v4.4h, v0.4s
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s

	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE div_vector_loop

	CBZ R5, div_done

div_scalar_loop:
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x7c402025       // ldr h5, [x1], #2
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1ee240a1       // fcvt s1, h5
	WORD $0x1e211800       // fdiv s0, s0, s1
	WORD $0x1e624000       // fcvt h0, s0
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE div_scalar_loop

div_done:
	RET


// func binaryAddScalarFP16_neon_asm(a unsafe.Pointer, scalar uint16, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] + scalar for n FP16 elements
TEXT ·binaryAddScalarFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVW scalar+8(FP), R1  // scalar as uint16
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3

	// Load scalar into FP register and convert to FP32
	FMOVS R1, F5           // Move scalar bits to h5
	WORD $0x1ee240a1       // fcvt s1, h5 - convert scalar to FP32
	WORD $0x4f019421       // dup v1.4s, v1.s[0] - broadcast to all 4 lanes

	LSR $3, R3, R4
	AND $7, R3, R5

	CBZ R4, adds_scalar_loop

adds_vector_loop:
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16

	WORD $0x0e217880       // fcvtl v0.4s, v4.4h
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h

	WORD $0x4e21d400       // fadd v0.4s, v0.4s, v1.4s
	WORD $0x4e21d442       // fadd v2.4s, v2.4s, v1.4s

	WORD $0x0e216804       // fcvtn v4.4h, v0.4s
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s

	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE adds_vector_loop

	CBZ R5, adds_done

adds_scalar_loop:
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1e212800       // fadd s0, s0, s1
	WORD $0x1e624000       // fcvt h0, s0
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE adds_scalar_loop

adds_done:
	RET


// func binaryMulScalarFP16_neon_asm(a unsafe.Pointer, scalar uint16, out unsafe.Pointer, n int64)
// Performs out[i] = a[i] * scalar for n FP16 elements
TEXT ·binaryMulScalarFP16_neon_asm(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVW scalar+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3

	FMOVS R1, F5
	WORD $0x1ee240a1       // fcvt s1, h5
	WORD $0x4f019421       // dup v1.4s, v1.s[0]

	LSR $3, R3, R4
	AND $7, R3, R5

	CBZ R4, muls_scalar_loop

muls_vector_loop:
	WORD $0x4c407804       // ld1 {v4.8h}, [x0]
	WORD $0x91004000       // add x0, x0, #16

	WORD $0x0e217880       // fcvtl v0.4s, v4.4h
	WORD $0x4e217882       // fcvtl2 v2.4s, v4.8h

	WORD $0x6e21dc00       // fmul v0.4s, v0.4s, v1.4s
	WORD $0x6e21dc42       // fmul v2.4s, v2.4s, v1.4s

	WORD $0x0e216804       // fcvtn v4.4h, v0.4s
	WORD $0x4e216844       // fcvtn2 v4.8h, v2.4s

	WORD $0x4c007844       // st1 {v4.8h}, [x2]
	WORD $0x91004042       // add x2, x2, #16

	SUBS $1, R4, R4
	BNE muls_vector_loop

	CBZ R5, muls_done

muls_scalar_loop:
	WORD $0x7c402004       // ldr h4, [x0], #2
	WORD $0x1ee24084       // fcvt s0, h4
	WORD $0x1e210800       // fmul s0, s0, s1
	WORD $0x1e624000       // fcvt h0, s0
	WORD $0x7c002040       // str h0, [x2], #2

	SUBS $1, R5, R5
	BNE muls_scalar_loop

muls_done:
	RET
