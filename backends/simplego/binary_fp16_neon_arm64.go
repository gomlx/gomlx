//go:build !noasm && arm64

package simplego

import (
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/x448/float16"
)

// NEON-accelerated FP16 binary operations.
// These use FCVTL/FCVTL2 to convert FP16→FP32, perform the operation in FP32,
// then use FCVTN/FCVTN2 to convert back to FP16.
// This is 8-16x faster than scalar FP16 operations.

//go:noescape
func binaryAddFP16_neon_asm(a, b, out unsafe.Pointer, n int64)

//go:noescape
func binaryMulFP16_neon_asm(a, b, out unsafe.Pointer, n int64)

//go:noescape
func binarySubFP16_neon_asm(a, b, out unsafe.Pointer, n int64)

//go:noescape
func binaryDivFP16_neon_asm(a, b, out unsafe.Pointer, n int64)

//go:noescape
func binaryAddScalarFP16_neon_asm(a unsafe.Pointer, scalar uint16, out unsafe.Pointer, n int64)

//go:noescape
func binaryMulScalarFP16_neon_asm(a unsafe.Pointer, scalar uint16, out unsafe.Pointer, n int64)

// Minimum size for NEON operations (8 FP16 elements = 16 bytes)
const minBinaryNEONSize = 8

// execBinaryFloat16NEON performs a binary operation on two FP16 arrays using NEON.
// opType: 0=add, 1=mul, 2=sub, 3=div
func execBinaryFloat16NEON(opType int, lhs, rhs []float16.Float16, output []float16.Float16,
	lhsShape, rhsShape, outputShape shapes.Shape) {

	n := len(output)

	// Handle scalar on rhs side (most common case for bias addition)
	if len(rhs) == 1 {
		scalar := uint16(rhs[0])
		if n >= minBinaryNEONSize {
			switch opType {
			case 0: // Add
				binaryAddScalarFP16_neon_asm(
					unsafe.Pointer(&lhs[0]),
					scalar,
					unsafe.Pointer(&output[0]),
					int64(n),
				)
				return
			case 1: // Mul
				binaryMulScalarFP16_neon_asm(
					unsafe.Pointer(&lhs[0]),
					scalar,
					unsafe.Pointer(&output[0]),
					int64(n),
				)
				return
			}
		}
		// Fall back to scalar for small arrays or unsupported ops
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			switch opType {
			case 0:
				output[ii] = float16.Fromfloat32(a + c)
			case 1:
				output[ii] = float16.Fromfloat32(a * c)
			case 2:
				output[ii] = float16.Fromfloat32(a - c)
			case 3:
				output[ii] = float16.Fromfloat32(a / c)
			}
		}
		return
	}

	// Handle same-shape case (element-wise)
	if lhsShape.Equal(rhsShape) && n >= minBinaryNEONSize {
		switch opType {
		case 0: // Add
			binaryAddFP16_neon_asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&output[0]),
				int64(n),
			)
			return
		case 1: // Mul
			binaryMulFP16_neon_asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&output[0]),
				int64(n),
			)
			return
		case 2: // Sub
			binarySubFP16_neon_asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&output[0]),
				int64(n),
			)
			return
		case 3: // Div
			binaryDivFP16_neon_asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&output[0]),
				int64(n),
			)
			return
		}
	}

	// Fall back to scalar for broadcasting cases or unsupported ops
	if lhsShape.Equal(rhsShape) {
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			switch opType {
			case 0:
				output[outputIdx] = float16.Fromfloat32(a + b)
			case 1:
				output[outputIdx] = float16.Fromfloat32(a * b)
			case 2:
				output[outputIdx] = float16.Fromfloat32(a - b)
			case 3:
				output[outputIdx] = float16.Fromfloat32(a / b)
			}
		}
	} else {
		// Broadcasting case
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
			switch opType {
			case 0:
				output[outputIdx] = float16.Fromfloat32(a + b)
			case 1:
				output[outputIdx] = float16.Fromfloat32(a * b)
			case 2:
				output[outputIdx] = float16.Fromfloat32(a - b)
			case 3:
				output[outputIdx] = float16.Fromfloat32(a / b)
			}
		}
	}
}
