//go:build !noasm && arm64

package simplego

import (
	"runtime"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// Assembly functions for FP16↔FP32 conversion (defined in convert_fp16_neon_arm64.s)
// These use FCVTL/FCVTL2 and FCVTN/FCVTN2 which are base ARM64 SIMD instructions
//
//go:noescape
func convertFP16ToFP32_neon_asm(input, output unsafe.Pointer, n int64)

//go:noescape
func convertFP32ToFP16_neon_asm(input, output unsafe.Pointer, n int64)

// Minimum size for NEON conversion (processing 8 elements at a time)
const minConversionNEONSize = 8

// execConvertDTypeFloat16ToFloat32NEON uses NEON for fast FP16→FP32 conversion
func execConvertDTypeFloat16ToFloat32NEON(operand, output *Buffer) {
	operandFlat := operand.flat.([]float16.Float16)
	outputFlat := output.flat.([]float32)
	n := len(operandFlat)

	if n >= minConversionNEONSize {
		// Process multiples of 8 with NEON (20x faster than scalar)
		neonCount := (n / 8) * 8
		convertFP16ToFP32_neon_asm(
			unsafe.Pointer(&operandFlat[0]),
			unsafe.Pointer(&outputFlat[0]),
			int64(neonCount),
		)
		// Keep slices alive until after assembly completes
		runtime.KeepAlive(operandFlat)
		runtime.KeepAlive(outputFlat)
		// Handle remainder with scalar Go code
		for idx := neonCount; idx < n; idx++ {
			outputFlat[idx] = operandFlat[idx].Float32()
		}
	} else {
		// Scalar fallback for small arrays
		for idx, value := range operandFlat {
			outputFlat[idx] = value.Float32()
		}
	}
}

// execConvertDTypeFloat32ToFloat16NEON uses NEON for fast FP32→FP16 conversion
func execConvertDTypeFloat32ToFloat16NEON(operand, output *Buffer) {
	operandFlat := operand.flat.([]float32)
	outputFlat := output.flat.([]float16.Float16)
	n := len(operandFlat)

	if n >= minConversionNEONSize {
		// Process multiples of 8 with NEON (25x faster than scalar)
		neonCount := (n / 8) * 8
		convertFP32ToFP16_neon_asm(
			unsafe.Pointer(&operandFlat[0]),
			unsafe.Pointer(&outputFlat[0]),
			int64(neonCount),
		)
		// Keep slices alive until after assembly completes
		runtime.KeepAlive(operandFlat)
		runtime.KeepAlive(outputFlat)
		// Handle remainder with scalar Go code
		for idx := neonCount; idx < n; idx++ {
			outputFlat[idx] = float16.Fromfloat32(operandFlat[idx])
		}
	} else {
		// Scalar fallback for small arrays
		for idx, value := range operandFlat {
			outputFlat[idx] = float16.Fromfloat32(value)
		}
	}
}

// convertFloat32SliceToFloat16 converts a Float32 slice to Float16 using NEON when available.
// This is used for bulk conversion in DotGeneral output copy.
func convertFloat32SliceToFloat16(input []float32, output []float16.Float16) {
	n := len(input)
	if n >= minConversionNEONSize {
		// Process multiples of 8 with NEON
		neonCount := (n / 8) * 8
		convertFP32ToFP16_neon_asm(
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&output[0]),
			int64(neonCount),
		)
		// Keep slices alive until after assembly completes
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		// Handle remainder with scalar
		for idx := neonCount; idx < n; idx++ {
			output[idx] = float16.Fromfloat32(input[idx])
		}
	} else {
		// Scalar fallback for small slices
		for idx, value := range input {
			output[idx] = float16.Fromfloat32(value)
		}
	}
}

// convertFloat16SliceToFloat32 converts a Float16 slice to Float32 using NEON when available.
func convertFloat16SliceToFloat32(input []float16.Float16, output []float32) {
	n := len(input)
	if n >= minConversionNEONSize {
		// Process multiples of 8 with NEON
		neonCount := (n / 8) * 8
		convertFP16ToFP32_neon_asm(
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&output[0]),
			int64(neonCount),
		)
		// Keep slices alive until after assembly completes
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		// Handle remainder with scalar
		for idx := neonCount; idx < n; idx++ {
			output[idx] = input[idx].Float32()
		}
	} else {
		// Scalar fallback for small slices
		for idx, value := range input {
			output[idx] = value.Float32()
		}
	}
}

func init() {
	// Override Float16↔Float32 conversions with NEON-accelerated versions.
	// priorityArch overrides priorityTyped fallback in exec_special_ops.go
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Float32, priorityArch, execConvertDTypeFloat16ToFloat32NEON)
	convertDTypePairMap.Register(dtypes.Float32, dtypes.Float16, priorityArch, execConvertDTypeFloat32ToFloat16NEON)
}
