//go:build !noasm && arm64

package simplego

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/x448/float16"
)

func sizeStr(n int) string {
	return fmt.Sprintf("%d", n)
}

func TestConvertFP16ToFP32NEON(t *testing.T) {
	sizes := []int{8, 16, 64, 100, 127, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			input := make([]float16.Float16, size)
			output := make([]float32, size)
			expected := make([]float32, size)

			// Fill with test data
			for i := range input {
				val := float32(i) * 0.5
				input[i] = float16.Fromfloat32(val)
				expected[i] = input[i].Float32() // Get expected via scalar conversion
			}

			// Run NEON conversion via wrapper (handles remainders correctly)
			neonCount := (size / 8) * 8
			if neonCount > 0 {
				convertFP16ToFP32_neon_asm(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&output[0]),
					int64(neonCount),
				)
			}
			// Handle remainder with scalar Go code
			for idx := neonCount; idx < size; idx++ {
				output[idx] = input[idx].Float32()
			}

			// Verify results
			for i := 0; i < size; i++ {
				if math.Abs(float64(expected[i]-output[i])) > 0.001 {
					t.Errorf("FP16→FP32[%d]: expected %f, got %f", i, expected[i], output[i])
					if i > 10 {
						t.Fatalf("Too many errors, stopping")
					}
				}
			}
		})
	}
}

func TestConvertFP32ToFP16NEON(t *testing.T) {
	sizes := []int{8, 16, 64, 100, 127, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			input := make([]float32, size)
			output := make([]float16.Float16, size)
			expected := make([]float16.Float16, size)

			// Fill with test data
			for i := range input {
				input[i] = float32(i) * 0.5
				expected[i] = float16.Fromfloat32(input[i]) // Get expected via scalar conversion
			}

			// Run NEON conversion via wrapper (handles remainders correctly)
			neonCount := (size / 8) * 8
			if neonCount > 0 {
				convertFP32ToFP16_neon_asm(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&output[0]),
					int64(neonCount),
				)
			}
			// Handle remainder with scalar Go code
			for idx := neonCount; idx < size; idx++ {
				output[idx] = float16.Fromfloat32(input[idx])
			}

			// Verify results
			for i := 0; i < size; i++ {
				exp := expected[i].Float32()
				got := output[i].Float32()
				if math.Abs(float64(exp-got)) > 0.001 {
					t.Errorf("FP32→FP16[%d]: expected %f, got %f", i, exp, got)
					if i > 10 {
						t.Fatalf("Too many errors, stopping")
					}
				}
			}
		})
	}
}

func BenchmarkConvertFP16ToFP32(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		input := make([]float16.Float16, size)
		outputNEON := make([]float32, size)
		outputScalar := make([]float32, size)

		for i := range input {
			input[i] = float16.Fromfloat32(float32(i%100) / 10.0)
		}

		b.Run("NEON_"+sizeStr(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convertFP16ToFP32_neon_asm(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&outputNEON[0]),
					int64(size),
				)
			}
		})

		b.Run("Scalar_"+sizeStr(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j := 0; j < size; j++ {
					outputScalar[j] = input[j].Float32()
				}
			}
		})
	}
}

func BenchmarkConvertFP32ToFP16(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		input := make([]float32, size)
		outputNEON := make([]float16.Float16, size)
		outputScalar := make([]float16.Float16, size)

		for i := range input {
			input[i] = float32(i%100) / 10.0
		}

		b.Run("NEON_"+sizeStr(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convertFP32ToFP16_neon_asm(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&outputNEON[0]),
					int64(size),
				)
			}
		})

		b.Run("Scalar_"+sizeStr(size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j := 0; j < size; j++ {
					outputScalar[j] = float16.Fromfloat32(input[j])
				}
			}
		})
	}
}
