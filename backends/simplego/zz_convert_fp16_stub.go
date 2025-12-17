//go:build noasm || !arm64

package simplego

import "github.com/x448/float16"

// convertFloat32SliceToFloat16 converts a Float32 slice to Float16 (scalar fallback).
func convertFloat32SliceToFloat16(input []float32, output []float16.Float16) {
	for idx, value := range input {
		output[idx] = float16.Fromfloat32(value)
	}
}

// convertFloat16SliceToFloat32 converts a Float16 slice to Float32 (scalar fallback).
func convertFloat16SliceToFloat32(input []float16.Float16, output []float32) {
	for idx, value := range input {
		output[idx] = value.Float32()
	}
}
