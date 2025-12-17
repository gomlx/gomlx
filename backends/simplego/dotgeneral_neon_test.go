// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && arm64

package simplego

import (
	"fmt"
	"testing"
)

// TestNEONDetection tests NEON feature detection
func TestNEONDetection(t *testing.T) {
	if hasNEON {
		t.Log("NEON detected - using 128-bit vectors")
	} else {
		t.Log("NEON not available")
	}
}

// TestDotProductNEON tests the NEON dot product implementation with progressive sizes
func TestDotProductNEON(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available on this system")
	}

	// Test with progressively larger sizes
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := make([]float32, size)
			b := make([]float32, size)

			for i := 0; i < size; i++ {
				a[i] = float32(i + 1)
				b[i] = float32(i + 1)
			}

			var expected float32
			for i := 0; i < size; i++ {
				expected += a[i] * b[i]
			}

			result := dotProduct_neon(a, b, 0, 0, int64(size))

			// Allow small floating point tolerance
			// SIMD accumulation order differs from scalar, causing small differences
			diff := result - expected
			if diff < 0 {
				diff = -diff
			}
			// Relative tolerance of 1e-5 is appropriate for float32 SIMD operations
			tolerance := expected * 1e-5
			if tolerance < 1e-5 {
				tolerance = 1e-5
			}

			if diff > tolerance {
				t.Errorf("Size %d failed: got %f, expected %f, diff %f", size, result, expected, diff)
			}
		})
	}
}

// TestDotProductNEONWithOffset tests NEON with non-zero offsets
func TestDotProductNEONWithOffset(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available on this system")
	}

	// Create arrays with padding
	a := make([]float32, 100)
	b := make([]float32, 100)

	for i := range a {
		a[i] = float32(i)
		b[i] = float32(i)
	}

	// Test with offset 10, length 50
	var expected float32
	for i := 10; i < 60; i++ {
		expected += a[i] * b[i]
	}

	result := dotProduct_neon(a, b, 10, 10, 50)

	diff := result - expected
	if diff < 0 {
		diff = -diff
	}
	if diff > expected*1e-6 {
		t.Errorf("Offset test failed: got %f, expected %f", result, expected)
	}
}

// BenchmarkDotProductNEON benchmarks the NEON implementation
func BenchmarkDotProductNEON(b *testing.B) {
	if !hasNEON {
		b.Skip("NEON not available on this system")
	}

	sizes := []int{64, 512, 2048, 8192}

	for _, size := range sizes {
		a := make([]float32, size)
		c := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = float32(i)
			c[i] = 2.0
		}

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = dotProduct_neon(a, c, 0, 0, int64(size))
			}
		})
	}
}
