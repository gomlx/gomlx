// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build benchmark && !noasm && arm64

package simplego

import (
	"fmt"
	"testing"
)

// dotProductScalarGroup4 computes 4 dot products using pure Go scalar loop.
// This serves as the baseline for benchmarking.
func dotProductScalarGroup4(lhs, rhs []float32, blockDim int) (sum0, sum1, sum2, sum3 float32) {
	lhsIdx := 0
	rhsIdx := 0
	for i := 0; i < blockDim; i++ {
		l := lhs[lhsIdx]
		sum0 += l * rhs[rhsIdx]
		sum1 += l * rhs[rhsIdx+blockDim]
		sum2 += l * rhs[rhsIdx+2*blockDim]
		sum3 += l * rhs[rhsIdx+3*blockDim]
		lhsIdx++
		rhsIdx++
	}
	return
}

func BenchmarkDotProductImplementations(b *testing.B) {
	// Vector lengths to benchmark
	sizes := []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536}

	for _, size := range sizes {
		// Setup data
		// LHS: 1 vector of length 'size'
		// RHS: 4 vectors of length 'size', stored as columns in a block [size, 4] (conceptually)
		// But the memory layout expected by our kernels is:
		// Vector 0 starts at rhs[0], Vector 1 at rhs[size], etc.
		// So we need a buffer of size * 4.
		lhs := make([]float32, size)
		rhs := make([]float32, size*4)
		output := make([]float32, 4) // Dummy output for accumulation

		// Initialize with some data to avoid denormal issues (though unlikely with integers)
		for i := range lhs {
			lhs[i] = 1.0
		}
		for i := range rhs {
			rhs[i] = 1.0
		}

		// Scalar Baseline
		b.Run(fmt.Sprintf("Scalar/size_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _, _, _ = dotProductScalarGroup4(lhs, rhs, size)
			}
		})

		// NEON Optimization (Group 4)
		if hasNEON {
			b.Run(fmt.Sprintf("NEON/size_%d", size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_, _, _, _ = dotProductInnerLoopNEON(lhs, rhs, output, 0, 0, 0, size)
				}
			})
		}

	}
}

// BenchmarkDotProductSingle benchmarks the single-vector dot product (old style)
// just for comparison to see how much Group4 helps vs doing 4 individual calls.
func BenchmarkDotProductSingleComparison(b *testing.B) {
	size := 1024
	lhs := make([]float32, size)
	rhs := make([]float32, size)
	for i := range lhs {
		lhs[i] = 1.0
		rhs[i] = 1.0
	}

	b.Run("Scalar_Single_1024", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var sum float32
			for j := 0; j < size; j++ {
				sum += lhs[j] * rhs[j]
			}
		}
	})

	if hasNEON {
		b.Run("NEON_Single_1024", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = dotProduct_neon(lhs, rhs, 0, 0, int64(size))
			}
		})
	}
	
}
