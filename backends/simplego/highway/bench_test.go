// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/matmul"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// BenchmarkHighwayVsNormalized compares highway matmul with other paths
// for the linear layer pattern that was causing performance issues.
func BenchmarkLinearLayerPattern(b *testing.B) {
	// Linear layer: [1, 11, 1024] × [1024, 1024] → [1, 11, 1024]
	// This tests the pattern: input @ weights for a transformer layer

	inputData := make([]float32, 1*11*1024)
	weightsData := make([]float32, 1024*1024)
	for i := range inputData {
		inputData[i] = float32(i%100) * 0.01
	}
	for i := range weightsData {
		weightsData[i] = float32(i%100) * 0.001
	}

	b.Run("via_graph", func(b *testing.B) {
		execFn := func(g *graph.Graph) *graph.Node {
			input := graph.Reshape(graph.Const(g, inputData), 1, 11, 1024)
			weights := graph.Reshape(graph.Const(g, weightsData), 1024, 1024)
			return graph.Einsum("bsk,kh->bsh", input, weights)
		}

		exec := graph.MustNewExec(backend, execFn)

		// Warmup
		exec.MustExec()

		flops := float64(2 * 11 * 1024 * 1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			exec.MustExec()
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})

	b.Run("direct_highway_matmul", func(b *testing.B) {
		// Direct highway matmul: [11, 1024] × [1024, 1024] → [11, 1024]
		lhs := make([]float32, 11*1024)
		rhs := make([]float32, 1024*1024)
		out := make([]float32, 11*1024)
		for i := range lhs {
			lhs[i] = float32(i%100) * 0.01
		}
		for i := range rhs {
			rhs[i] = float32(i%100) * 0.001
		}

		// Warmup
		matmul.MatMulAuto(lhs, rhs, out, 11, 1024, 1024)

		flops := float64(2 * 11 * 1024 * 1024)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			matmul.MatMulAuto(lhs, rhs, out, 11, 1024, 1024)
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}

// BenchmarkMatMulSizes compares highway matmul performance at different sizes
func BenchmarkMatMulSizes(b *testing.B) {
	sizes := []struct {
		name    string
		m, n, k int
	}{
		{"11x1024x1024", 11, 1024, 1024},     // Linear layer (small M)
		{"128x1024x1024", 128, 1024, 1024},   // Larger batch
		{"512x512x512", 512, 512, 512},       // Square
		{"1024x1024x1024", 1024, 1024, 1024}, // Large square
	}

	for _, sz := range sizes {
		lhs := make([]float32, sz.m*sz.k)
		rhs := make([]float32, sz.k*sz.n)
		out := make([]float32, sz.m*sz.n)
		for i := range lhs {
			lhs[i] = float32(i%100) * 0.01
		}
		for i := range rhs {
			rhs[i] = float32(i%100) * 0.001
		}

		b.Run(sz.name, func(b *testing.B) {
			// Warmup
			matmul.MatMulAuto(lhs, rhs, out, sz.m, sz.n, sz.k)

			flops := float64(2 * sz.m * sz.n * sz.k)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmul.MatMulAuto(lhs, rhs, out, sz.m, sz.n, sz.k)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
