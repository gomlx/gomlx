// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/x448/float16"
)

func BenchmarkDotGeneralPaths(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires SimpleGo backend")
	}

	// Matrix sizes typical for LLM inference
	sizes := []struct {
		name    string
		M, K, N int
	}{
		{"Small_64x128x64", 64, 128, 64},
		{"Medium_256x512x256", 256, 512, 256},
		{"Large_512x1024x512", 512, 1024, 512},
	}

	dtypeTests := []struct {
		name  string
		dtype dtypes.DType
	}{
		{"Float32", dtypes.Float32},
		{"Float16", dtypes.Float16},
		{"BFloat16", dtypes.BFloat16},
	}

	for _, sizeTest := range sizes {
		for _, dtypeTest := range dtypeTests {
			// Create test tensors
			M, K, N := sizeTest.M, sizeTest.K, sizeTest.N
			var lhs, rhs *tensors.Tensor

			switch dtypeTest.dtype {
			case dtypes.Float32:
				lhsData := make([]float32, M*K)
				rhsData := make([]float32, K*N)
				for i := range lhsData {
					lhsData[i] = float32(i%100) * 0.01
				}
				for i := range rhsData {
					rhsData[i] = float32(i%100) * 0.01
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			case dtypes.Float16:
				lhsData := make([]float16.Float16, M*K)
				rhsData := make([]float16.Float16, K*N)
				for i := range lhsData {
					lhsData[i] = float16.Fromfloat32(float32(i%100) * 0.01)
				}
				for i := range rhsData {
					rhsData[i] = float16.Fromfloat32(float32(i%100) * 0.01)
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			case dtypes.BFloat16:
				lhsData := make([]bfloat16.BFloat16, M*K)
				rhsData := make([]bfloat16.BFloat16, K*N)
				for i := range lhsData {
					lhsData[i] = bfloat16.FromFloat32(float32(i%100) * 0.01)
				}
				for i := range rhsData {
					rhsData[i] = bfloat16.FromFloat32(float32(i%100) * 0.01)
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			}

			// Test each path
			paths := []struct {
				name string
				path dotGeneralExecutionPath
				skip func() bool
			}{
				{"normalized", normalizedPath, func() bool { return false }},
				{"blocked", blockedPath, func() bool { return false }},
				{"packgemm", packgemmPath, func() bool {
					return !goBackend.enablePackgemm || !packgemm.HasDTypeSupport(dtypeTest.dtype, dtypeTest.dtype)
				}},
				{"highway", highwayPath, func() bool {
					return !Highway.HasDTypeSupport(dtypeTest.dtype, dtypeTest.dtype)
				}},
			}

			for _, pathTest := range paths {
				if pathTest.skip() {
					continue
				}

				benchName := fmt.Sprintf("%s/%s/%s", sizeTest.name, dtypeTest.name, pathTest.name)
				b.Run(benchName, func(b *testing.B) {
					goBackend.dotGeneralForceExecutionPath = pathTest.path

					exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
						return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
					})

					flops := float64(2 * M * K * N)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						exec.MustExec(lhs, rhs)
					}
					b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
				})
			}
		}
	}

	// Reset to default
	goBackend.dotGeneralForceExecutionPath = autoSelectPath
}

// BenchmarkDotGeneralMultiCrossDim benchmarks multi-cross-dimension patterns like
// "bsi,oi->bso" where LHS has multiple cross dimensions (batch, seq).
// Highway can handle this pattern with MatMulKLast when K is last in both operands.
func BenchmarkDotGeneralMultiCrossDim(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires SimpleGo backend")
	}

	// Dense layer sizes typical for transformers (MLP layers)
	sizes := []struct {
		name                                string
		batch, seq, inFeatures, outFeatures int
	}{
		{"Small_1x32x256x512", 1, 32, 256, 512},
		{"Medium_4x128x768x3072", 4, 128, 768, 3072},   // BERT-like MLP
		{"Large_1x512x1024x4096", 1, 512, 1024, 4096}, // Large transformer
	}

	for _, sizeTest := range sizes {
		batch, seq, inFeatures, outFeatures := sizeTest.batch, sizeTest.seq, sizeTest.inFeatures, sizeTest.outFeatures

		// Create test tensors: LHS [batch, seq, in], RHS [out, in] (PyTorch format)
		lhsData := make([]float32, batch*seq*inFeatures)
		rhsData := make([]float32, outFeatures*inFeatures)
		for i := range lhsData {
			lhsData[i] = float32(i%100) * 0.01
		}
		for i := range rhsData {
			rhsData[i] = float32(i%100) * 0.01
		}
		lhs := tensors.FromFlatDataAndDimensions(lhsData, batch, seq, inFeatures)
		rhs := tensors.FromFlatDataAndDimensions(rhsData, outFeatures, inFeatures)

		b.Run(fmt.Sprintf("%s/normalized", sizeTest.name), func(b *testing.B) {
			goBackend.dotGeneralForceExecutionPath = normalizedPath

			exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.Einsum("bsi,oi->bso", lhs, rhs)
			})

			flops := float64(2 * batch * seq * inFeatures * outFeatures)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				exec.MustExec(lhs, rhs)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})

		b.Run(fmt.Sprintf("%s/blocked", sizeTest.name), func(b *testing.B) {
			goBackend.dotGeneralForceExecutionPath = blockedPath

			exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.Einsum("bsi,oi->bso", lhs, rhs)
			})

			flops := float64(2 * batch * seq * inFeatures * outFeatures)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				exec.MustExec(lhs, rhs)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})

		// Highway path - uses MatMulKLast for K-last patterns
		if Highway.HasDTypeSupport(dtypes.Float32, dtypes.Float32) {
			b.Run(fmt.Sprintf("%s/highway", sizeTest.name), func(b *testing.B) {
				goBackend.dotGeneralForceExecutionPath = highwayPath

				exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.Einsum("bsi,oi->bso", lhs, rhs)
				})

				flops := float64(2 * batch * seq * inFeatures * outFeatures)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					exec.MustExec(lhs, rhs)
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}

	// Reset to default
	goBackend.dotGeneralForceExecutionPath = autoSelectPath
}

// BenchmarkDotGeneralKLast2D benchmarks the 2D K-last pattern where highway CAN be used:
// [M, K] × [N, K] → [M, N] (K is last in both operands)
// This pattern is compatible with highway's MatMulKLast.
func BenchmarkDotGeneralKLast2D(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires SimpleGo backend")
	}

	// Matrix sizes where K is last in both
	sizes := []struct {
		name    string
		M, K, N int
	}{
		{"Small_512x768x512", 512, 768, 512},
		{"Medium_1024x768x3072", 1024, 768, 3072},
		{"Large_2048x1024x4096", 2048, 1024, 4096},
	}

	for _, sizeTest := range sizes {
		M, K, N := sizeTest.M, sizeTest.K, sizeTest.N

		// LHS [M, K], RHS [N, K] - both have K last
		lhsData := make([]float32, M*K)
		rhsData := make([]float32, N*K)
		for i := range lhsData {
			lhsData[i] = float32(i%100) * 0.01
		}
		for i := range rhsData {
			rhsData[i] = float32(i%100) * 0.01
		}
		lhs := tensors.FromFlatDataAndDimensions(lhsData, M, K)
		rhs := tensors.FromFlatDataAndDimensions(rhsData, N, K)

		b.Run(fmt.Sprintf("%s/normalized", sizeTest.name), func(b *testing.B) {
			goBackend.dotGeneralForceExecutionPath = normalizedPath

			exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				// mi,ni->mn: contract on last dim of both
				return graph.Einsum("mi,ni->mn", lhs, rhs)
			})

			flops := float64(2 * M * K * N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				exec.MustExec(lhs, rhs)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})

		b.Run(fmt.Sprintf("%s/blocked", sizeTest.name), func(b *testing.B) {
			goBackend.dotGeneralForceExecutionPath = blockedPath

			exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.Einsum("mi,ni->mn", lhs, rhs)
			})

			flops := float64(2 * M * K * N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				exec.MustExec(lhs, rhs)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})

		if Highway.HasDTypeSupport(dtypes.Float32, dtypes.Float32) {
			b.Run(fmt.Sprintf("%s/highway", sizeTest.name), func(b *testing.B) {
				goBackend.dotGeneralForceExecutionPath = highwayPath

				exec := graph.MustNewExec(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.Einsum("mi,ni->mn", lhs, rhs)
				})

				flops := float64(2 * M * K * N)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					exec.MustExec(lhs, rhs)
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}

	// Reset to default
	goBackend.dotGeneralForceExecutionPath = autoSelectPath
}

func BenchmarkDotGeneralAutoSelect(b *testing.B) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		b.Skip("Test requires SimpleGo backend")
	}

	sizes := []struct {
		name    string
		M, K, N int
	}{
		{"256x512x256", 256, 512, 256},
		{"512x1024x512", 512, 1024, 512},
	}

	dtypeTests := []struct {
		name  string
		dtype dtypes.DType
	}{
		{"Float32", dtypes.Float32},
		{"Float16", dtypes.Float16},
		{"BFloat16", dtypes.BFloat16},
	}

	for _, sizeTest := range sizes {
		for _, dtypeTest := range dtypeTests {
			M, K, N := sizeTest.M, sizeTest.K, sizeTest.N
			var lhs, rhs *tensors.Tensor

			switch dtypeTest.dtype {
			case dtypes.Float32:
				lhsData := make([]float32, M*K)
				rhsData := make([]float32, K*N)
				for i := range lhsData {
					lhsData[i] = float32(i%100) * 0.01
				}
				for i := range rhsData {
					rhsData[i] = float32(i%100) * 0.01
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			case dtypes.Float16:
				lhsData := make([]float16.Float16, M*K)
				rhsData := make([]float16.Float16, K*N)
				for i := range lhsData {
					lhsData[i] = float16.Fromfloat32(float32(i%100) * 0.01)
				}
				for i := range rhsData {
					rhsData[i] = float16.Fromfloat32(float32(i%100) * 0.01)
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			case dtypes.BFloat16:
				lhsData := make([]bfloat16.BFloat16, M*K)
				rhsData := make([]bfloat16.BFloat16, K*N)
				for i := range lhsData {
					lhsData[i] = bfloat16.FromFloat32(float32(i%100) * 0.01)
				}
				for i := range rhsData {
					rhsData[i] = bfloat16.FromFloat32(float32(i%100) * 0.01)
				}
				lhs = tensors.FromFlatDataAndDimensions(lhsData, M, K)
				rhs = tensors.FromFlatDataAndDimensions(rhsData, K, N)
			}

			benchName := fmt.Sprintf("%s/%s/auto", sizeTest.name, dtypeTest.name)
			b.Run(benchName, func(b *testing.B) {
				goBackend.dotGeneralForceExecutionPath = autoSelectPath

				exec := graph.MustNewExec(goBackend, func(g *graph.Graph) *graph.Node {
					lhsNode := graph.Parameter(g, "lhs", shapes.Make(dtypeTest.dtype, M, K))
					rhsNode := graph.Parameter(g, "rhs", shapes.Make(dtypeTest.dtype, K, N))
					return graph.DotGeneral(lhsNode, []int{1}, []int{}, rhsNode, []int{0}, []int{})
				})

				flops := float64(2 * M * K * N)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					exec.MustExec(lhs, rhs)
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}
