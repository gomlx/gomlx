//go:build benchmark

package simplego

import (
	"fmt"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/x448/float16"
)

// Benchmark tests for DotGeneral optimizations
// Run with: go test -bench=BenchmarkDotGeneral -benchmem -count=3

// BenchmarkDotGeneralMatMul benchmarks standard matrix multiplication
func BenchmarkDotGeneralMatMul(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	sizes := []struct {
		M, K, N int
	}{
		{64, 64, 64},      // Small matrices
		{128, 128, 128},   // Medium matrices
		{256, 256, 256},   // Large matrices
		{512, 768, 512},   // Transformer-like (attention)
		{1024, 1024, 1024}, // Very large
		{1, 768, 768},     // Single row (common in inference)
		{32, 768, 3072},   // Typical MLP layer
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("M%d_K%d_N%d", size.M, size.K, size.N), func(b *testing.B) {
			// Create input buffers
			lhsShape := shapes.Make(dtypes.Float32, size.M, size.K)
			rhsShape := shapes.Make(dtypes.Float32, size.K, size.N)
			outputShape := shapes.Make(dtypes.Float32, size.M, size.N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)
			output := backend.NewBuffer(outputShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       size.M,
				rhsCrossSize:       size.N,
				contractingSize:    size.K,
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				output.Zeros()
				_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
			}

			// Report GFLOPS
			flops := 2 * int64(size.M) * int64(size.K) * int64(size.N) // multiply-add = 2 ops
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkDotGeneralFastPathVsNormalized compares fast path vs normalized path
func BenchmarkDotGeneralFastPathVsNormalized(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	M, K, N := 256, 256, 256

	lhsShape := shapes.Make(dtypes.Float32, M, K)
	rhsShape := shapes.Make(dtypes.Float32, K, N)
	outputShape := shapes.Make(dtypes.Float32, M, N)

	lhs := backend.NewBuffer(lhsShape)
	rhs := backend.NewBuffer(rhsShape)

	// Fill with test data
	lhsFlat := lhs.flat.([]float32)
	rhsFlat := rhs.flat.([]float32)
	for i := range lhsFlat {
		lhsFlat[i] = float32(i%100) / 100.0
	}
	for i := range rhsFlat {
		rhsFlat[i] = float32(i%100) / 100.0
	}

	params := &dotGeneralNodeData{
		lhsContractingAxes: []int{1},
		rhsContractingAxes: []int{0},
		lhsBatchAxes:       []int{},
		rhsBatchAxes:       []int{},
		batchSize:          1,
		lhsCrossSize:       M,
		rhsCrossSize:       N,
		contractingSize:    K,
	}

	b.Run("DirectPath", func(b *testing.B) {
		output := backend.NewBuffer(outputShape)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			output.Zeros()
			execDotGeneralSmallMatMulFloat32(backend, lhs, rhs, params, output)
		}
	})

	b.Run("NormalizedPath", func(b *testing.B) {
		output := backend.NewBuffer(outputShape)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			output.Zeros()
			_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
		}
	})
}

// BenchmarkNEONDotProduct benchmarks NEON vs scalar dot product
func BenchmarkNEONDotProduct(b *testing.B) {
	if !hasNEON {
		b.Skip("NEON not available")
	}

	sizes := []int{64, 256, 768, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			a := make([]float32, size)
			bVec := make([]float32, size)
			for i := range a {
				a[i] = float32(i) / float32(size)
				bVec[i] = float32(size-i) / float32(size)
			}

			b.Run("NEON", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = dotProduct_neon_asm(unsafe.Pointer(&a[0]), unsafe.Pointer(&bVec[0]), int64(size))
				}
			})

			b.Run("Scalar", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var sum float32
					for j := 0; j < size; j++ {
						sum += a[j] * bVec[j]
					}
					_ = sum
				}
			})
		})
	}
}

// BenchmarkBufferPool benchmarks exact-size buffer pooling
func BenchmarkBufferPool(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Simulate various allocation sizes typical in ML workloads
	allocationSizes := []int{
		256,   // Small
		1024,  // 1K
		4096,  // 4K
		16384, // 16K
		65536, // 64K
	}

	b.Run("ExactSize", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			size := allocationSizes[i%len(allocationSizes)]
			shape := shapes.Make(dtypes.Float32, size)
			buf := backend.getBufferForShape(shape)
			backend.putBuffer(buf)
		}
	})
}

// BenchmarkBatchedMatMul benchmarks batched matrix multiplication
func BenchmarkBatchedMatMul(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	batchSizes := []int{1, 4, 16, 32}
	M, K, N := 128, 256, 128

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch%d", batchSize), func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.Float32, batchSize, M, K)
			rhsShape := shapes.Make(dtypes.Float32, batchSize, K, N)
			outputShape := shapes.Make(dtypes.Float32, batchSize, M, N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)
			output := backend.NewBuffer(outputShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{2},
				rhsContractingAxes: []int{1},
				lhsBatchAxes:       []int{0},
				rhsBatchAxes:       []int{0},
				batchSize:          batchSize,
				lhsCrossSize:       M,
				rhsCrossSize:       N,
				contractingSize:    K,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output.Zeros()
				_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
			}

			flops := 2 * int64(batchSize) * int64(M) * int64(K) * int64(N)
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkFP16Operations benchmarks FP16 dot product performance
func BenchmarkFP16Operations(b *testing.B) {
	if !hasFP16NEON {
		b.Skip("FP16 NEON not available")
	}

	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	sizes := []struct {
		M, K, N int
	}{
		{64, 64, 64},
		{128, 128, 128},
		{256, 256, 256},
		{512, 512, 512},
	}

	for _, size := range sizes {
		M, K, N := size.M, size.K, size.N
		b.Run(fmt.Sprintf("%dx%dx%d", M, K, N), func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.Float16, M, K)
			rhsShape := shapes.Make(dtypes.Float16, K, N)
			outputShape := shapes.Make(dtypes.Float16, M, N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)
			output := backend.NewBuffer(outputShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float16.Float16)
			rhsFlat := rhs.flat.([]float16.Float16)
			for i := range lhsFlat {
				lhsFlat[i] = float16.Fromfloat32(float32(i%100) / 100.0)
			}
			for i := range rhsFlat {
				rhsFlat[i] = float16.Fromfloat32(float32(i%100) / 100.0)
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       M,
				rhsCrossSize:       N,
				contractingSize:    K,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output.Zeros()
				_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
			}

			flops := 2 * int64(M) * int64(K) * int64(N)
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkBF16Operations benchmarks BF16 dot product performance
func BenchmarkBF16Operations(b *testing.B) {
	if !hasBF16NEON {
		b.Skip("BF16 NEON not available")
	}

	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	sizes := []struct {
		M, K, N int
	}{
		{64, 64, 64},
		{128, 128, 128},
		{256, 256, 256},
		{512, 512, 512},
	}

	for _, size := range sizes {
		M, K, N := size.M, size.K, size.N
		b.Run(fmt.Sprintf("%dx%dx%d", M, K, N), func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.BFloat16, M, K)
			rhsShape := shapes.Make(dtypes.BFloat16, K, N)
			outputShape := shapes.Make(dtypes.BFloat16, M, N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)
			output := backend.NewBuffer(outputShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]bfloat16.BFloat16)
			rhsFlat := rhs.flat.([]bfloat16.BFloat16)
			for i := range lhsFlat {
				lhsFlat[i] = bfloat16.FromFloat32(float32(i%100) / 100.0)
			}
			for i := range rhsFlat {
				rhsFlat[i] = bfloat16.FromFloat32(float32(i%100) / 100.0)
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       M,
				rhsCrossSize:       N,
				contractingSize:    K,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output.Zeros()
				_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
			}

			flops := 2 * int64(M) * int64(K) * int64(N)
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkF32Operations benchmarks Float32 dot product for comparison with FP16/BF16
func BenchmarkF32Operations(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	sizes := []struct {
		M, K, N int
	}{
		{64, 64, 64},
		{128, 128, 128},
		{256, 256, 256},
		{512, 512, 512},
	}

	for _, size := range sizes {
		M, K, N := size.M, size.K, size.N
		b.Run(fmt.Sprintf("%dx%dx%d", M, K, N), func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.Float32, M, K)
			rhsShape := shapes.Make(dtypes.Float32, K, N)
			outputShape := shapes.Make(dtypes.Float32, M, N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)
			output := backend.NewBuffer(outputShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       M,
				rhsCrossSize:       N,
				contractingSize:    K,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output.Zeros()
				_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
			}

			flops := 2 * int64(M) * int64(K) * int64(N)
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMemoryBandwidth estimates memory bandwidth utilization
func BenchmarkMemoryBandwidth(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Large matrix to measure memory bandwidth
	size := 1024 * 1024 // 1M elements = 4MB per matrix

	shape := shapes.Make(dtypes.Float32, size)
	src := backend.NewBuffer(shape)
	dst := backend.NewBuffer(shape)

	srcFlat := src.flat.([]float32)
	for i := range srcFlat {
		srcFlat[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst.flat.([]float32), srcFlat)
	}

	bytesTransferred := int64(size) * 4 * 2 // read + write
	gbPerSec := float64(bytesTransferred) * float64(b.N) / b.Elapsed().Seconds() / 1e9
	b.ReportMetric(gbPerSec, "GB/s")
}

// TestDotGeneralOptimizationReport prints a summary of optimization capabilities
func TestDotGeneralOptimizationReport(t *testing.T) {
	fmt.Println("=== GoMLX SimpleGo Backend Optimization Report ===")
	fmt.Printf("Architecture: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("NEON available: %v\n", hasNEON)
	fmt.Printf("FP16 NEON available: %v\n", hasFP16NEON)
	fmt.Printf("BF16 NEON available: %v\n", hasBF16NEON)

	// Quick sanity check for NEON
	if hasNEON {
		a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		b := []float32{1, 1, 1, 1, 1, 1, 1, 1}
		result := dotProduct_neon_asm(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), 8)
		expected := float32(36) // 1+2+3+4+5+6+7+8
		if result != expected {
			t.Errorf("NEON dot product sanity check failed: got %f, expected %f", result, expected)
		} else {
			fmt.Printf("\nNEON sanity check: PASSED (dot([1..8], [1..1]) = %f)\n", result)
		}
	}

	fmt.Println("\n=== End Report ===")
}

// BenchmarkOptimizationOverhead measures the overhead of optimization checks
func BenchmarkOptimizationOverhead(b *testing.B) {
	// Measure the cost of checking if we can use fast path
	lhsShape := shapes.Make(dtypes.Float32, 256, 256)
	rhsShape := shapes.Make(dtypes.Float32, 256, 256)

	b.Run("isMatMulOrder", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = isMatMulOrder(lhsShape, rhsShape, []int{1}, []int{0}, []int{}, []int{})
		}
	})
}

// Benchmark comparing warmup vs cold start
func BenchmarkWarmupEffect(b *testing.B) {
	sizes := []int{256, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			b.Run("Cold", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					backendIface, _ := New("")
					backend := backendIface.(*Backend)
					lhsShape := shapes.Make(dtypes.Float32, size, size)
					_ = backend.NewBuffer(lhsShape)
					b.StartTimer()

					// Just measure buffer allocation in cold state
					_ = backend.getBufferForShape(lhsShape)

					b.StopTimer()
					backendIface.Finalize()
				}
			})

			b.Run("Warm", func(b *testing.B) {
				backendIface, _ := New("")
				defer backendIface.Finalize()
				backend := backendIface.(*Backend)
				lhsShape := shapes.Make(dtypes.Float32, size, size)

				// Warmup
				for j := 0; j < 10; j++ {
					buf := backend.getBufferForShape(lhsShape)
					backend.putBuffer(buf)
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					buf := backend.getBufferForShape(lhsShape)
					backend.putBuffer(buf)
				}
			})
		})
	}
}

// Helper to measure time taken for a specific operation
func measureTime(name string, iterations int, fn func()) time.Duration {
	start := time.Now()
	for i := 0; i < iterations; i++ {
		fn()
	}
	elapsed := time.Since(start)
	fmt.Printf("%s: %v per iteration (total: %v for %d iterations)\n",
		name, elapsed/time.Duration(iterations), elapsed, iterations)
	return elapsed
}

// BenchmarkPreBlockedWeights compares pre-blocked vs standard path for weight matrices
func BenchmarkPreBlockedWeights(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Typical embedding/transformer weight sizes
	sizes := []struct {
		M, K, N int
		name    string
	}{
		{1, 768, 768, "embedding_single"},
		{32, 768, 768, "embedding_batch32"},
		{1, 768, 3072, "ffn_up_single"},
		{32, 768, 3072, "ffn_up_batch32"},
		{1, 3072, 768, "ffn_down_single"},
		{32, 3072, 768, "ffn_down_batch32"},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create weight matrix [K, N] and activation [M, K]
			// Note: For direct execDotGeneralBlocked calls, we need rank 3 shapes [batch, cross, contract/cross]
			lhsShape := shapes.Make(dtypes.Float32, 1, size.M, size.K)
			rhsShape := shapes.Make(dtypes.Float32, 1, size.K, size.N)
			outputShape := shapes.Make(dtypes.Float32, 1, size.M, size.N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{2}, // For rank 3 [1, M, K], contracting is axis 2
				rhsContractingAxes: []int{1}, // For rank 3 [1, K, N], contracting is axis 1
				lhsBatchAxes:       []int{0}, // Batch is axis 0
				rhsBatchAxes:       []int{0}, // Batch is axis 0
				batchSize:          1,
				lhsCrossSize:       size.M,
				rhsCrossSize:       size.N,
				contractingSize:    size.K,
			}

			// Set up blocked shapes
			blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float32]
			params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
			params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
			params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

			b.Run("Standard", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					_ = execDotGeneralBlocked(backend, lhs, rhs, params, output)
				}

				flops := 2 * int64(size.M) * int64(size.K) * int64(size.N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})

			b.Run("PreBlocked", func(b *testing.B) {
				// Pre-block the weight - requires rank 2 shape [K, N]
				rhsRank2Shape := shapes.Make(dtypes.Float32, size.K, size.N)
				rhsRank2 := backend.NewBuffer(rhsRank2Shape)
				rhsRank2Flat := rhsRank2.flat.([]float32)
				for i := range rhsRank2Flat {
					rhsRank2Flat[i] = float32(i%100) / 100.0
				}

				pbw := PreBlockTensorForMatMul(rhsRank2)
				if pbw == nil {
					b.Skip("Pre-blocking not supported for this shape")
				}

				// Create block data for the pre-blocked RHS
				blockData := &blockForDotGeneralData{
					blockLog2Dim:    pbw.BlockLog2Dim,
					blockedShape:    pbw.BlockedShape,
					batchSize:       1, // RHS has no batch dimension
					crossSize:       size.N,
					contractingSize: size.K,
					contractingAxes: []int{0}, // K is axis 0 in [K, N]
					batchAxes:       []int{},
				}

				preBlockedBuf := pbw.GetPreBlockedBuffer()
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					// Use unified blocked path with RHS pre-blocked (lhsBlockData=nil, rhsBlockData=blockData)
					_ = execDotGeneralBlockedUnified(backend, lhs, preBlockedBuf, nil, blockData, params, output)
				}

				flops := 2 * int64(size.M) * int64(size.K) * int64(size.N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})
		})
	}
}

// BenchmarkPreBlockingOverhead measures the one-time cost of pre-blocking weights
func BenchmarkPreBlockingOverhead(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	sizes := []struct {
		K, N int
		name string
	}{
		{768, 768, "768x768"},
		{768, 3072, "768x3072"},
		{3072, 768, "3072x768"},
		{4096, 4096, "4096x4096"},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			rhsShape := shapes.Make(dtypes.Float32, size.K, size.N)
			rhs := backend.NewBuffer(rhsShape)

			// Fill with test data
			rhsFlat := rhs.flat.([]float32)
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = PreBlockTensorForMatMul(rhs)
			}

			// Report memory overhead
			pbw := PreBlockTensorForMatMul(rhs)
			if pbw != nil {
				origBytes := int64(pbw.OriginalShape.Size()) * 4
				blockedBytes := int64(pbw.BlockedShape.Size()) * 4
				overhead := float64(blockedBytes-origBytes) / float64(origBytes) * 100
				b.ReportMetric(overhead, "%_memory_overhead")
			}
		})
	}
}

// BenchmarkDirectPathThreshold tests different contracting sizes to find the optimal
// smallMatMulMaxContractingSize threshold. It compares execDotGeneralSmallMatMulFloat32
// (small matmul/no-transpose path with strided RHS access) vs execDotGeneralSmallNormalized
// (transposes RHS for sequential access).
//
// Run with: go test -tags=benchmark -bench=BenchmarkDirectPathThreshold -benchtime=1s
func BenchmarkDirectPathThreshold(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Test various contracting sizes around and beyond the current threshold
	// (smallMatMulMaxContractingSize = 128) to find the optimal crossover point
	contractingSizes := []int{64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384}

	// Fixed M and N to isolate the effect of contracting size
	M, N := 256, 256

	for _, K := range contractingSizes {
		b.Run(fmt.Sprintf("K%d", K), func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.Float32, M, K)
			rhsShape := shapes.Make(dtypes.Float32, K, N)
			outputShape := shapes.Make(dtypes.Float32, M, N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)

			// Fill with test data
			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       M,
				rhsCrossSize:       N,
				contractingSize:    K,
			}

			// Benchmark the direct path (no transpose, strided RHS access)
			b.Run("DirectPath", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					execDotGeneralSmallMatMulFloat32(backend, lhs, rhs, params, output)
				}
				flops := 2 * int64(M) * int64(K) * int64(N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})

			// Benchmark the normalized path (transposes RHS for sequential access)
			b.Run("NormalizedPath", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
				}
				flops := 2 * int64(M) * int64(K) * int64(N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})
		})
	}
}

// BenchmarkDirectPathThresholdVariedShapes tests the threshold with different matrix shapes
// to ensure the threshold works well across common ML workloads.
func BenchmarkDirectPathThresholdVariedShapes(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Common ML workload shapes
	shapes_to_test := []struct {
		name string
		M, K, N int
	}{
		// Single row inference
		{"SingleRow_K768", 1, 768, 768},
		{"SingleRow_K1024", 1, 1024, 1024},
		{"SingleRow_K2048", 1, 2048, 2048},
		{"SingleRow_K4096", 1, 4096, 4096},

		// Small batch inference
		{"SmallBatch_K768", 32, 768, 768},
		{"SmallBatch_K1024", 32, 1024, 1024},
		{"SmallBatch_K2048", 32, 2048, 2048},
		{"SmallBatch_K4096", 32, 4096, 4096},

		// Square matrices around threshold
		{"Square_256x256", 256, 256, 256},
		{"Square_512x512", 512, 512, 512},
		{"Square_1024x1024", 1024, 1024, 1024},

		// FFN-style shapes
		{"FFN_Up", 32, 768, 3072},
		{"FFN_Down", 32, 3072, 768},
	}

	for _, shape := range shapes_to_test {
		b.Run(shape.name, func(b *testing.B) {
			lhsShape := shapes.Make(dtypes.Float32, shape.M, shape.K)
			rhsShape := shapes.Make(dtypes.Float32, shape.K, shape.N)
			outputShape := shapes.Make(dtypes.Float32, shape.M, shape.N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)

			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{1},
				rhsContractingAxes: []int{0},
				lhsBatchAxes:       []int{},
				rhsBatchAxes:       []int{},
				batchSize:          1,
				lhsCrossSize:       shape.M,
				rhsCrossSize:       shape.N,
				contractingSize:    shape.K,
			}

			b.Run("DirectPath", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					execDotGeneralSmallMatMulFloat32(backend, lhs, rhs, params, output)
				}
				flops := 2 * int64(shape.M) * int64(shape.K) * int64(shape.N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})

			b.Run("NormalizedPath", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					_ = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
				}
				flops := 2 * int64(shape.M) * int64(shape.K) * int64(shape.N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})
		})
	}
}

// BenchmarkPreBlockingThreshold tests when pre-blocking becomes beneficial
// compared to on-the-fly blocking in execDotGeneralBlocked.
func BenchmarkPreBlockingThreshold(b *testing.B) {
	backendIface, _ := New("")
	defer backendIface.Finalize()
	backend := backendIface.(*Backend)

	// Typical weight matrix sizes
	shapes_to_test := []struct {
		name string
		K, N int
	}{
		{"384x384", 384, 384},
		{"768x768", 768, 768},
		{"768x3072", 768, 3072},
		{"1024x1024", 1024, 1024},
		{"2048x2048", 2048, 2048},
		{"4096x4096", 4096, 4096},
	}

	M := 32 // Batch size for activations

	for _, shape := range shapes_to_test {
		b.Run(shape.name, func(b *testing.B) {
			// For blocked path, we need rank 3 shapes [batch, cross, contract]
			lhsShape := shapes.Make(dtypes.Float32, 1, M, shape.K)
			rhsShape := shapes.Make(dtypes.Float32, 1, shape.K, shape.N)
			outputShape := shapes.Make(dtypes.Float32, 1, M, shape.N)

			lhs := backend.NewBuffer(lhsShape)
			rhs := backend.NewBuffer(rhsShape)

			lhsFlat := lhs.flat.([]float32)
			rhsFlat := rhs.flat.([]float32)
			for i := range lhsFlat {
				lhsFlat[i] = float32(i%100) / 100.0
			}
			for i := range rhsFlat {
				rhsFlat[i] = float32(i%100) / 100.0
			}

			params := &dotGeneralNodeData{
				lhsContractingAxes: []int{2},
				rhsContractingAxes: []int{1},
				lhsBatchAxes:       []int{0},
				rhsBatchAxes:       []int{0},
				batchSize:          1,
				lhsCrossSize:       M,
				rhsCrossSize:       shape.N,
				contractingSize:    shape.K,
			}

			// Set up blocked shapes for blocked path
			blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtypes.Float32]
			params.lhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
			params.rhsBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
			params.outputBlockedShape = dgCreateBlockedShape(dtypes.Float32, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

			b.Run("OnTheFlyBlocking", func(b *testing.B) {
				output := backend.NewBuffer(outputShape)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output.Zeros()
					_ = execDotGeneralBlocked(backend, lhs, rhs, params, output)
				}
				flops := 2 * int64(M) * int64(shape.K) * int64(shape.N)
				gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
				b.ReportMetric(gflops, "GFLOPS")
			})

			// Pre-block the RHS weight matrix
			rhsRank2Shape := shapes.Make(dtypes.Float32, shape.K, shape.N)
			rhsRank2 := backend.NewBuffer(rhsRank2Shape)
			rhsRank2Flat := rhsRank2.flat.([]float32)
			for i := range rhsRank2Flat {
				rhsRank2Flat[i] = float32(i%100) / 100.0
			}

			pbw := PreBlockTensorForMatMul(rhsRank2)
			if pbw != nil {
				b.Run("PreBlocked", func(b *testing.B) {
					// Create block data for the pre-blocked RHS
					blockData := &blockForDotGeneralData{
						blockLog2Dim:    pbw.BlockLog2Dim,
						blockedShape:    pbw.BlockedShape,
						batchSize:       1, // RHS has no batch dimension
						crossSize:       shape.N,
						contractingSize: shape.K,
						contractingAxes: []int{0}, // K is axis 0 in [K, N]
						batchAxes:       []int{},
					}

					preBlockedBuf := pbw.GetPreBlockedBuffer()
					output := backend.NewBuffer(outputShape)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						output.Zeros()
						// Use unified blocked path with RHS pre-blocked
						_ = execDotGeneralBlockedUnified(backend, lhs, preBlockedBuf, nil, blockData, params, output)
					}
					flops := 2 * int64(M) * int64(shape.K) * int64(shape.N)
					gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
					b.ReportMetric(gflops, "GFLOPS")
				})
			}
		})
	}
}
