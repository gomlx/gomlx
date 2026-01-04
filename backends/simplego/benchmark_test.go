package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BenchmarkMatMulExecution benchmarks matrix multiplication execution.
func BenchmarkMatMulExecution(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	sizes := []struct {
		name string
		m, k, n int
	}{
		{"64x64x64", 64, 64, 64},
		{"128x128x128", 128, 128, 128},
		{"256x256x256", 256, 256, 256},
		{"512x512x512", 512, 512, 512},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			builder := backend.Builder("benchmark_matmul")

			lhsShape := shapes.Make(dtypes.Float32, size.m, size.k)
			rhsShape := shapes.Make(dtypes.Float32, size.k, size.n)

			lhs, _ := builder.Parameter("lhs", lhsShape, nil)
			rhs, _ := builder.Parameter("rhs", rhsShape, nil)

			result, _ := builder.DotGeneral(
				lhs,
				[]int{1}, []int{},
				rhs,
				[]int{0}, []int{},
			)

			exec, err := builder.Compile([]backends.Op{result}, nil)
			if err != nil {
				b.Fatalf("Compile failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers
			lhsData := make([]float32, size.m*size.k)
			rhsData := make([]float32, size.k*size.n)
			for i := range lhsData {
				lhsData[i] = float32(i) * 0.001
			}
			for i := range rhsData {
				rhsData[i] = float32(i) * 0.001
			}

			lhsBuf, _ := backend.BufferFromFlatData(0, lhsData, lhsShape)
			rhsBuf, _ := backend.BufferFromFlatData(0, rhsData, rhsShape)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
				if err != nil {
					b.Fatalf("Execute failed: %v", err)
				}
				_ = outputs
			}
		})
	}
}

// BenchmarkBatchedMatMul benchmarks batched matrix multiplication.
func BenchmarkBatchedMatMul(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New failed: %v", err)
	}
	defer backend.Finalize()

	sizes := []struct {
		name    string
		batch, m, k, n int
	}{
		{"1x64x64x64", 1, 64, 64, 64},
		{"4x64x64x64", 4, 64, 64, 64},
		{"8x128x128x128", 8, 128, 128, 128},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			builder := backend.Builder("benchmark_batched_matmul")

			lhsShape := shapes.Make(dtypes.Float32, size.batch, size.m, size.k)
			rhsShape := shapes.Make(dtypes.Float32, size.batch, size.k, size.n)

			lhs, _ := builder.Parameter("lhs", lhsShape, nil)
			rhs, _ := builder.Parameter("rhs", rhsShape, nil)

			result, _ := builder.DotGeneral(
				lhs,
				[]int{2}, []int{0},
				rhs,
				[]int{1}, []int{0},
			)

			exec, err := builder.Compile([]backends.Op{result}, nil)
			if err != nil {
				b.Fatalf("Compile failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers
			lhsData := make([]float32, size.batch*size.m*size.k)
			rhsData := make([]float32, size.batch*size.k*size.n)
			for i := range lhsData {
				lhsData[i] = float32(i) * 0.001
			}
			for i := range rhsData {
				rhsData[i] = float32(i) * 0.001
			}

			lhsBuf, _ := backend.BufferFromFlatData(0, lhsData, lhsShape)
			rhsBuf, _ := backend.BufferFromFlatData(0, rhsData, rhsShape)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
				if err != nil {
					b.Fatalf("Execute failed: %v", err)
				}
				_ = outputs
			}
		})
	}
}
