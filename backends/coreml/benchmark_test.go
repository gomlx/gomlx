//go:build darwin && cgo

package coreml

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BenchmarkMatMulCompilation benchmarks matrix multiplication compilation time.
// Uses basic Dot operation which is implemented.
func BenchmarkMatMulCompilation(b *testing.B) {
	sizes := []struct {
		name    string
		m, k, n int
	}{
		{"64x64x64", 64, 64, 64},
		{"128x128x128", 128, 128, 128},
		{"256x256x256", 256, 256, 256},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				backend, err := New("")
				if err != nil {
					b.Fatalf("New() failed: %v", err)
				}

				builder := backend.Builder("bench_matmul")
				mainFn := builder.Main()

				// Create input shapes: [M, K] @ [K, N] -> [M, N]
				lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
				rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

				lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for lhs failed: %v", err)
				}

				rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for rhs failed: %v", err)
				}

				// Matrix multiplication using Dot
				result, err := mainFn.Dot(lhs, rhs)
				if err != nil {
					b.Fatalf("Dot() failed: %v", err)
				}

				if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
					b.Fatalf("Return() failed: %v", err)
				}

				b.StartTimer()
				exec, err := builder.Compile()
				b.StopTimer()

				if err != nil {
					b.Fatalf("Compile() failed: %v", err)
				}
				exec.Finalize()
				backend.Finalize()
			}
		})
	}
}

// BenchmarkMatMulExecution benchmarks matrix multiplication execution time.
func BenchmarkMatMulExecution(b *testing.B) {
	sizes := []struct {
		name    string
		m, k, n int
	}{
		{"64x64x64", 64, 64, 64},
		{"128x128x128", 128, 128, 128},
		{"256x256x256", 256, 256, 256},
		{"512x512x512", 512, 512, 512},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			builder := backend.Builder("bench_matmul")
			mainFn := builder.Main()

			// Create input shapes: [M, K] @ [K, N] -> [M, N]
			lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
			rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

			lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for lhs failed: %v", err)
			}

			rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for rhs failed: %v", err)
			}

			// Matrix multiplication using Dot
			result, err := mainFn.Dot(lhs, rhs)
			if err != nil {
				b.Fatalf("Dot() failed: %v", err)
			}

			if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
				b.Fatalf("Return() failed: %v", err)
			}

			// Compile once
			exec, err := builder.Compile()
			if err != nil {
				b.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers
			lhsData := make([]float32, sz.m*sz.k)
			rhsData := make([]float32, sz.k*sz.n)
			for i := range lhsData {
				lhsData[i] = float32(i) * 0.01
			}
			for i := range rhsData {
				rhsData[i] = float32(i) * 0.01
			}

			lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for lhs failed: %v", err)
			}

			rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for rhs failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
				if err != nil {
					b.Fatalf("Execute() failed: %v", err)
				}
				_ = outputs
			}
		})
	}
}

// BenchmarkConvolutionCompilation benchmarks 2D convolution compilation time.
// SKIPPED: ConvGeneral is not implemented in CoreML backend.
func BenchmarkConvolutionCompilation(b *testing.B) {
	b.Skip("ConvGeneral not implemented in CoreML backend")
}

// BenchmarkConvolutionExecution benchmarks 2D convolution execution time.
// SKIPPED: ConvGeneral is not implemented in CoreML backend.
func BenchmarkConvolutionExecution(b *testing.B) {
	b.Skip("ConvGeneral not implemented in CoreML backend")
}

// BenchmarkCNNPattern benchmarks a CNN-like pattern: Conv + ReLU + MaxPool.
// SKIPPED: ConvGeneral and ReduceWindow are not implemented in CoreML backend.
func BenchmarkCNNPattern(b *testing.B) {
	b.Skip("ConvGeneral and ReduceWindow not implemented in CoreML backend")
}

// BenchmarkAttentionPattern benchmarks an attention-like pattern.
// SKIPPED: DotGeneral is not implemented in CoreML backend.
func BenchmarkAttentionPattern(b *testing.B) {
	b.Skip("DotGeneral not implemented in CoreML backend")
}

// BenchmarkBatchedMatMul benchmarks batched matrix multiplication.
// SKIPPED: DotGeneral is not implemented in CoreML backend.
func BenchmarkBatchedMatMul(b *testing.B) {
	b.Skip("DotGeneral not implemented in CoreML backend")
}

// BenchmarkUnaryOps benchmarks unary operations.
func BenchmarkUnaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_unary")
	mainFn := builder.Main()

	// Input: [1024]
	inputShape := shapes.Make(dtypes.Float32, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	// Chain of unary ops: exp(log(abs(x)))
	absX, err := mainFn.Abs(x)
	if err != nil {
		b.Fatalf("Abs() failed: %v", err)
	}

	logX, err := mainFn.Log(absX)
	if err != nil {
		b.Fatalf("Log() failed: %v", err)
	}

	expX, err := mainFn.Exp(logX)
	if err != nil {
		b.Fatalf("Exp() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{expX}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := make([]float32, 1024)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.01
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		_ = outputs
	}
}

// BenchmarkBinaryOps benchmarks binary operations.
func BenchmarkBinaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_binary")
	mainFn := builder.Main()

	// Input: [1024]
	inputShape := shapes.Make(dtypes.Float32, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() for x failed: %v", err)
	}

	y, err := mainFn.Parameter("y", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() for y failed: %v", err)
	}

	// Chain of binary ops: (x + y) * (x - y)
	sum, err := mainFn.Add(x, y)
	if err != nil {
		b.Fatalf("Add() failed: %v", err)
	}

	diff, err := mainFn.Sub(x, y)
	if err != nil {
		b.Fatalf("Sub() failed: %v", err)
	}

	prod, err := mainFn.Mul(sum, diff)
	if err != nil {
		b.Fatalf("Mul() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{prod}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers
	xData := make([]float32, 1024)
	yData := make([]float32, 1024)
	for i := range xData {
		xData[i] = float32(i) * 0.01
		yData[i] = float32(1024-i) * 0.01
	}

	xBuf, err := backend.BufferFromFlatData(0, xData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() for x failed: %v", err)
	}

	yBuf, err := backend.BufferFromFlatData(0, yData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() for y failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		_ = outputs
	}
}

// BenchmarkReduceOps benchmarks reduce operations.
func BenchmarkReduceOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_reduce")
	mainFn := builder.Main()

	// Input: [1024, 1024]
	inputShape := shapes.Make(dtypes.Float32, 1024, 1024)

	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	// ReduceSum along axis 1
	reduced, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		b.Fatalf("ReduceSum() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{reduced}, nil); err != nil {
		b.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := make([]float32, 1024*1024)
	for i := range inputData {
		inputData[i] = float32(i) * 0.0001
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		b.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
		if err != nil {
			b.Fatalf("Execute() failed: %v", err)
		}
		_ = outputs
	}
}
