//go:build darwin

package coreml

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BenchmarkMatMulCompilation benchmarks matrix multiplication compilation time.
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

				// Create input shapes: [M, K] @ [K, N] -> [M, N]
				lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
				rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

				lhs, err := builder.Parameter("lhs", lhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for lhs failed: %v", err)
				}

				rhs, err := builder.Parameter("rhs", rhsShape, nil)
				if err != nil {
					b.Fatalf("Parameter() for rhs failed: %v", err)
				}

				// Matrix multiplication
				result, err := builder.DotGeneral(
					lhs,
					[]int{1}, []int{}, // lhs contracting: axis 1, no batch
					rhs,
					[]int{0}, []int{}, // rhs contracting: axis 0, no batch
				)
				if err != nil {
					b.Fatalf("DotGeneral() failed: %v", err)
				}

				b.StartTimer()
				exec, err := builder.Compile([]backends.Op{result}, nil)
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

			// Create input shapes: [M, K] @ [K, N] -> [M, N]
			lhsShape := shapes.Make(dtypes.Float32, sz.m, sz.k)
			rhsShape := shapes.Make(dtypes.Float32, sz.k, sz.n)

			lhs, err := builder.Parameter("lhs", lhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for lhs failed: %v", err)
			}

			rhs, err := builder.Parameter("rhs", rhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for rhs failed: %v", err)
			}

			// Matrix multiplication
			result, err := builder.DotGeneral(
				lhs,
				[]int{1}, []int{}, // lhs contracting: axis 1, no batch
				rhs,
				[]int{0}, []int{}, // rhs contracting: axis 0, no batch
			)
			if err != nil {
				b.Fatalf("DotGeneral() failed: %v", err)
			}

			// Compile once
			exec, err := builder.Compile([]backends.Op{result}, nil)
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
func BenchmarkConvolutionCompilation(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		backend, err := New("")
		if err != nil {
			b.Fatalf("New() failed: %v", err)
		}

		builder := backend.Builder("bench_conv")

		// Input: [batch, channels, height, width]
		inputShape := shapes.Make(dtypes.Float32, 1, 64, 64, 64)
		input, err := builder.Parameter("input", inputShape, nil)
		if err != nil {
			b.Fatalf("Parameter() failed: %v", err)
		}

		// Kernel: [out_channels, in_channels, kH, kW]
		kernelData := make([]float32, 64*64*3*3)
		for i := range kernelData {
			kernelData[i] = 0.01
		}
		kernel, err := builder.Constant(kernelData, 64, 64, 3, 3)
		if err != nil {
			b.Fatalf("Constant() for kernel failed: %v", err)
		}

		// Configure axes (NCHW format)
		axes := backends.ConvolveAxesConfig{
			InputBatch:           0,
			InputChannels:        1,
			InputSpatial:         []int{2, 3},
			KernelOutputChannels: 0,
			KernelInputChannels:  1,
			KernelSpatial:        []int{2, 3},
			OutputBatch:          0,
			OutputChannels:       1,
			OutputSpatial:        []int{2, 3},
		}

		// Apply convolution
		convOut, err := builder.ConvGeneral(
			input,
			kernel,
			axes,
			[]int{1, 1},              // strides
			[][2]int{{1, 1}, {1, 1}}, // padding 1 on each side
			[]int{1, 1}, []int{1, 1}, // no dilation
			1, 1, // channel group, batch group
		)
		if err != nil {
			b.Fatalf("ConvGeneral() failed: %v", err)
		}

		b.StartTimer()
		exec, err := builder.Compile([]backends.Op{convOut}, nil)
		b.StopTimer()

		if err != nil {
			b.Fatalf("Compile() failed: %v", err)
		}
		exec.Finalize()
		backend.Finalize()
	}
}

// BenchmarkConvolutionExecution benchmarks 2D convolution execution time.
func BenchmarkConvolutionExecution(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_conv")

	// Input: [batch, channels, height, width]
	inputShape := shapes.Make(dtypes.Float32, 1, 64, 64, 64)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	// Kernel: [out_channels, in_channels, kH, kW]
	kernelData := make([]float32, 64*64*3*3)
	for i := range kernelData {
		kernelData[i] = 0.01
	}
	kernel, err := builder.Constant(kernelData, 64, 64, 3, 3)
	if err != nil {
		b.Fatalf("Constant() for kernel failed: %v", err)
	}

	// Configure axes (NCHW format)
	axes := backends.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         []int{2, 3},
		KernelOutputChannels: 0,
		KernelInputChannels:  1,
		KernelSpatial:        []int{2, 3},
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        []int{2, 3},
	}

	// Apply convolution
	convOut, err := builder.ConvGeneral(
		input,
		kernel,
		axes,
		[]int{1, 1},              // strides
		[][2]int{{1, 1}, {1, 1}}, // padding 1 on each side
		[]int{1, 1}, []int{1, 1}, // no dilation
		1, 1, // channel group, batch group
	)
	if err != nil {
		b.Fatalf("ConvGeneral() failed: %v", err)
	}

	// Compile once
	exec, err := builder.Compile([]backends.Op{convOut}, nil)
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := make([]float32, 1*64*64*64)
	for i := range inputData {
		inputData[i] = float32(i) * 0.001
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

// BenchmarkCNNPattern benchmarks a CNN-like pattern: Conv + ReLU + MaxPool.
func BenchmarkCNNPattern(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("bench_cnn_pattern")

	// Input: [batch=1, channels=64, height=32, width=32]
	inputShape := shapes.Make(dtypes.Float32, 1, 64, 32, 32)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		b.Fatalf("Parameter() failed: %v", err)
	}

	// Conv kernel: [out_channels=64, in_channels=64, kH=3, kW=3]
	kernelData := make([]float32, 64*64*3*3)
	for i := range kernelData {
		kernelData[i] = 0.01
	}
	kernel, err := builder.Constant(kernelData, 64, 64, 3, 3)
	if err != nil {
		b.Fatalf("Constant() for kernel failed: %v", err)
	}

	// Configure axes (NCHW format)
	axes := backends.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         []int{2, 3},
		KernelOutputChannels: 0,
		KernelInputChannels:  1,
		KernelSpatial:        []int{2, 3},
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        []int{2, 3},
	}

	// Conv2D with padding to maintain size
	convOut, err := builder.ConvGeneral(
		input,
		kernel,
		axes,
		[]int{1, 1},              // strides
		[][2]int{{1, 1}, {1, 1}}, // padding 1 on each side
		[]int{1, 1}, []int{1, 1}, // no dilation
		1, 1, // channel group, batch group
	)
	if err != nil {
		b.Fatalf("ConvGeneral() failed: %v", err)
	}

	// ReLU: Max(x, 0)
	zeroConst, err := builder.Constant([]float32{0})
	if err != nil {
		b.Fatalf("Constant() for zero failed: %v", err)
	}
	reluOut, err := builder.Max(convOut, zeroConst)
	if err != nil {
		b.Fatalf("Max() for ReLU failed: %v", err)
	}

	// MaxPool: 2x2 window, stride 2 -> [1, 64, 16, 16]
	poolOut, err := builder.ReduceWindow(
		reluOut,
		backends.ReduceOpMax,
		[]int{1, 1, 2, 2}, // window
		[]int{1, 1, 2, 2}, // stride
		nil, nil,          // no dilations
		nil,               // no padding
	)
	if err != nil {
		b.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile([]backends.Op{poolOut}, nil)
	if err != nil {
		b.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := make([]float32, 1*64*32*32)
	for i := range inputData {
		inputData[i] = float32(i) * 0.001
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

// BenchmarkAttentionPattern benchmarks an attention-like pattern.
func BenchmarkAttentionPattern(b *testing.B) {
	configs := []struct {
		name              string
		batch, seqLen, dModel int
	}{
		{"1x16x64", 1, 16, 64},
		{"1x32x128", 1, 32, 128},
		{"4x32x128", 4, 32, 128},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			builder := backend.Builder("bench_attention")

			// Q, K, V: [batch, seq_len, d_model]
			qkvShape := shapes.Make(dtypes.Float32, cfg.batch, cfg.seqLen, cfg.dModel)

			q, err := builder.Parameter("Q", qkvShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for Q failed: %v", err)
			}

			k, err := builder.Parameter("K", qkvShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for K failed: %v", err)
			}

			v, err := builder.Parameter("V", qkvShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for V failed: %v", err)
			}

			// K^T: [batch, d_model, seq_len]
			kTranspose, err := builder.Transpose(k, 0, 2, 1)
			if err != nil {
				b.Fatalf("Transpose() failed: %v", err)
			}

			// Attention scores: Q @ K^T = [batch, seq_len, seq_len]
			scores, err := builder.DotGeneral(
				q,
				[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
				kTranspose,
				[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
			)
			if err != nil {
				b.Fatalf("DotGeneral() for scores failed: %v", err)
			}

			// Softmax: exp(x) / sum(exp(x))
			expScores, err := builder.Exp(scores)
			if err != nil {
				b.Fatalf("Exp() failed: %v", err)
			}

			sumExp, err := builder.ReduceSum(expScores, 2)
			if err != nil {
				b.Fatalf("ReduceSum() failed: %v", err)
			}

			// Reshape for broadcasting: [batch, seq_len] -> [batch, seq_len, 1]
			sumExpBroadcast, err := builder.Reshape(sumExp, cfg.batch, cfg.seqLen, 1)
			if err != nil {
				b.Fatalf("Reshape() failed: %v", err)
			}

			softmax, err := builder.Div(expScores, sumExpBroadcast)
			if err != nil {
				b.Fatalf("Div() for softmax failed: %v", err)
			}

			// Weighted sum: softmax @ V = [batch, seq_len, d_model]
			output, err := builder.DotGeneral(
				softmax,
				[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
				v,
				[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
			)
			if err != nil {
				b.Fatalf("DotGeneral() for weighted sum failed: %v", err)
			}

			// Compile
			exec, err := builder.Compile([]backends.Op{output}, nil)
			if err != nil {
				b.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers
			dataSize := cfg.batch * cfg.seqLen * cfg.dModel
			qData := make([]float32, dataSize)
			kData := make([]float32, dataSize)
			vData := make([]float32, dataSize)

			for i := range qData {
				qData[i] = float32(i) * 0.01
				kData[i] = float32(i) * 0.01
				vData[i] = float32(i) * 0.01
			}

			qBuf, err := backend.BufferFromFlatData(0, qData, qkvShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for Q failed: %v", err)
			}

			kBuf, err := backend.BufferFromFlatData(0, kData, qkvShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for K failed: %v", err)
			}

			vBuf, err := backend.BufferFromFlatData(0, vData, qkvShape)
			if err != nil {
				b.Fatalf("BufferFromFlatData() for V failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				outputs, err := exec.Execute([]backends.Buffer{qBuf, kBuf, vBuf}, nil, 0)
				if err != nil {
					b.Fatalf("Execute() failed: %v", err)
				}
				_ = outputs
			}
		})
	}
}

// BenchmarkBatchedMatMul benchmarks batched matrix multiplication.
func BenchmarkBatchedMatMul(b *testing.B) {
	configs := []struct {
		name           string
		batch, m, k, n int
	}{
		{"1x64x64x64", 1, 64, 64, 64},
		{"4x64x64x64", 4, 64, 64, 64},
		{"8x128x128x128", 8, 128, 128, 128},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			backend, err := New("")
			if err != nil {
				b.Fatalf("New() failed: %v", err)
			}
			defer backend.Finalize()

			builder := backend.Builder("bench_batched_matmul")

			// LHS: [batch, M, K]
			lhsShape := shapes.Make(dtypes.Float32, cfg.batch, cfg.m, cfg.k)
			lhs, err := builder.Parameter("lhs", lhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for lhs failed: %v", err)
			}

			// RHS: [batch, K, N]
			rhsShape := shapes.Make(dtypes.Float32, cfg.batch, cfg.k, cfg.n)
			rhs, err := builder.Parameter("rhs", rhsShape, nil)
			if err != nil {
				b.Fatalf("Parameter() for rhs failed: %v", err)
			}

			// Batched matmul: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
			result, err := builder.DotGeneral(
				lhs,
				[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
				rhs,
				[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
			)
			if err != nil {
				b.Fatalf("DotGeneral() failed: %v", err)
			}

			// Compile
			exec, err := builder.Compile([]backends.Op{result}, nil)
			if err != nil {
				b.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			// Create input buffers
			lhsData := make([]float32, cfg.batch*cfg.m*cfg.k)
			rhsData := make([]float32, cfg.batch*cfg.k*cfg.n)

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
