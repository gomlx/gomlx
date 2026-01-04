//go:build darwin

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestSimpleCNN tests a simple CNN-like pattern:
// Input -> Conv2D -> MaxPool -> output
//
// This verifies that convolution and pooling operations work together end-to-end.
func TestSimpleCNN(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_cnn")

	// Input: [batch=1, channels=1, height=4, width=4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Conv weights: [out_channels=2, in_channels=1, kH=2, kW=2]
	convWeightData := []float32{
		// Filter 0
		1, 0,
		0, 1,
		// Filter 1
		0, 1,
		1, 0,
	}
	convWeights, err := builder.Constant(convWeightData, 2, 1, 2, 2)
	if err != nil {
		t.Fatalf("Constant() for conv weights failed: %v", err)
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

	// Apply convolution: output will be [1, 2, 3, 3]
	convOut, err := builder.ConvGeneral(
		input,
		convWeights,
		axes,
		[]int{1, 1},              // strides
		[][2]int{{0, 0}, {0, 0}}, // no padding
		[]int{1, 1}, []int{1, 1}, // no dilation
		1, 1, // channel group, batch group
	)
	if err != nil {
		t.Fatalf("ConvGeneral() failed: %v", err)
	}

	// Apply ReLU using Max(x, 0)
	// Use true scalar (no dimensions) for broadcasting
	zeroConst, err := builder.Constant([]float32{0})
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}
	reluOut, err := builder.Max(convOut, zeroConst)
	if err != nil {
		t.Fatalf("Max() for ReLU failed: %v", err)
	}

	// Apply MaxPool: [1, 2, 3, 3] -> [1, 2, 1, 1] (3x3 window covers entire spatial)
	poolOut, err := builder.ReduceWindow(
		reluOut,
		backends.ReduceOpMax,
		[]int{1, 1, 3, 3}, // window covers entire 3x3 spatial dims
		[]int{1, 1, 1, 1}, // stride
		nil, nil,          // no dilations
		nil,               // no padding
	)
	if err != nil {
		t.Fatalf("ReduceWindow() for MaxPool failed: %v", err)
	}

	// Reshape for output: [1, 2, 1, 1] -> [1, 2]
	flatOut, err := builder.Reshape(poolOut, 1, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile([]backends.Op{flatOut}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for input failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output shape
	outputShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	expectedShape := shapes.Make(dtypes.Float32, 1, 2)
	if !outputShape.Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShape, expectedShape)
	}

	// Get output data
	outputData := make([]float32, 2)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Output should be non-zero (ReLU preserves positive values)
	for i, v := range outputData {
		if v < 0 {
			t.Errorf("Output[%d] = %f should be >= 0 after ReLU", i, v)
		}
	}

	t.Logf("CNN integration test passed! Output: %v", outputData)
}

// TestBatchedMatMul tests batched matrix multiplication for attention-like patterns.
// Tests: [B, M, K] @ [B, K, N] -> [B, M, N]
func TestBatchedMatMul(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_batched_matmul")

	// Q: [batch=2, seq_len=3, d_model=4]
	qShape := shapes.Make(dtypes.Float32, 2, 3, 4)
	q, err := builder.Parameter("Q", qShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// K: [batch=2, seq_len=3, d_model=4]
	kShape := shapes.Make(dtypes.Float32, 2, 3, 4)
	k, err := builder.Parameter("K", kShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// K^T: [batch=2, d_model=4, seq_len=3]
	kTranspose, err := builder.Transpose(k, 0, 2, 1)
	if err != nil {
		t.Fatalf("Transpose() for K failed: %v", err)
	}

	// Attention scores: Q @ K^T = [2, 3, 4] @ [2, 4, 3] -> [2, 3, 3]
	scores, err := builder.DotGeneral(
		q,
		[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
		kTranspose,
		[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
	)
	if err != nil {
		t.Fatalf("DotGeneral() for attention scores failed: %v", err)
	}

	// Compile and execute
	exec, err := builder.Compile([]backends.Op{scores}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data
	qData := make([]float32, 2*3*4)
	kData := make([]float32, 2*3*4)
	for i := range qData {
		qData[i] = float32(i) * 0.1
		kData[i] = float32(i) * 0.05
	}

	qBuf, err := backend.BufferFromFlatData(0, qData, qShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for Q failed: %v", err)
	}
	kBuf, err := backend.BufferFromFlatData(0, kData, kShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for K failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{qBuf, kBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output shape: [2, 3, 3]
	outputShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	expectedShape := shapes.Make(dtypes.Float32, 2, 3, 3)
	if !outputShape.Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShape, expectedShape)
	}

	// Get output data
	outputData := make([]float32, 2*3*3)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	t.Logf("Batched MatMul test passed! Output shape: %v", outputShape)
}

// TestAttentionBlock tests a simplified self-attention mechanism.
// Q, K, V -> scores -> softmax -> weighted sum
func TestAttentionBlock(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_attention")

	// Input: [batch=1, seq_len=4, d_model=8]
	inputShape := shapes.Make(dtypes.Float32, 1, 4, 8)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// For simplicity, Q = K = V = input (self-attention without projections)
	q := input
	k := input
	v := input

	// K^T: [1, 8, 4]
	kTranspose, err := builder.Transpose(k, 0, 2, 1)
	if err != nil {
		t.Fatalf("Transpose() for K failed: %v", err)
	}

	// Attention scores: Q @ K^T = [1, 4, 8] @ [1, 8, 4] -> [1, 4, 4]
	scores, err := builder.DotGeneral(
		q,
		[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
		kTranspose,
		[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
	)
	if err != nil {
		t.Fatalf("DotGeneral() for attention scores failed: %v", err)
	}

	// Scale by 1/sqrt(d_k) = 1/sqrt(8) ≈ 0.354
	// Use true scalar (no dimensions) for broadcasting
	scaleVal := float32(1.0 / math.Sqrt(8.0))
	scale, err := builder.Constant([]float32{scaleVal})
	if err != nil {
		t.Fatalf("Constant() for scale failed: %v", err)
	}
	scaledScores, err := builder.Mul(scores, scale)
	if err != nil {
		t.Fatalf("Mul() for scaling failed: %v", err)
	}

	// Softmax along last axis: exp(x) / sum(exp(x))
	// Step 1: exp(scores)
	expScores, err := builder.Exp(scaledScores)
	if err != nil {
		t.Fatalf("Exp() failed: %v", err)
	}

	// Step 2: sum along last axis
	sumExp, err := builder.ReduceSum(expScores, 2)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %v", err)
	}

	// Step 3: reshape sum to broadcast: [1, 4] -> [1, 4, 1]
	sumExpBroadcast, err := builder.Reshape(sumExp, 1, 4, 1)
	if err != nil {
		t.Fatalf("Reshape() for broadcast failed: %v", err)
	}

	// Step 4: divide
	softmax, err := builder.Div(expScores, sumExpBroadcast)
	if err != nil {
		t.Fatalf("Div() for softmax failed: %v", err)
	}

	// Weighted sum: softmax @ V = [1, 4, 4] @ [1, 4, 8] -> [1, 4, 8]
	output, err := builder.DotGeneral(
		softmax,
		[]int{2}, []int{0}, // lhs contracting: axis 2, batch: axis 0
		v,
		[]int{1}, []int{0}, // rhs contracting: axis 1, batch: axis 0
	)
	if err != nil {
		t.Fatalf("DotGeneral() for weighted sum failed: %v", err)
	}

	// Compile and execute
	exec, err := builder.Compile([]backends.Op{output}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data
	inputData := make([]float32, 1*4*8)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output shape: same as input [1, 4, 8]
	outputShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	if !outputShape.Equal(inputShape) {
		t.Errorf("Output shape = %v, want %v", outputShape, inputShape)
	}

	// Get output data
	outputData := make([]float32, 1*4*8)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Verify output is valid (no NaN, no Inf)
	for i, v := range outputData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Output[%d] = %f is invalid (NaN or Inf)", i, v)
		}
	}

	t.Logf("Attention block test passed! Output shape: %v", outputShape)
}

// TestReduceWindowMaxPool tests ReduceWindow with MaxPool semantics.
func TestReduceWindowMaxPool(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_window_maxpool")

	// Input: [batch=1, channels=1, height=4, width=4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// MaxPool with 2x2 window, stride 2
	// Output should be [1, 1, 2, 2]
	poolOut, err := builder.ReduceWindow(
		input,
		backends.ReduceOpMax,
		[]int{1, 1, 2, 2}, // window
		[]int{1, 1, 2, 2}, // stride
		nil, nil,          // no dilations
		nil,               // no padding
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Compile and execute
	exec, err := builder.Compile([]backends.Op{poolOut}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data with known pattern
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Verify output shape
	outputShape, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	if !outputShape.Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShape, expectedShape)
	}

	// Get output data
	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: max of each 2x2 window
	// Top-left: max(1,2,5,6) = 6
	// Top-right: max(3,4,7,8) = 8
	// Bottom-left: max(9,10,13,14) = 14
	// Bottom-right: max(11,12,15,16) = 16
	expected := []float32{6, 8, 14, 16}

	for i, v := range expected {
		if outputData[i] != v {
			t.Errorf("Output[%d] = %f, want %f", i, outputData[i], v)
		}
	}

	t.Logf("ReduceWindow MaxPool test passed! Output: %v", outputData)
}

// TestReduceWindowSumPool tests ReduceWindow with Sum (via AvgPool scaling).
func TestReduceWindowSumPool(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_window_sumpool")

	// Input: [batch=1, channels=1, height=4, width=4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	input, err := builder.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// SumPool with 2x2 window, stride 2
	// Output should be [1, 1, 2, 2]
	poolOut, err := builder.ReduceWindow(
		input,
		backends.ReduceOpSum,
		[]int{1, 1, 2, 2}, // window
		[]int{1, 1, 2, 2}, // stride
		nil, nil,          // no dilations
		nil,               // no padding
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Compile and execute
	exec, err := builder.Compile([]backends.Op{poolOut}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data with known pattern
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	inputBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Get output data
	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: sum of each 2x2 window
	// Top-left: 1+2+5+6 = 14
	// Top-right: 3+4+7+8 = 22
	// Bottom-left: 9+10+13+14 = 46
	// Bottom-right: 11+12+15+16 = 54
	expected := []float32{14, 22, 46, 54}

	for i, v := range expected {
		if math.Abs(float64(outputData[i]-v)) > 0.001 {
			t.Errorf("Output[%d] = %f, want %f", i, outputData[i], v)
		}
	}

	t.Logf("ReduceWindow SumPool test passed! Output: %v", outputData)
}
