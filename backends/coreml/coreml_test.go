//go:build darwin && cgo

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestBackendCreation tests that the backend can be created.
func TestBackendCreation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	if backend.Name() != "CoreML (coreml)" {
		t.Errorf("Name() = %q, want %q", backend.Name(), "CoreML (coreml)")
	}

	if backend.String() != "coreml" {
		t.Errorf("String() = %q, want %q", backend.String(), "coreml")
	}

	if backend.NumDevices() != 1 {
		t.Errorf("NumDevices() = %d, want 1", backend.NumDevices())
	}
}

// TestBufferOperations tests basic buffer creation and data transfer.
func TestBufferOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	// Test creating a buffer
	shape := shapes.Make(dtypes.Float32, 2, 3)
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	buffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Test getting shape
	gotShape, err := backend.BufferShape(buffer)
	if err != nil {
		t.Fatalf("BufferShape() failed: %v", err)
	}
	if !gotShape.Equal(shape) {
		t.Errorf("BufferShape() = %v, want %v", gotShape, shape)
	}

	// Test getting device number
	deviceNum, err := backend.BufferDeviceNum(buffer)
	if err != nil {
		t.Fatalf("BufferDeviceNum() failed: %v", err)
	}
	if deviceNum != 0 {
		t.Errorf("BufferDeviceNum() = %d, want 0", deviceNum)
	}

	// Test reading data back
	outputData := make([]float32, 6)
	err = backend.BufferToFlatData(buffer, outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}

	// Test finalize
	err = backend.BufferFinalize(buffer)
	if err != nil {
		t.Fatalf("BufferFinalize() failed: %v", err)
	}
}

// TestSharedBuffer tests shared buffer functionality.
func TestSharedBuffer(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	if !backend.HasSharedBuffers() {
		t.Fatal("HasSharedBuffers() = false, want true")
	}

	shape := shapes.Make(dtypes.Float32, 4)
	buffer, flat, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer() failed: %v", err)
	}

	// Modify data directly
	flatData := flat.([]float32)
	for i := range flatData {
		flatData[i] = float32(i + 1)
	}

	// Verify through BufferData
	gotFlat, err := backend.BufferData(buffer)
	if err != nil {
		t.Fatalf("BufferData() failed: %v", err)
	}

	gotData := gotFlat.([]float32)
	for i := range gotData {
		if gotData[i] != float32(i+1) {
			t.Errorf("gotData[%d] = %f, want %f", i, gotData[i], float32(i+1))
		}
	}

	_ = backend.BufferFinalize(buffer)
}

// TestBuilderParameterAndConstant tests creating parameters and constants via the Function interface.
func TestBuilderParameterAndConstant(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test")
	mainFn := builder.Main()

	// Create a parameter
	shape := shapes.Make(dtypes.Float32, 2, 3)
	param, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	paramShape, err := builder.OpShape(param)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	if !paramShape.Equal(shape) {
		t.Errorf("OpShape(param) = %v, want %v", paramShape, shape)
	}

	// Create a constant
	constData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	constant, err := mainFn.Constant(constData, 2, 3)
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}

	constShape, err := builder.OpShape(constant)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	if !constShape.Equal(shape) {
		t.Errorf("OpShape(constant) = %v, want %v", constShape, shape)
	}
}

// TestAddOperation tests the Add operation through compilation and execution.
func TestAddOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_add")
	mainFn := builder.Main()

	// Create two parameters
	shape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	y, err := mainFn.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Add them
	z, err := mainFn.Add(x, y)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Mark outputs
	err = mainFn.Return([]backends.Value{z}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify inputs/outputs
	names, inputShapes := exec.Inputs()
	if len(names) != 2 {
		t.Errorf("Inputs() returned %d names, want 2", len(names))
	}
	if len(inputShapes) != 2 {
		t.Errorf("Inputs() returned %d shapes, want 2", len(inputShapes))
	}

	outputShapes := exec.Outputs()
	if len(outputShapes) != 1 {
		t.Errorf("Outputs() returned %d shapes, want 1", len(outputShapes))
	}

	// Create input buffers
	xData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	yData := []float32{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}

	xBuf, err := backend.BufferFromFlatData(0, xData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %v", err)
	}

	yBuf, err := backend.BufferFromFlatData(0, yData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(y) failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	if len(outputs) != 1 {
		t.Fatalf("Execute() returned %d outputs, want 1", len(outputs))
	}

	// Verify output
	zData := make([]float32, 6)
	err = backend.BufferToFlatData(outputs[0], zData)
	if err != nil {
		t.Fatalf("BufferToFlatData(z) failed: %v", err)
	}

	expected := []float32{11.0, 22.0, 33.0, 44.0, 55.0, 66.0}
	for i := range expected {
		if math.Abs(float64(zData[i]-expected[i])) > 1e-5 {
			t.Errorf("zData[%d] = %f, want %f", i, zData[i], expected[i])
		}
	}
}

// TestUnaryOperations tests unary operations like Abs, Neg, Exp, etc.
func TestUnaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name     string
		opFunc   func(backends.Function, backends.Value) (backends.Value, error)
		input    []float32
		expected []float32
	}{
		{
			name:     "Abs",
			opFunc:   func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Abs(x) },
			input:    []float32{-1.0, 2.0, -3.0, 4.0},
			expected: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:     "Neg",
			opFunc:   func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Neg(x) },
			input:    []float32{-1.0, 2.0, -3.0, 4.0},
			expected: []float32{1.0, -2.0, 3.0, -4.0},
		},
		{
			name:     "Exp",
			opFunc:   func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Exp(x) },
			input:    []float32{0.0, 1.0, 2.0, -1.0},
			expected: []float32{1.0, float32(math.E), float32(math.E * math.E), float32(1.0 / math.E)},
		},
		{
			name:     "Sqrt",
			opFunc:   func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Sqrt(x) },
			input:    []float32{1.0, 4.0, 9.0, 16.0},
			expected: []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)
			mainFn := builder.Main()

			shape := shapes.Make(dtypes.Float32, len(tc.input))
			x, err := mainFn.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := tc.opFunc(mainFn, x)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			err = mainFn.Return([]backends.Value{y}, nil)
			if err != nil {
				t.Fatalf("Return() failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			xBuf, err := backend.BufferFromFlatData(0, tc.input, shape)
			if err != nil {
				t.Fatalf("BufferFromFlatData() failed: %v", err)
			}

			outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %v", err)
			}

			yData := make([]float32, len(tc.expected))
			err = backend.BufferToFlatData(outputs[0], yData)
			if err != nil {
				t.Fatalf("BufferToFlatData() failed: %v", err)
			}

			for i := range tc.expected {
				if math.Abs(float64(yData[i]-tc.expected[i])) > 1e-4 {
					t.Errorf("%s: yData[%d] = %f, want %f", tc.name, i, yData[i], tc.expected[i])
				}
			}
		})
	}
}

// TestBinaryOperations tests binary operations like Add, Sub, Mul, Div.
func TestBinaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name     string
		opFunc   func(backends.Function, backends.Value, backends.Value) (backends.Value, error)
		a        []float32
		b        []float32
		expected []float32
	}{
		{
			name:     "Add",
			opFunc:   func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Add(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{10.0, 20.0, 30.0, 40.0},
			expected: []float32{11.0, 22.0, 33.0, 44.0},
		},
		{
			name:     "Sub",
			opFunc:   func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Sub(x, y) },
			a:        []float32{10.0, 20.0, 30.0, 40.0},
			b:        []float32{1.0, 2.0, 3.0, 4.0},
			expected: []float32{9.0, 18.0, 27.0, 36.0},
		},
		{
			name:     "Mul",
			opFunc:   func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Mul(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{10.0, 10.0, 10.0, 10.0},
			expected: []float32{10.0, 20.0, 30.0, 40.0},
		},
		{
			name:     "Div",
			opFunc:   func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Div(x, y) },
			a:        []float32{10.0, 20.0, 30.0, 40.0},
			b:        []float32{2.0, 4.0, 5.0, 8.0},
			expected: []float32{5.0, 5.0, 6.0, 5.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)
			mainFn := builder.Main()

			shape := shapes.Make(dtypes.Float32, len(tc.a))
			x, err := mainFn.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := mainFn.Parameter("y", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			z, err := tc.opFunc(mainFn, x, y)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			err = mainFn.Return([]backends.Value{z}, nil)
			if err != nil {
				t.Fatalf("Return() failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile() failed: %v", err)
			}
			defer exec.Finalize()

			xBuf, err := backend.BufferFromFlatData(0, tc.a, shape)
			if err != nil {
				t.Fatalf("BufferFromFlatData(x) failed: %v", err)
			}

			yBuf, err := backend.BufferFromFlatData(0, tc.b, shape)
			if err != nil {
				t.Fatalf("BufferFromFlatData(y) failed: %v", err)
			}

			outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %v", err)
			}

			zData := make([]float32, len(tc.expected))
			err = backend.BufferToFlatData(outputs[0], zData)
			if err != nil {
				t.Fatalf("BufferToFlatData() failed: %v", err)
			}

			for i := range tc.expected {
				if math.Abs(float64(zData[i]-tc.expected[i])) > 1e-5 {
					t.Errorf("%s: zData[%d] = %f, want %f", tc.name, i, zData[i], tc.expected[i])
				}
			}
		})
	}
}

// TestReduceSum tests the ReduceSum operation.
func TestReduceSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_sum")
	mainFn := builder.Main()

	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reduce sum along axis 1
	y, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 2, 3], [4, 5, 6]]
	// Sum along axis 1: [6, 15]
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 2)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{6.0, 15.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestSlice tests the Slice operation.
func TestSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_slice")
	mainFn := builder.Main()

	inputShape := shapes.Make(dtypes.Float32, 4, 4)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Slice from [1,1] with size [2,2]
	y, err := mainFn.Slice(x, []int{1, 1}, []int{3, 3}, []int{1, 1})
	if err != nil {
		t.Fatalf("Slice() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: 4x4 matrix with values 0-15
	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Slicing [1:3, 1:3] from a 4x4 matrix:
	// [[ 0,  1,  2,  3],
	//  [ 4,  5,  6,  7],
	//  [ 8,  9, 10, 11],
	//  [12, 13, 14, 15]]
	// Result: [[5, 6], [9, 10]]
	expected := []float32{5.0, 6.0, 9.0, 10.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestChainedOperations tests chaining multiple operations.
func TestChainedOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_chained")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Compute: (x + 1) * 2
	one := []float32{1.0, 1.0, 1.0, 1.0}
	oneConst, err := mainFn.Constant(one, 4)
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}

	two := []float32{2.0, 2.0, 2.0, 2.0}
	twoConst, err := mainFn.Constant(two, 4)
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}

	xPlusOne, err := mainFn.Add(x, oneConst)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	result, err := mainFn.Mul(xPlusOne, twoConst)
	if err != nil {
		t.Fatalf("Mul() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{result}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: (x + 1) * 2 = [4, 6, 8, 10]
	expected := []float32{4.0, 6.0, 8.0, 10.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestComparisonOperations tests comparison operations.
func TestComparisonOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("Equal", func(t *testing.T) {
		builder := backend.Builder("test_equal")
		mainFn := builder.Main()

		shape := shapes.Make(dtypes.Float32, 4)
		x, _ := mainFn.Parameter("x", shape, nil)
		y, _ := mainFn.Parameter("y", shape, nil)

		// Create comparison
		cond, err := mainFn.Equal(x, y)
		if err != nil {
			t.Fatalf("Equal() failed: %v", err)
		}

		// Convert bool to float using Where: Where(cond, 1.0, 0.0)
		trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
		falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)

		result, err := mainFn.Where(cond, trueConst, falseConst)
		if err != nil {
			t.Fatalf("Where() failed: %v", err)
		}

		err = mainFn.Return([]backends.Value{result}, nil)
		if err != nil {
			t.Fatalf("Return() failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %v", err)
		}
		defer exec.Finalize()

		// Create input buffers
		xData := []float32{1.0, 2.0, 3.0, 4.0}
		yData := []float32{1.0, 2.0, 5.0, 6.0} // Equal at indices 0, 1; not equal at 2, 3

		xBuffer, err := backend.BufferFromFlatData(0, xData, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData() x failed: %v", err)
		}
		yBuffer, err := backend.BufferFromFlatData(0, yData, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData() y failed: %v", err)
		}

		// Execute
		outputs, err := exec.Execute([]backends.Buffer{xBuffer, yBuffer}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %v", err)
		}

		// Read output
		outputData := make([]float32, 4)
		err = backend.BufferToFlatData(outputs[0], outputData)
		if err != nil {
			t.Fatalf("BufferToFlatData() failed: %v", err)
		}

		// Expected: [1.0, 1.0, 0.0, 0.0] (equal at 0,1; not equal at 2,3)
		expected := []float32{1.0, 1.0, 0.0, 0.0}
		for i := range expected {
			if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
				t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
			}
		}
	})
}

// TestReshape tests the Reshape operation.
func TestReshape(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape")
	mainFn := builder.Main()

	// Input: [2, 3] matrix
	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reshape to [3, 2]
	y, err := mainFn.Reshape(x, 3, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify output shape
	outputShapes := exec.Outputs()
	if len(outputShapes) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputShapes))
	}
	expectedShape := shapes.Make(dtypes.Float32, 3, 2)
	if !outputShapes[0].Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShapes[0], expectedShape)
	}

	// Input data: [[1, 2, 3], [4, 5, 6]]
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 6)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Reshape should preserve data order: [[1, 2], [3, 4], [5, 6]]
	expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestReshapeFlatten tests reshaping to flatten a tensor.
func TestReshapeFlatten(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape_flatten")
	mainFn := builder.Main()

	// Input: [2, 3] matrix
	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Flatten to [6]
	y, err := mainFn.Reshape(x, 6)
	if err != nil {
		t.Fatalf("Reshape() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify output shape
	outputShapes := exec.Outputs()
	expectedShape := shapes.Make(dtypes.Float32, 6)
	if !outputShapes[0].Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShapes[0], expectedShape)
	}

	// Execute and verify
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 6)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range inputData {
		if math.Abs(float64(outputData[i]-inputData[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}
}

// TestTranspose tests the Transpose operation.
func TestTranspose(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transpose")
	mainFn := builder.Main()

	// Input: [2, 3] matrix
	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Transpose: [2, 3] -> [3, 2] using permutation [1, 0]
	y, err := mainFn.Transpose(x, 1, 0)
	if err != nil {
		t.Fatalf("Transpose() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify output shape
	outputShapes := exec.Outputs()
	if len(outputShapes) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputShapes))
	}
	expectedShape := shapes.Make(dtypes.Float32, 3, 2)
	if !outputShapes[0].Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShapes[0], expectedShape)
	}

	// Input data: [[1, 2, 3], [4, 5, 6]] (row-major)
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 6)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Transposed: [[1, 4], [2, 5], [3, 6]] (row-major)
	expected := []float32{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestTranspose3D tests the Transpose operation on a 3D tensor.
func TestTranspose3D(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transpose_3d")
	mainFn := builder.Main()

	// Input: [2, 3, 4] tensor
	inputShape := shapes.Make(dtypes.Float32, 2, 3, 4)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Transpose with permutation [2, 0, 1]: [2, 3, 4] -> [4, 2, 3]
	y, err := mainFn.Transpose(x, 2, 0, 1)
	if err != nil {
		t.Fatalf("Transpose() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify output shape
	outputShapes := exec.Outputs()
	expectedShape := shapes.Make(dtypes.Float32, 4, 2, 3)
	if !outputShapes[0].Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShapes[0], expectedShape)
	}
}

// TestFunctionInterface tests the Function interface basics.
func TestFunctionInterface(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_function")
	mainFn := builder.Main()

	// Test Name()
	if mainFn.Name() != backends.MainName {
		t.Errorf("mainFn.Name() = %q, want %q", mainFn.Name(), backends.MainName)
	}

	// Test Parent() - main function should have no parent
	if mainFn.Parent() != nil {
		t.Errorf("mainFn.Parent() should be nil for main function")
	}

	// Test Closure()
	closure, err := mainFn.Closure()
	if err != nil {
		t.Fatalf("Closure() failed: %v", err)
	}

	// Closure should have empty name
	if closure.Name() != "" {
		t.Errorf("closure.Name() = %q, want empty string", closure.Name())
	}

	// Closure should have mainFn as parent
	if closure.Parent() != mainFn {
		t.Errorf("closure.Parent() should be mainFn")
	}
}

// TestConvertDType tests the ConvertDType operation.
func TestConvertDType(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_convert_dtype")
	mainFn := builder.Main()

	// Input: Float32 tensor
	inputShape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Convert to Int32
	y, err := mainFn.ConvertDType(x, dtypes.Int32)
	if err != nil {
		t.Fatalf("ConvertDType() failed: %v", err)
	}

	// Convert back to Float32 for output (since we need to read float data)
	z, err := mainFn.ConvertDType(y, dtypes.Float32)
	if err != nil {
		t.Fatalf("ConvertDType() back failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{z}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Verify output shape
	outputShapes := exec.Outputs()
	expectedShape := shapes.Make(dtypes.Float32, 4)
	if !outputShapes[0].Equal(expectedShape) {
		t.Errorf("Output shape = %v, want %v", outputShapes[0], expectedShape)
	}

	// Input data with fractional values
	inputData := []float32{1.7, 2.3, 3.9, 4.1}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// After Float32 -> Int32 -> Float32, fractional parts should be truncated
	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestWhereScalarCondition tests Where with scalar condition broadcasting.
func TestWhereScalarCondition(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_where_scalar")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 4)
	x, _ := mainFn.Parameter("x", shape, nil)
	y, _ := mainFn.Parameter("y", shape, nil)

	// Create a comparison that yields non-scalar bool
	cond, err := mainFn.GreaterThan(x, y)
	if err != nil {
		t.Fatalf("GreaterThan() failed: %v", err)
	}

	// Where should select from x where x > y, else from y
	result, err := mainFn.Where(cond, x, y)
	if err != nil {
		t.Fatalf("Where() failed: %v", err)
	}

	err = mainFn.Return([]backends.Value{result}, nil)
	if err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// x = [5, 2, 7, 1]
	// y = [3, 4, 6, 2]
	// x > y at indices 0, 2; y >= x at indices 1, 3
	// Expected: [5, 4, 7, 2]
	xData := []float32{5.0, 2.0, 7.0, 1.0}
	yData := []float32{3.0, 4.0, 6.0, 2.0}

	xBuffer, err := backend.BufferFromFlatData(0, xData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() x failed: %v", err)
	}
	yBuffer, err := backend.BufferFromFlatData(0, yData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() y failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuffer, yBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{5.0, 4.0, 7.0, 2.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}
