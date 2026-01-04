//go:build darwin

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

// TestBuilderParameterAndConstant tests creating parameters and constants.
func TestBuilderParameterAndConstant(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test")

	// Create a parameter
	shape := shapes.Make(dtypes.Float32, 2, 3)
	param, err := builder.Parameter("x", shape, nil)
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
	constant, err := builder.Constant(constData, 2, 3)
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

	// Create two parameters
	shape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	y, err := builder.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Add them
	z, err := builder.Add(x, y)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile([]backends.Op{z}, nil)
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
		opFunc   func(backends.Builder, backends.Op) (backends.Op, error)
		input    []float32
		expected []float32
	}{
		{
			name:     "Abs",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Abs(x) },
			input:    []float32{-1.0, 2.0, -3.0, 4.0},
			expected: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:     "Neg",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Neg(x) },
			input:    []float32{-1.0, 2.0, -3.0, 4.0},
			expected: []float32{1.0, -2.0, 3.0, -4.0},
		},
		{
			name:     "Exp",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Exp(x) },
			input:    []float32{0.0, 1.0, 2.0, -1.0},
			expected: []float32{1.0, float32(math.E), float32(math.E * math.E), float32(1.0 / math.E)},
		},
		{
			name:     "Sqrt",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Sqrt(x) },
			input:    []float32{1.0, 4.0, 9.0, 16.0},
			expected: []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			shape := shapes.Make(dtypes.Float32, len(tc.input))
			x, err := builder.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := tc.opFunc(builder, x)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			exec, err := builder.Compile([]backends.Op{y}, nil)
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
		opFunc   func(backends.Builder, backends.Op, backends.Op) (backends.Op, error)
		a        []float32
		b        []float32
		expected []float32
	}{
		{
			name:     "Add",
			opFunc:   func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.Add(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{10.0, 20.0, 30.0, 40.0},
			expected: []float32{11.0, 22.0, 33.0, 44.0},
		},
		{
			name:     "Sub",
			opFunc:   func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.Sub(x, y) },
			a:        []float32{10.0, 20.0, 30.0, 40.0},
			b:        []float32{1.0, 2.0, 3.0, 4.0},
			expected: []float32{9.0, 18.0, 27.0, 36.0},
		},
		{
			name:     "Mul",
			opFunc:   func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.Mul(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{10.0, 10.0, 10.0, 10.0},
			expected: []float32{10.0, 20.0, 30.0, 40.0},
		},
		{
			name:     "Div",
			opFunc:   func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.Div(x, y) },
			a:        []float32{10.0, 20.0, 30.0, 40.0},
			b:        []float32{2.0, 4.0, 5.0, 8.0},
			expected: []float32{5.0, 5.0, 6.0, 5.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			shape := shapes.Make(dtypes.Float32, len(tc.a))
			x, err := builder.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := builder.Parameter("y", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			z, err := tc.opFunc(builder, x, y)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			exec, err := builder.Compile([]backends.Op{z}, nil)
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

// TestReshape tests the Reshape operation.
func TestReshape(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape")

	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reshape from [2, 3] to [3, 2]
	y, err := builder.Reshape(x, 3, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %v", err)
	}

	yShape, err := builder.OpShape(y)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}

	expectedShape := shapes.Make(dtypes.Float32, 3, 2)
	if !yShape.Equal(expectedShape) {
		t.Errorf("OpShape(y) = %v, want %v", yShape, expectedShape)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

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

	// Data should be the same, just reshaped
	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
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

	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reduce sum along axis 1
	y, err := builder.ReduceSum(x, 1)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
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

// TestComparisonOperationsViaWhere tests comparison operations indirectly via Where.
// Note: CoreML doesn't allow Bool outputs directly from models, so we test comparison
// ops by using them with Where, which converts the result back to float.
func TestComparisonOperationsViaWhere(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name      string
		cmpFunc   func(backends.Builder, backends.Op, backends.Op) (backends.Op, error)
		a         []float32
		b         []float32
		trueVal   float32
		falseVal  float32
		expected  []float32
	}{
		{
			name:     "Equal",
			cmpFunc:  func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.Equal(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{1.0, 3.0, 3.0, 5.0},
			trueVal:  1.0,
			falseVal: 0.0,
			expected: []float32{1.0, 0.0, 1.0, 0.0}, // true, false, true, false
		},
		{
			name:     "LessThan",
			cmpFunc:  func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.LessThan(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{2.0, 2.0, 2.0, 2.0},
			trueVal:  1.0,
			falseVal: 0.0,
			expected: []float32{1.0, 0.0, 0.0, 0.0}, // true, false, false, false
		},
		{
			name:     "GreaterThan",
			cmpFunc:  func(bld backends.Builder, x, y backends.Op) (backends.Op, error) { return bld.GreaterThan(x, y) },
			a:        []float32{1.0, 2.0, 3.0, 4.0},
			b:        []float32{2.0, 2.0, 2.0, 2.0},
			trueVal:  1.0,
			falseVal: 0.0,
			expected: []float32{0.0, 0.0, 1.0, 1.0}, // false, false, true, true
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			shape := shapes.Make(dtypes.Float32, len(tc.a))
			x, err := builder.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := builder.Parameter("y", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			// Create comparison
			cond, err := tc.cmpFunc(builder, x, y)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			// Create constants for true/false values
			trueConst, _ := builder.Constant([]float32{tc.trueVal, tc.trueVal, tc.trueVal, tc.trueVal}, len(tc.a))
			falseConst, _ := builder.Constant([]float32{tc.falseVal, tc.falseVal, tc.falseVal, tc.falseVal}, len(tc.a))

			// Use Where to convert bool to float
			z, err := builder.Where(cond, trueConst, falseConst)
			if err != nil {
				t.Fatalf("Where() failed: %v", err)
			}

			exec, err := builder.Compile([]backends.Op{z}, nil)
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
					t.Errorf("%s: zData[%d] = %v, want %v", tc.name, i, zData[i], tc.expected[i])
				}
			}
		})
	}
}

// TestWhereOperation tests the Where/Select operation.
func TestWhereOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_where")

	shape := shapes.Make(dtypes.Float32, 4)
	boolShape := shapes.Make(dtypes.Bool, 4)

	x, err := builder.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(x) failed: %v", err)
	}

	y, err := builder.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(y) failed: %v", err)
	}

	// Create condition: x > y
	cond, err := builder.GreaterThan(x, y)
	if err != nil {
		t.Fatalf("GreaterThan() failed: %v", err)
	}

	// Where(cond, x, y) - returns x where cond is true, y otherwise
	result, err := builder.Where(cond, x, y)
	if err != nil {
		t.Fatalf("Where() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{result}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	xData := []float32{1.0, 5.0, 3.0, 8.0}
	yData := []float32{2.0, 4.0, 3.0, 6.0}

	xBuf, err := backend.BufferFromFlatData(0, xData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %v", err)
	}

	yBuf, err := backend.BufferFromFlatData(0, yData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(y) failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 4)
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// x > y: [false, true, false, true]
	// Expected: [y[0], x[1], y[2], x[3]] = [2.0, 5.0, 3.0, 8.0]
	expected := []float32{2.0, 5.0, 3.0, 8.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}

	_ = boolShape // silence unused variable warning
}

// TestMathOperations tests additional math operations.
func TestMathOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("Pow", func(t *testing.T) {
		builder := backend.Builder("test_pow")

		shape := shapes.Make(dtypes.Float32, 4)
		x, _ := builder.Parameter("x", shape, nil)
		y, _ := builder.Parameter("y", shape, nil)

		z, err := builder.Pow(x, y)
		if err != nil {
			t.Fatalf("Pow() failed: %v", err)
		}

		exec, _ := builder.Compile([]backends.Op{z}, nil)
		defer exec.Finalize()

		xData := []float32{2.0, 3.0, 4.0, 2.0}
		yData := []float32{2.0, 2.0, 0.5, 3.0}

		xBuf, _ := backend.BufferFromFlatData(0, xData, shape)
		yBuf, _ := backend.BufferFromFlatData(0, yData, shape)

		outputs, _ := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)

		outputData := make([]float32, 4)
		backend.BufferToFlatData(outputs[0], outputData)

		expected := []float32{4.0, 9.0, 2.0, 8.0}
		for i := range expected {
			if math.Abs(float64(outputData[i]-expected[i])) > 1e-4 {
				t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
			}
		}
	})

	t.Run("Max", func(t *testing.T) {
		builder := backend.Builder("test_max")

		shape := shapes.Make(dtypes.Float32, 4)
		x, _ := builder.Parameter("x", shape, nil)
		y, _ := builder.Parameter("y", shape, nil)

		z, err := builder.Max(x, y)
		if err != nil {
			t.Fatalf("Max() failed: %v", err)
		}

		exec, _ := builder.Compile([]backends.Op{z}, nil)
		defer exec.Finalize()

		xData := []float32{1.0, 5.0, 3.0, 7.0}
		yData := []float32{2.0, 4.0, 6.0, 8.0}

		xBuf, _ := backend.BufferFromFlatData(0, xData, shape)
		yBuf, _ := backend.BufferFromFlatData(0, yData, shape)

		outputs, _ := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)

		outputData := make([]float32, 4)
		backend.BufferToFlatData(outputs[0], outputData)

		expected := []float32{2.0, 5.0, 6.0, 8.0}
		for i := range expected {
			if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
				t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
			}
		}
	})

	t.Run("Floor", func(t *testing.T) {
		builder := backend.Builder("test_floor")

		shape := shapes.Make(dtypes.Float32, 4)
		x, _ := builder.Parameter("x", shape, nil)

		y, err := builder.Floor(x)
		if err != nil {
			t.Fatalf("Floor() failed: %v", err)
		}

		exec, _ := builder.Compile([]backends.Op{y}, nil)
		defer exec.Finalize()

		xData := []float32{1.5, 2.9, -1.5, -2.9}
		xBuf, _ := backend.BufferFromFlatData(0, xData, shape)

		outputs, _ := exec.Execute([]backends.Buffer{xBuf}, nil, 0)

		outputData := make([]float32, 4)
		backend.BufferToFlatData(outputs[0], outputData)

		expected := []float32{1.0, 2.0, -2.0, -3.0}
		for i := range expected {
			if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
				t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
			}
		}
	})

	t.Run("Ceil", func(t *testing.T) {
		builder := backend.Builder("test_ceil")

		shape := shapes.Make(dtypes.Float32, 4)
		x, _ := builder.Parameter("x", shape, nil)

		y, err := builder.Ceil(x)
		if err != nil {
			t.Fatalf("Ceil() failed: %v", err)
		}

		exec, _ := builder.Compile([]backends.Op{y}, nil)
		defer exec.Finalize()

		xData := []float32{1.5, 2.1, -1.5, -2.1}
		xBuf, _ := backend.BufferFromFlatData(0, xData, shape)

		outputs, _ := exec.Execute([]backends.Buffer{xBuf}, nil, 0)

		outputData := make([]float32, 4)
		backend.BufferToFlatData(outputs[0], outputData)

		expected := []float32{2.0, 3.0, -1.0, -2.0}
		for i := range expected {
			if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
				t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
			}
		}
	})
}

// TestTrigOperations tests trigonometric operations.
func TestTrigOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name     string
		opFunc   func(backends.Builder, backends.Op) (backends.Op, error)
		input    []float32
		expected []float32
	}{
		{
			name:     "Cos",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Cos(x) },
			input:    []float32{0.0, float32(math.Pi / 2), float32(math.Pi), float32(3 * math.Pi / 2)},
			expected: []float32{1.0, 0.0, -1.0, 0.0},
		},
		{
			name:     "Sin",
			opFunc:   func(b backends.Builder, x backends.Op) (backends.Op, error) { return b.Sin(x) },
			input:    []float32{0.0, float32(math.Pi / 2), float32(math.Pi), float32(3 * math.Pi / 2)},
			expected: []float32{0.0, 1.0, 0.0, -1.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			shape := shapes.Make(dtypes.Float32, len(tc.input))
			x, err := builder.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter() failed: %v", err)
			}

			y, err := tc.opFunc(builder, x)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			exec, err := builder.Compile([]backends.Op{y}, nil)
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

// TestReduceMin tests the ReduceMin operation.
func TestReduceMin(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_min")

	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reduce min along axis 1
	y, err := builder.ReduceMin(x, 1)
	if err != nil {
		t.Fatalf("ReduceMin() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [[3, 1, 2], [6, 4, 5]]
	// Min along axis 1: [1, 4]
	inputData := []float32{3.0, 1.0, 2.0, 6.0, 4.0, 5.0}
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

	expected := []float32{1.0, 4.0}
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

	inputShape := shapes.Make(dtypes.Float32, 4, 4)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Slice from [1,1] with size [2,2]
	y, err := builder.Slice(x, []int{1, 1}, []int{3, 3}, []int{1, 1})
	if err != nil {
		t.Fatalf("Slice() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
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

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := builder.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Compute: (x + 1) * 2
	one := []float32{1.0, 1.0, 1.0, 1.0}
	oneConst, err := builder.Constant(one, 4)
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}

	two := []float32{2.0, 2.0, 2.0, 2.0}
	twoConst, err := builder.Constant(two, 4)
	if err != nil {
		t.Fatalf("Constant() failed: %v", err)
	}

	xPlusOne, err := builder.Add(x, oneConst)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	result, err := builder.Mul(xPlusOne, twoConst)
	if err != nil {
		t.Fatalf("Mul() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{result}, nil)
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

// TestPadOperation tests the Pad operation with edge and start/end padding.
func TestPadOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_pad")

	// Create a small 2x3 tensor
	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create fill value (zero) - must be a scalar (empty dims)
	fillValue, err := builder.Constant([]float32{0.0})
	if err != nil {
		t.Fatalf("Constant(fillValue) failed: %v", err)
	}

	// Pad with [1, 1] before and [1, 1] after for each dimension
	// Result should be [4, 5]
	axesConfig := []backends.PadAxis{
		{Start: 1, End: 1, Interior: 0}, // axis 0: add 1 before, 1 after
		{Start: 1, End: 1, Interior: 0}, // axis 1: add 1 before, 1 after
	}

	y, err := builder.Pad(x, fillValue, axesConfig...)
	if err != nil {
		t.Fatalf("Pad() failed: %v", err)
	}

	// Check output shape
	yShape, err := builder.OpShape(y)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	expectedShape := shapes.Make(dtypes.Float32, 4, 5)
	if !yShape.Equal(expectedShape) {
		t.Errorf("OpShape(y) = %v, want %v", yShape, expectedShape)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 2, 3], [4, 5, 6]]
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 20) // 4 * 5 = 20
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: [[0, 0, 0, 0, 0],
	//            [0, 1, 2, 3, 0],
	//            [0, 4, 5, 6, 0],
	//            [0, 0, 0, 0, 0]]
	expected := []float32{
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 2.0, 3.0, 0.0,
		0.0, 4.0, 5.0, 6.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
	}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestReverseOperation tests the Reverse operation along specified axes.
func TestReverseOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reverse")

	// Create a 2x3 tensor
	inputShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Reverse along axis 1 (columns)
	y, err := builder.Reverse(x, 1)
	if err != nil {
		t.Fatalf("Reverse() failed: %v", err)
	}

	// Check output shape (should be same as input)
	yShape, err := builder.OpShape(y)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	if !yShape.Equal(inputShape) {
		t.Errorf("OpShape(y) = %v, want %v", yShape, inputShape)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 2, 3], [4, 5, 6]]
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

	// Expected: [[3, 2, 1], [6, 5, 4]]
	expected := []float32{3.0, 2.0, 1.0, 6.0, 5.0, 4.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestConvGeneral tests the ConvGeneral operation with 2D convolution.
func TestConvGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_conv_general")

	// Create a simple 2D input: [N=1, C=1, H=4, W=4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter(x) failed: %v", err)
	}

	// Create a simple 2D kernel: [C_out=1, C_in=1, kH=2, kW=2] with all ones
	kernelData := []float32{1.0, 1.0, 1.0, 1.0}
	kernel, err := builder.Constant(kernelData, 1, 1, 2, 2)
	if err != nil {
		t.Fatalf("Constant(kernel) failed: %v", err)
	}

	// Configure axes (NCHW format)
	axes := backends.ConvolveAxesConfig{
		InputBatch:          0,
		InputChannels:       1,
		InputSpatial:        []int{2, 3},
		KernelOutputChannels: 0,
		KernelInputChannels:  1,
		KernelSpatial:       []int{2, 3},
		OutputBatch:         0,
		OutputChannels:      1,
		OutputSpatial:       []int{2, 3},
	}

	// Apply convolution with stride=1, valid padding
	strides := []int{1, 1}
	paddings := [][2]int{{0, 0}, {0, 0}} // no padding
	inputDilations := []int{1, 1}
	kernelDilations := []int{1, 1}
	channelGroupCount := 1
	batchGroupCount := 1

	y, err := builder.ConvGeneral(
		x, kernel, axes, strides, paddings,
		inputDilations, kernelDilations,
		channelGroupCount, batchGroupCount,
	)
	if err != nil {
		t.Fatalf("ConvGeneral() failed: %v", err)
	}

	// Check output shape: [1, 1, 3, 3] (4-2+1=3 for each spatial dimension)
	yShape, err := builder.OpShape(y)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 3, 3)
	if !yShape.Equal(expectedShape) {
		t.Errorf("OpShape(y) = %v, want %v", yShape, expectedShape)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: all ones in a 4x4 grid
	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = 1.0
	}
	xBuf, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 9) // 1 * 1 * 3 * 3 = 9
	err = backend.BufferToFlatData(outputs[0], outputData)
	if err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: each position should be sum of 2x2 kernel = 4.0
	expected := []float32{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-4 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestConcatenate tests the Concatenate operation.
func TestConcatenate(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("ConcatAxis0", func(t *testing.T) {
		builder := backend.Builder("test_concat_axis0")

		// Create two tensors to concatenate along axis 0
		shape1 := shapes.Make(dtypes.Float32, 2, 3)
		shape2 := shapes.Make(dtypes.Float32, 3, 3)

		x, err := builder.Parameter("x", shape1, nil)
		if err != nil {
			t.Fatalf("Parameter(x) failed: %v", err)
		}

		y, err := builder.Parameter("y", shape2, nil)
		if err != nil {
			t.Fatalf("Parameter(y) failed: %v", err)
		}

		// Concatenate along axis 0 -> [5, 3]
		z, err := builder.Concatenate(0, x, y)
		if err != nil {
			t.Fatalf("Concatenate() failed: %v", err)
		}

		// Check output shape
		zShape, err := builder.OpShape(z)
		if err != nil {
			t.Fatalf("OpShape() failed: %v", err)
		}
		expectedShape := shapes.Make(dtypes.Float32, 5, 3)
		if !zShape.Equal(expectedShape) {
			t.Errorf("OpShape(z) = %v, want %v", zShape, expectedShape)
		}

		exec, err := builder.Compile([]backends.Op{z}, nil)
		if err != nil {
			t.Fatalf("Compile() failed: %v", err)
		}
		defer exec.Finalize()

		// Input data:
		// x: [[1, 2, 3], [4, 5, 6]]
		// y: [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
		xData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		yData := []float32{7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}

		xBuf, err := backend.BufferFromFlatData(0, xData, shape1)
		if err != nil {
			t.Fatalf("BufferFromFlatData(x) failed: %v", err)
		}

		yBuf, err := backend.BufferFromFlatData(0, yData, shape2)
		if err != nil {
			t.Fatalf("BufferFromFlatData(y) failed: %v", err)
		}

		outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %v", err)
		}

		zData := make([]float32, 15) // 5 * 3 = 15
		err = backend.BufferToFlatData(outputs[0], zData)
		if err != nil {
			t.Fatalf("BufferToFlatData() failed: %v", err)
		}

		// Expected: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
		expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}
		for i := range expected {
			if math.Abs(float64(zData[i]-expected[i])) > 1e-5 {
				t.Errorf("zData[%d] = %f, want %f", i, zData[i], expected[i])
			}
		}
	})

	t.Run("ConcatAxis1", func(t *testing.T) {
		builder := backend.Builder("test_concat_axis1")

		// Create two tensors to concatenate along axis 1
		shape1 := shapes.Make(dtypes.Float32, 2, 2)
		shape2 := shapes.Make(dtypes.Float32, 2, 3)

		x, err := builder.Parameter("x", shape1, nil)
		if err != nil {
			t.Fatalf("Parameter(x) failed: %v", err)
		}

		y, err := builder.Parameter("y", shape2, nil)
		if err != nil {
			t.Fatalf("Parameter(y) failed: %v", err)
		}

		// Concatenate along axis 1 -> [2, 5]
		z, err := builder.Concatenate(1, x, y)
		if err != nil {
			t.Fatalf("Concatenate() failed: %v", err)
		}

		// Check output shape
		zShape, err := builder.OpShape(z)
		if err != nil {
			t.Fatalf("OpShape() failed: %v", err)
		}
		expectedShape := shapes.Make(dtypes.Float32, 2, 5)
		if !zShape.Equal(expectedShape) {
			t.Errorf("OpShape(z) = %v, want %v", zShape, expectedShape)
		}

		exec, err := builder.Compile([]backends.Op{z}, nil)
		if err != nil {
			t.Fatalf("Compile() failed: %v", err)
		}
		defer exec.Finalize()

		// Input data:
		// x: [[1, 2], [3, 4]]
		// y: [[5, 6, 7], [8, 9, 10]]
		xData := []float32{1.0, 2.0, 3.0, 4.0}
		yData := []float32{5.0, 6.0, 7.0, 8.0, 9.0, 10.0}

		xBuf, err := backend.BufferFromFlatData(0, xData, shape1)
		if err != nil {
			t.Fatalf("BufferFromFlatData(x) failed: %v", err)
		}

		yBuf, err := backend.BufferFromFlatData(0, yData, shape2)
		if err != nil {
			t.Fatalf("BufferFromFlatData(y) failed: %v", err)
		}

		outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %v", err)
		}

		zData := make([]float32, 10) // 2 * 5 = 10
		err = backend.BufferToFlatData(outputs[0], zData)
		if err != nil {
			t.Fatalf("BufferToFlatData() failed: %v", err)
		}

		// Expected: [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]]
		expected := []float32{1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0}
		for i := range expected {
			if math.Abs(float64(zData[i]-expected[i])) > 1e-5 {
				t.Errorf("zData[%d] = %f, want %f", i, zData[i], expected[i])
			}
		}
	})

	t.Run("ConcatThreeTensors", func(t *testing.T) {
		builder := backend.Builder("test_concat_three")

		// Create three tensors to concatenate along axis 0
		shape := shapes.Make(dtypes.Float32, 2, 2)

		x, err := builder.Parameter("x", shape, nil)
		if err != nil {
			t.Fatalf("Parameter(x) failed: %v", err)
		}

		y, err := builder.Parameter("y", shape, nil)
		if err != nil {
			t.Fatalf("Parameter(y) failed: %v", err)
		}

		z, err := builder.Parameter("z", shape, nil)
		if err != nil {
			t.Fatalf("Parameter(z) failed: %v", err)
		}

		// Concatenate along axis 0 -> [6, 2]
		result, err := builder.Concatenate(0, x, y, z)
		if err != nil {
			t.Fatalf("Concatenate() failed: %v", err)
		}

		// Check output shape
		resultShape, err := builder.OpShape(result)
		if err != nil {
			t.Fatalf("OpShape() failed: %v", err)
		}
		expectedShape := shapes.Make(dtypes.Float32, 6, 2)
		if !resultShape.Equal(expectedShape) {
			t.Errorf("OpShape(result) = %v, want %v", resultShape, expectedShape)
		}

		exec, err := builder.Compile([]backends.Op{result}, nil)
		if err != nil {
			t.Fatalf("Compile() failed: %v", err)
		}
		defer exec.Finalize()

		// Input data
		xData := []float32{1.0, 2.0, 3.0, 4.0}
		yData := []float32{5.0, 6.0, 7.0, 8.0}
		zData := []float32{9.0, 10.0, 11.0, 12.0}

		xBuf, err := backend.BufferFromFlatData(0, xData, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(x) failed: %v", err)
		}

		yBuf, err := backend.BufferFromFlatData(0, yData, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(y) failed: %v", err)
		}

		zBuf, err := backend.BufferFromFlatData(0, zData, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(z) failed: %v", err)
		}

		outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf, zBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %v", err)
		}

		resultData := make([]float32, 12) // 6 * 2 = 12
		err = backend.BufferToFlatData(outputs[0], resultData)
		if err != nil {
			t.Fatalf("BufferToFlatData() failed: %v", err)
		}

		// Expected: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
		expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
		for i := range expected {
			if math.Abs(float64(resultData[i]-expected[i])) > 1e-5 {
				t.Errorf("resultData[%d] = %f, want %f", i, resultData[i], expected[i])
			}
		}
	})
}

// TestBatchNormForInference tests the BatchNormForInference operation.
func TestBatchNormForInference(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_batch_norm")

	// Create a simple input: [N=2, C=2, H=1, W=1]
	inputShape := shapes.Make(dtypes.Float32, 2, 2, 1, 1)
	x, err := builder.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter(x) failed: %v", err)
	}

	// Create mean, variance, scale (gamma), offset (beta) tensors for 2 channels
	mean, err := builder.Constant([]float32{0.0, 1.0}, 2)
	if err != nil {
		t.Fatalf("Constant(mean) failed: %v", err)
	}

	variance, err := builder.Constant([]float32{1.0, 1.0}, 2)
	if err != nil {
		t.Fatalf("Constant(variance) failed: %v", err)
	}

	scale, err := builder.Constant([]float32{1.0, 1.0}, 2)
	if err != nil {
		t.Fatalf("Constant(scale) failed: %v", err)
	}

	offset, err := builder.Constant([]float32{0.0, 0.0}, 2)
	if err != nil {
		t.Fatalf("Constant(offset) failed: %v", err)
	}

	epsilon := float32(1e-5)
	featureAxis := 1 // channels axis

	y, err := builder.BatchNormForInference(x, scale, offset, mean, variance, epsilon, featureAxis)
	if err != nil {
		t.Fatalf("BatchNormForInference() failed: %v", err)
	}

	// Check output shape (should match input shape)
	yShape, err := builder.OpShape(y)
	if err != nil {
		t.Fatalf("OpShape() failed: %v", err)
	}
	if !yShape.Equal(inputShape) {
		t.Errorf("OpShape(y) = %v, want %v", yShape, inputShape)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]
	// Batch 0: channel 0 = 1.0, channel 1 = 2.0
	// Batch 1: channel 0 = 3.0, channel 1 = 4.0
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
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

	// Expected normalized values:
	// Channel 0: mean=0, var=1, scale=1, offset=0
	//   - (1.0 - 0.0) / sqrt(1.0 + epsilon) * 1.0 + 0.0 ≈ 1.0
	//   - (3.0 - 0.0) / sqrt(1.0 + epsilon) * 1.0 + 0.0 ≈ 3.0
	// Channel 1: mean=1, var=1, scale=1, offset=0
	//   - (2.0 - 1.0) / sqrt(1.0 + epsilon) * 1.0 + 0.0 ≈ 1.0
	//   - (4.0 - 1.0) / sqrt(1.0 + epsilon) * 1.0 + 0.0 ≈ 3.0
	expected := []float32{1.0, 1.0, 3.0, 3.0}

	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-3 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestRsqrtOperation tests the Rsqrt (reciprocal square root) operation.
func TestRsqrtOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_rsqrt")

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := builder.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Compute rsqrt(x)
	y, err := builder.Rsqrt(x)
	if err != nil {
		t.Fatalf("Rsqrt() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{y}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [1, 4, 9, 16]
	// Expected: [1/sqrt(1), 1/sqrt(4), 1/sqrt(9), 1/sqrt(16)] = [1.0, 0.5, 0.333..., 0.25]
	inputData := []float32{1.0, 4.0, 9.0, 16.0}
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

	expected := []float32{1.0, 0.5, 1.0 / 3.0, 0.25}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-4 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestLogicalOperationsViaWhere tests logical operations indirectly via Where.
// Note: CoreML doesn't allow Bool outputs directly from models, so we test logical
// ops by using them with Where, which converts the result back to float.
func TestLogicalOperationsViaWhere(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	testCases := []struct {
		name     string
		logicFunc func(backends.Builder, backends.Op, backends.Op) (backends.Op, error)
		a        []float32
		b        []float32
		expected []float32
	}{
		{
			name:     "LogicalAnd",
			logicFunc: func(bld backends.Builder, x, y backends.Op) (backends.Op, error) {
				// x > 0.5 && y > 0.5
				half, _ := bld.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
				cond1, _ := bld.GreaterThan(x, half)
				cond2, _ := bld.GreaterThan(y, half)
				return bld.LogicalAnd(cond1, cond2)
			},
			a:        []float32{0.0, 1.0, 0.0, 1.0},
			b:        []float32{0.0, 0.0, 1.0, 1.0},
			expected: []float32{0.0, 0.0, 0.0, 1.0}, // false, false, false, true
		},
		{
			name:     "LogicalOr",
			logicFunc: func(bld backends.Builder, x, y backends.Op) (backends.Op, error) {
				// x > 0.5 || y > 0.5
				half, _ := bld.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
				cond1, _ := bld.GreaterThan(x, half)
				cond2, _ := bld.GreaterThan(y, half)
				return bld.LogicalOr(cond1, cond2)
			},
			a:        []float32{0.0, 1.0, 0.0, 1.0},
			b:        []float32{0.0, 0.0, 1.0, 1.0},
			expected: []float32{0.0, 1.0, 1.0, 1.0}, // false, true, true, true
		},
		{
			name:     "LogicalXor",
			logicFunc: func(bld backends.Builder, x, y backends.Op) (backends.Op, error) {
				// x > 0.5 ^ y > 0.5
				half, _ := bld.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
				cond1, _ := bld.GreaterThan(x, half)
				cond2, _ := bld.GreaterThan(y, half)
				return bld.LogicalXor(cond1, cond2)
			},
			a:        []float32{0.0, 1.0, 0.0, 1.0},
			b:        []float32{0.0, 0.0, 1.0, 1.0},
			expected: []float32{0.0, 1.0, 1.0, 0.0}, // false, true, true, false
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_" + tc.name)

			shape := shapes.Make(dtypes.Float32, 4)
			x, err := builder.Parameter("x", shape, nil)
			if err != nil {
				t.Fatalf("Parameter(x) failed: %v", err)
			}

			y, err := builder.Parameter("y", shape, nil)
			if err != nil {
				t.Fatalf("Parameter(y) failed: %v", err)
			}

			// Create logical condition
			cond, err := tc.logicFunc(builder, x, y)
			if err != nil {
				t.Fatalf("%s() failed: %v", tc.name, err)
			}

			// Create constants for true/false values
			trueConst, _ := builder.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
			falseConst, _ := builder.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)

			// Use Where to convert bool to float
			z, err := builder.Where(cond, trueConst, falseConst)
			if err != nil {
				t.Fatalf("Where() failed: %v", err)
			}

			exec, err := builder.Compile([]backends.Op{z}, nil)
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

			zData := make([]float32, 4)
			err = backend.BufferToFlatData(outputs[0], zData)
			if err != nil {
				t.Fatalf("BufferToFlatData() failed: %v", err)
			}

			for i := range tc.expected {
				if math.Abs(float64(zData[i]-tc.expected[i])) > 1e-5 {
					t.Errorf("%s: zData[%d] = %v, want %v", tc.name, i, zData[i], tc.expected[i])
				}
			}
		})
	}
}

// TestLogicalNotViaWhere tests LogicalNot operation indirectly via Where.
func TestLogicalNotViaWhere(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_logical_not")

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := builder.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create condition: x > 0.5
	half, _ := builder.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
	cond, err := builder.GreaterThan(x, half)
	if err != nil {
		t.Fatalf("GreaterThan() failed: %v", err)
	}

	// Apply NOT
	notCond, err := builder.LogicalNot(cond)
	if err != nil {
		t.Fatalf("LogicalNot() failed: %v", err)
	}

	// Convert to float using Where
	trueConst, _ := builder.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	falseConst, _ := builder.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)

	z, err := builder.Where(notCond, trueConst, falseConst)
	if err != nil {
		t.Fatalf("Where() failed: %v", err)
	}

	exec, err := builder.Compile([]backends.Op{z}, nil)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input: [0.0, 1.0, 0.0, 1.0]
	// x > 0.5: [false, true, false, true]
	// NOT(x > 0.5): [true, false, true, false]
	// Expected: [1.0, 0.0, 1.0, 0.0]
	inputData := []float32{0.0, 1.0, 0.0, 1.0}
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

	expected := []float32{1.0, 0.0, 1.0, 0.0}
	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}
