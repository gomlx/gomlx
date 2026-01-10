// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
)

// TestFunctionCapabilities verifies that the SimpleGo backend reports Functions capability.
func TestFunctionCapabilities(t *testing.T) {
	caps := backend.Capabilities()
	require.True(t, caps.Functions, "SimpleGo should support Functions capability")
}

// TestClosureCreation tests that closures can be created from the main function.
func TestClosureCreation(t *testing.T) {
	builder := backend.Builder("test_closure_creation")
	mainFn := builder.Main()
	require.NotNil(t, mainFn)

	// Create a closure from the main function
	closure, err := mainFn.Closure()
	require.NoError(t, err)
	require.NotNil(t, closure)

	// Verify closure properties
	require.Equal(t, "", closure.Name(), "Closure should have empty name")
	require.Equal(t, mainFn, closure.Parent(), "Closure parent should be main function")
}

// TestNestedClosures tests creating closures within closures.
func TestNestedClosures(t *testing.T) {
	builder := backend.Builder("test_nested_closures")
	mainFn := builder.Main()

	// Create first level closure
	closure1, err := mainFn.Closure()
	require.NoError(t, err)
	require.NotNil(t, closure1)
	require.Equal(t, mainFn, closure1.Parent())

	// Create second level closure
	closure2, err := closure1.Closure()
	require.NoError(t, err)
	require.NotNil(t, closure2)
	require.Equal(t, closure1, closure2.Parent())

	// Verify the chain
	require.Equal(t, "", closure1.Name())
	require.Equal(t, "", closure2.Name())
}

// TestNamedFunctionCreation tests that named functions can be created.
func TestNamedFunctionCreation(t *testing.T) {
	builder := backend.Builder("test_named_function")

	// Create a named function
	fn, err := builder.NewFunction("my_function")
	require.NoError(t, err)
	require.NotNil(t, fn)

	// Verify function properties
	require.Equal(t, "my_function", fn.Name())
	require.Nil(t, fn.Parent(), "Top-level function should have nil parent")
}

// TestEmptyFunctionNameError tests that empty function names are rejected.
func TestEmptyFunctionNameError(t *testing.T) {
	builder := backend.Builder("test_empty_name")

	_, err := builder.NewFunction("")
	require.Error(t, err, "Empty function name should be rejected")
}

// TestClosureParameter tests that parameters can be created in closures.
func TestClosureParameter(t *testing.T) {
	builder := backend.Builder("test_closure_parameter")
	mainFn := builder.Main()

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a parameter in the closure
	param, err := closure.Parameter("input", shapes.Make(dtypes.Float32, 2, 3), nil)
	require.NoError(t, err)
	require.NotNil(t, param)
}

// TestClosureConstant tests that constants can be created in closures.
func TestClosureConstant(t *testing.T) {
	builder := backend.Builder("test_closure_constant")
	mainFn := builder.Main()

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a constant in the closure
	constant, err := closure.Constant([]float32{1.0, 2.0, 3.0}, 3)
	require.NoError(t, err)
	require.NotNil(t, constant)
}

// TestClosureOperations tests that operations can be performed in closures.
func TestClosureOperations(t *testing.T) {
	builder := backend.Builder("test_closure_operations")
	mainFn := builder.Main()

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create constants and perform operations in the closure
	a, err := closure.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	b, err := closure.Constant([]float32{3.0, 4.0}, 2)
	require.NoError(t, err)

	// Add operation in closure
	sum, err := closure.Add(a, b)
	require.NoError(t, err)
	require.NotNil(t, sum)

	// Return from closure
	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)
}

// TestClosureReturn tests that Return() works correctly in closures.
func TestClosureReturn(t *testing.T) {
	builder := backend.Builder("test_closure_return")
	mainFn := builder.Main()

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a constant in the closure
	constant, err := closure.Constant([]float32{1.0, 2.0, 3.0}, 3)
	require.NoError(t, err)

	// Return from closure
	err = closure.Return([]backends.Value{constant}, nil)
	require.NoError(t, err)
}

// TestMultipleClosures tests creating multiple independent closures.
func TestMultipleClosures(t *testing.T) {
	builder := backend.Builder("test_multiple_closures")
	mainFn := builder.Main()

	// Create first closure
	closure1, err := mainFn.Closure()
	require.NoError(t, err)

	// Create second closure
	closure2, err := mainFn.Closure()
	require.NoError(t, err)

	// Both should have the same parent
	require.Equal(t, mainFn, closure1.Parent())
	require.Equal(t, mainFn, closure2.Parent())

	// But they should be different closure instances
	require.NotSame(t, closure1, closure2, "Multiple closures should be distinct instances")
}

// TestClosureFromNamedFunction tests creating closures from named functions.
func TestClosureFromNamedFunction(t *testing.T) {
	builder := backend.Builder("test_closure_from_named")

	// Create a named function
	namedFn, err := builder.NewFunction("helper")
	require.NoError(t, err)

	// Create a closure from the named function
	closure, err := namedFn.Closure()
	require.NoError(t, err)
	require.NotNil(t, closure)

	// Verify closure parent is the named function
	require.Equal(t, namedFn, closure.Parent())
}

// TestControlFlowOpsNotImplemented tests that control flow ops return not implemented errors.
func TestControlFlowOpsNotImplemented(t *testing.T) {
	builder := backend.Builder("test_control_flow")
	mainFn := builder.Main()

	// Create a simple closure for testing
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Sort should return not implemented
	_, err = mainFn.Sort(closure, 0, true)
	require.Error(t, err)

	// While should return not implemented
	_, err = mainFn.While(closure, closure)
	require.Error(t, err)

	// If should return not implemented (need a predicate)
	pred, _ := mainFn.Constant([]bool{true})
	_, err = mainFn.If(pred, closure, closure)
	require.Error(t, err)
}

// TestCallNotImplemented tests that Call returns not implemented error.
func TestCallNotImplemented(t *testing.T) {
	builder := backend.Builder("test_call")
	mainFn := builder.Main()

	// Create a named function
	namedFn, err := builder.NewFunction("helper")
	require.NoError(t, err)

	// Call should return not implemented
	_, err = mainFn.Call(namedFn)
	require.Error(t, err)
}

// TestClosurePreCompilation tests that closures are pre-compiled during Return().
func TestClosurePreCompilation(t *testing.T) {
	builder := backend.Builder("test_closure_precompilation")
	mainFn := builder.Main()

	// Create a closure with operations
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Add a parameter
	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Add a constant
	c, err := closure.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Add operation
	sum, err := closure.Add(x, c)
	require.NoError(t, err)

	// Before Return, compiled should be nil
	closureFn := closure.(*Function)
	require.Nil(t, closureFn.compiled, "Closure should not be compiled before Return()")

	// Return from closure
	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// After Return, compiled should be set
	require.NotNil(t, closureFn.compiled, "Closure should be compiled after Return()")

	// Verify compiled closure properties
	cc := closureFn.compiled
	require.Greater(t, cc.numNodesToProcess, 0, "Should have nodes to process")
	require.Len(t, cc.outputNodes, 1, "Should have one output")
	require.NotNil(t, cc.numUses, "Should have numUses")
}

// TestCompiledClosureExecute tests CompiledClosure.Execute() with a simple add operation.
func TestCompiledClosureExecute(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_execute")
	mainFn := builder.Main()

	// Create a closure: f(x, y) = x + y
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)

	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)

	sum, err := closure.Add(x, y)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Get the compiled closure
	closureFn := closure.(*Function)
	cc := closureFn.Compiled()
	require.NotNil(t, cc, "Should have compiled closure")

	// Create input buffers
	xBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 3),
		flat:  []float32{1.0, 2.0, 3.0},
		valid: true,
	}
	yBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 3),
		flat:  []float32{10.0, 20.0, 30.0},
		valid: true,
	}

	// Execute the closure
	b := backend.(*Backend)
	outputs, err := cc.Execute(b, []*Buffer{xBuf, yBuf}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 1, "Should have one output")

	// Verify the result
	result := outputs[0]
	require.NotNil(t, result)
	require.True(t, result.shape.Equal(shapes.Make(dtypes.Float32, 3)))

	resultFlat := result.flat.([]float32)
	require.Equal(t, []float32{11.0, 22.0, 33.0}, resultFlat)
}

// TestCompiledClosureMultipleExecutions tests executing a closure multiple times with different inputs.
func TestCompiledClosureMultipleExecutions(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_multiple")
	mainFn := builder.Main()

	// Create a closure: f(x) = x * 2
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	two, err := closure.Constant([]float32{2.0, 2.0}, 2)
	require.NoError(t, err)

	product, err := closure.Mul(x, two)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{product}, nil)
	require.NoError(t, err)

	cc := closure.(*Function).Compiled()
	require.NotNil(t, cc)

	b := backend.(*Backend)

	// Execute multiple times with different inputs
	testCases := []struct {
		input    []float32
		expected []float32
	}{
		{[]float32{1.0, 2.0}, []float32{2.0, 4.0}},
		{[]float32{5.0, 10.0}, []float32{10.0, 20.0}},
		{[]float32{-1.0, 0.0}, []float32{-2.0, 0.0}},
	}

	for i, tc := range testCases {
		inputBuf := &Buffer{
			shape: shapes.Make(dtypes.Float32, 2),
			flat:  tc.input,
			valid: true,
		}

		outputs, err := cc.Execute(b, []*Buffer{inputBuf}, nil)
		require.NoError(t, err, "Execution %d failed", i)
		require.Len(t, outputs, 1)

		resultFlat := outputs[0].flat.([]float32)
		require.Equal(t, tc.expected, resultFlat, "Execution %d result mismatch", i)
	}
}

// TestCompiledClosureWithConstants tests a closure that uses only constants.
func TestCompiledClosureWithConstants(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_constants")
	mainFn := builder.Main()

	// Create a closure that returns a constant sum: f() = 1 + 2
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	a, err := closure.Constant([]float32{1.0}, 1)
	require.NoError(t, err)

	b, err := closure.Constant([]float32{2.0}, 1)
	require.NoError(t, err)

	sum, err := closure.Add(a, b)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	cc := closure.(*Function).Compiled()
	require.NotNil(t, cc)

	// Execute with no inputs
	simpleGoBackend := backend.(*Backend)
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	resultFlat := outputs[0].flat.([]float32)
	require.Equal(t, []float32{3.0}, resultFlat)
}

// TestCompiledClosureMultipleOutputs tests a closure with multiple outputs.
func TestCompiledClosureMultipleOutputs(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_multi_outputs")
	mainFn := builder.Main()

	// Create a closure: f(x) = (x+1, x*2)
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	one, err := closure.Constant([]float32{1.0, 1.0}, 2)
	require.NoError(t, err)

	two, err := closure.Constant([]float32{2.0, 2.0}, 2)
	require.NoError(t, err)

	sum, err := closure.Add(x, one)
	require.NoError(t, err)

	product, err := closure.Mul(x, two)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum, product}, nil)
	require.NoError(t, err)

	cc := closure.(*Function).Compiled()
	require.NotNil(t, cc)

	inputBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{5.0, 10.0},
		valid: true,
	}

	b := backend.(*Backend)
	outputs, err := cc.Execute(b, []*Buffer{inputBuf}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 2)

	// First output: x + 1 = [6, 11]
	result0 := outputs[0].flat.([]float32)
	require.Equal(t, []float32{6.0, 11.0}, result0)

	// Second output: x * 2 = [10, 20]
	result1 := outputs[1].flat.([]float32)
	require.Equal(t, []float32{10.0, 20.0}, result1)
}

// TestCompiledClosureChainedOperations tests a closure with chained operations.
func TestCompiledClosureChainedOperations(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_chained")
	mainFn := builder.Main()

	// Create a closure: f(x) = (x + 1) * 2 - 3
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	one, err := closure.Constant([]float32{1.0, 1.0}, 2)
	require.NoError(t, err)

	two, err := closure.Constant([]float32{2.0, 2.0}, 2)
	require.NoError(t, err)

	three, err := closure.Constant([]float32{3.0, 3.0}, 2)
	require.NoError(t, err)

	sum, err := closure.Add(x, one)
	require.NoError(t, err)

	product, err := closure.Mul(sum, two)
	require.NoError(t, err)

	diff, err := closure.Sub(product, three)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{diff}, nil)
	require.NoError(t, err)

	cc := closure.(*Function).Compiled()
	require.NotNil(t, cc)

	// x = [1, 2]
	// (x + 1) = [2, 3]
	// (x + 1) * 2 = [4, 6]
	// (x + 1) * 2 - 3 = [1, 3]
	inputBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{1.0, 2.0},
		valid: true,
	}

	simpleGoBackend := backend.(*Backend)
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{inputBuf}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	resultFlat := outputs[0].flat.([]float32)
	require.Equal(t, []float32{1.0, 3.0}, resultFlat)
}

// TestCompiledClosureInputValidation tests that Execute validates input count.
func TestCompiledClosureInputValidation(t *testing.T) {
	builder := backend.Builder("test_compiled_closure_validation")
	mainFn := builder.Main()

	// Create a closure with 2 parameters
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	sum, err := closure.Add(x, y)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	cc := closure.(*Function).Compiled()
	require.NotNil(t, cc)

	// Try to execute with wrong number of inputs
	xBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{1.0, 2.0},
		valid: true,
	}

	simpleGoBackend := backend.(*Backend)

	// Too few inputs
	_, err = cc.Execute(simpleGoBackend, []*Buffer{xBuf}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expects 2 inputs, got 1")

	// Too many inputs
	_, err = cc.Execute(simpleGoBackend, []*Buffer{xBuf, xBuf, xBuf}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expects 2 inputs, got 3")
}

// TestMainFunctionNotCompiled tests that main functions are not pre-compiled.
func TestMainFunctionNotCompiled(t *testing.T) {
	builder := backend.Builder("test_main_not_compiled")
	mainFn := builder.Main()

	// Create a constant and return it
	c, err := mainFn.Constant([]float32{1.0}, 1)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{c}, nil)
	require.NoError(t, err)

	// Main function should not have a compiled closure
	mainFnImpl := mainFn.(*Function)
	require.Nil(t, mainFnImpl.compiled, "Main function should not be pre-compiled")
}
