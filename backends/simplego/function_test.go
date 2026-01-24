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

// TestControlFlowOpsValidationErrors tests that control flow ops properly validate their inputs.
func TestControlFlowOpsValidationErrors(t *testing.T) {
	builder := backend.Builder("test_control_flow")
	mainFn := builder.Main()

	// Create a closure without calling Return() - this should be rejected
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Sort requires at least one input tensor (validated before closure)
	_, err = mainFn.Sort(closure, 0, true)
	require.Error(t, err)
	require.Contains(t, err.Error(), "requires at least one input tensor")

	// Sort with input should error: closure has no Return() called
	input, _ := mainFn.Constant([]float32{1.0, 2.0}, 2)
	_, err = mainFn.Sort(closure, 0, true, input)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must have Return() called")

	// While requires at least one initial state value (validated before closure)
	_, err = mainFn.While(closure, closure)
	require.Error(t, err)
	require.Contains(t, err.Error(), "requires at least one initial state value")

	// While with state should error: closure has no Return() called
	state, _ := mainFn.Constant([]int32{0})
	_, err = mainFn.While(closure, closure, state)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must have Return() called")

	// If should error: closure has no Return() called
	pred, _ := mainFn.Constant([]bool{true})
	_, err = mainFn.If(pred, closure, closure)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must have Return() called")
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
		inUse: true,
	}
	yBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 3),
		flat:  []float32{10.0, 20.0, 30.0},
		inUse: true,
	}

	// Execute the closure
	b := backend.(*Backend)
	outputs, err := cc.Execute(b, []*Buffer{xBuf, yBuf}, nil, nil, nil)
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
			inUse: true,
		}

		outputs, err := cc.Execute(b, []*Buffer{inputBuf}, nil, nil, nil)
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
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{}, nil, nil, nil)
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
		inUse: true,
	}

	b := backend.(*Backend)
	outputs, err := cc.Execute(b, []*Buffer{inputBuf}, nil, nil, nil)
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
		inUse: true,
	}

	simpleGoBackend := backend.(*Backend)
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{inputBuf}, nil, nil, nil)
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
		inUse: true,
	}

	simpleGoBackend := backend.(*Backend)

	// Too few inputs
	_, err = cc.Execute(simpleGoBackend, []*Buffer{xBuf}, nil, nil, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expects 2 inputs, got 1")

	// Too many inputs
	_, err = cc.Execute(simpleGoBackend, []*Buffer{xBuf, xBuf, xBuf}, nil, nil, nil)
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

// TestClosureCapturingParentNode tests that using a node from a parent function
// (closure capturing) works correctly by creating capture nodes.
func TestClosureCapturingParentNode(t *testing.T) {
	builder := backend.Builder("test_closure_capture")
	mainFn := builder.Main()

	// Create a constant in the main function
	parentNode, err := mainFn.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a parameter in the closure
	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the parent node in the closure - this should create a capture node
	sum, err := closure.Add(parentNode, y)
	require.NoError(t, err, "Using a parent function's node in a closure should work")
	require.NotNil(t, sum)

	// Return the sum
	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Verify the closure has captured the parent node
	closureFn := closure.(*Function)
	require.Len(t, closureFn.capturedParentNodes, 1, "Should have captured one parent node")
	require.Len(t, closureFn.capturedLocalNodes, 1, "Should have one capture node")
}

// TestClosureExecuteWithCapturedValues tests that executing a closure with captured values
// works correctly. This verifies that the function-local nodes architecture handles
// captured value buffers correctly during execution.
func TestClosureExecuteWithCapturedValues(t *testing.T) {
	builder := backend.Builder("test_closure_execute_capture")
	mainFn := builder.Main()

	// Create a constant in the main function that will be captured
	parentConst, err := mainFn.Constant([]float32{10.0, 20.0}, 2)
	require.NoError(t, err)

	// Create a closure that captures the parent constant
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a parameter in the closure
	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the captured parent constant in the closure: result = parentConst + y
	sum, err := closure.Add(parentConst, y)
	require.NoError(t, err)

	// Return the sum
	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Get the compiled closure
	closureFn := closure.(*Function)
	require.Len(t, closureFn.capturedParentNodes, 1, "Should have captured one parent node")

	cc := closureFn.Compiled()
	require.NotNil(t, cc)

	// Prepare the captured value buffer (simulating what an If/While executor would do)
	capturedBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{10.0, 20.0}, // The captured constant value
		inUse: true,
	}

	// Prepare the input parameter buffer: y = [1, 2]
	inputBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{1.0, 2.0},
		inUse: true,
	}

	// Execute the closure with captured values
	// Expected: [10, 20] + [1, 2] = [11, 22]
	simpleGoBackend := backend.(*Backend)
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{inputBuf}, nil, []*Buffer{capturedBuf}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	resultFlat := outputs[0].flat.([]float32)
	require.Equal(t, []float32{11.0, 22.0}, resultFlat)
}

// TestClosureExecuteWithNestedCapturedValues tests that nested closures with captured values
// from grandparent scope work correctly during execution.
func TestClosureExecuteWithNestedCapturedValues(t *testing.T) {
	builder := backend.Builder("test_nested_closure_execute_capture")
	mainFn := builder.Main()

	// Create a constant in the main function (grandparent)
	grandparentConst, err := mainFn.Constant([]float32{100.0, 200.0}, 2)
	require.NoError(t, err)

	// Create first closure (parent) - this will also capture the grandparent value
	closure1, err := mainFn.Closure()
	require.NoError(t, err)

	// Create nested closure (child) that captures the grandparent value
	closure2, err := closure1.Closure()
	require.NoError(t, err)

	// Create a parameter in the nested closure
	y, err := closure2.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the captured grandparent constant: result = grandparentConst * y
	product, err := closure2.Mul(grandparentConst, y)
	require.NoError(t, err)

	// Return the product
	err = closure2.Return([]backends.Value{product}, nil)
	require.NoError(t, err)

	// Verify capture chain: grandparent -> parent capture -> child capture
	closure1Fn := closure1.(*Function)
	closure2Fn := closure2.(*Function)

	// Parent closure should capture the grandparent value
	require.Len(t, closure1Fn.capturedParentNodes, 1, "Parent closure should capture grandparent")

	// Child closure should capture from parent (the parent's capture node)
	require.Len(t, closure2Fn.capturedParentNodes, 1, "Child closure should capture from parent")
	require.Equal(t, closure1Fn.capturedLocalNodes[0], closure2Fn.capturedParentNodes[0],
		"Child should capture parent's capture node, not grandparent directly")

	// Get the compiled closure
	cc := closure2Fn.Compiled()
	require.NotNil(t, cc)

	// Prepare the captured value buffer (the grandparent constant value)
	capturedBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{100.0, 200.0},
		inUse: true,
	}

	// Prepare the input parameter buffer: y = [2, 3]
	inputBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 2),
		flat:  []float32{2.0, 3.0},
		inUse: true,
	}

	// Execute the nested closure with captured values
	// Expected: [100, 200] * [2, 3] = [200, 600]
	simpleGoBackend := backend.(*Backend)
	outputs, err := cc.Execute(simpleGoBackend, []*Buffer{inputBuf}, nil, []*Buffer{capturedBuf}, nil)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	resultFlat := outputs[0].flat.([]float32)
	require.Equal(t, []float32{200.0, 600.0}, resultFlat)
}

// TestClosureCapturingGrandparentNode tests that using a node from a grandparent function
// (nested closure capturing) works correctly by creating capture nodes.
func TestClosureCapturingGrandparentNode(t *testing.T) {
	builder := backend.Builder("test_nested_closure_capture")
	mainFn := builder.Main()

	// Create a constant in the main function
	parentNode, err := mainFn.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Create first closure
	closure1, err := mainFn.Closure()
	require.NoError(t, err)

	// Create second (nested) closure
	closure2, err := closure1.Closure()
	require.NoError(t, err)

	// Create a parameter in the nested closure
	y, err := closure2.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the grandparent node in the nested closure - this should create a capture node
	sum, err := closure2.Add(parentNode, y)
	require.NoError(t, err, "Using a grandparent function's node in a nested closure should work")
	require.NotNil(t, sum)

	// Return from closure2
	err = closure2.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Verify the nested closure has captured the grandparent node
	closure2Fn := closure2.(*Function)
	require.Len(t, closure2Fn.capturedParentNodes, 1, "Should have captured one parent node")
	require.Len(t, closure2Fn.capturedLocalNodes, 1, "Should have one capture node")
}

// TestClosureSameFunctionNodesAllowed tests that using nodes from the same function is allowed.
func TestClosureSameFunctionNodesAllowed(t *testing.T) {
	builder := backend.Builder("test_same_function_nodes")
	mainFn := builder.Main()

	// Create a closure
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create nodes in the closure
	x, err := closure.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	c, err := closure.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Using nodes from the same function should work fine
	sum, err := closure.Add(x, c)
	require.NoError(t, err)
	require.NotNil(t, sum)

	// Return should also work
	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)
}

// TestCapturedParentNodesPropagation tests that captured values are properly tracked
// for DAG dependency and lifetime management.
func TestCapturedParentNodesPropagation(t *testing.T) {
	builder := backend.Builder("test_captured_values_propagation")
	mainFn := builder.Main()

	// Create a constant in the main function
	parentValue, err := mainFn.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Create a closure that captures the parent value
	closure, err := mainFn.Closure()
	require.NoError(t, err)

	// Create a parameter in the closure
	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the parent value in the closure
	sum, err := closure.Add(parentValue, y)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Verify the closure's captured values
	closureFn := closure.(*Function)
	require.Len(t, closureFn.capturedParentNodes, 1)
	require.Equal(t, parentValue.(*Node), closureFn.capturedParentNodes[0])

	// Verify that CapturedParentNodes() returns the list
	captured := closureFn.CapturedParentNodes()
	require.Len(t, captured, 1)
	require.Equal(t, parentValue.(*Node), captured[0])
}

// TestAddNodeCapturedInputs tests that AddNodeCapturedInputs properly sets up
// captured inputs on a node for DAG tracking.
func TestAddNodeCapturedInputs(t *testing.T) {
	builder := backend.Builder("test_add_node_captured_inputs")
	mainFnImpl := builder.Main().(*Function)

	// Create a value in the main function
	parentValue, err := mainFnImpl.Constant([]float32{1.0, 2.0}, 2)
	require.NoError(t, err)

	// Create a closure that captures the parent value
	closure, err := mainFnImpl.Closure()
	require.NoError(t, err)

	y, err := closure.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	sum, err := closure.Add(parentValue, y)
	require.NoError(t, err)

	err = closure.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	closureFn := closure.(*Function)

	// Create a dummy node (simulating an If/While op that uses the closure)
	dummyNode := &Node{
		idx:      999,
		opType:   backends.OpTypeIdentity,
		function: mainFnImpl,
	}

	// Add captured inputs to the node
	dummyNode.AddNodeCapturedInputs(closureFn)

	// Verify the node has captured inputs (one closure with one captured value)
	require.Len(t, dummyNode.capturedInputs, 1)
	require.Len(t, dummyNode.capturedInputs[0], 1)
	require.Equal(t, parentValue.(*Node), dummyNode.capturedInputs[0][0])
}

// TestNestedClosureCaptureChain tests that nested closures properly propagate
// captures through intermediate closures.
func TestNestedClosureCaptureChain(t *testing.T) {
	builder := backend.Builder("test_nested_closure_chain")
	mainFn := builder.Main()

	// Create a value in the main function (grandparent)
	grandparentValue, err := mainFn.Constant([]float32{10.0, 20.0}, 2)
	require.NoError(t, err)

	// Create first closure (parent)
	closure1, err := mainFn.Closure()
	require.NoError(t, err)

	// Create second closure (child) - nested
	closure2, err := closure1.Closure()
	require.NoError(t, err)

	// Create a parameter in the nested closure
	y, err := closure2.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	require.NoError(t, err)

	// Use the grandparent value in the nested closure
	// This should trigger capture propagation: grandparent -> parent -> child
	sum, err := closure2.Add(grandparentValue, y)
	require.NoError(t, err)

	err = closure2.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	// Verify the chain:
	// 1. Parent closure (closure1) should capture the grandparent value
	closure1Fn := closure1.(*Function)
	require.Len(t, closure1Fn.capturedParentNodes, 1)
	require.Equal(t, grandparentValue.(*Node), closure1Fn.capturedParentNodes[0])

	// 2. Child closure (closure2) should capture the parent's capture node
	closure2Fn := closure2.(*Function)
	require.Len(t, closure2Fn.capturedParentNodes, 1)
	// The captured value should be the parent's capture node, not the original
	require.Equal(t, closure1Fn.capturedLocalNodes[0], closure2Fn.capturedParentNodes[0])
}

// TestIfOperation tests the If control flow operation.
func TestIfOperation(t *testing.T) {
	builder := backend.Builder("test_if")
	mainFn := builder.Main()

	// Create true branch: returns constant 10
	trueBranch, err := mainFn.Closure()
	require.NoError(t, err)
	trueConst, err := trueBranch.Constant([]int32{10})
	require.NoError(t, err)
	err = trueBranch.Return([]backends.Value{trueConst}, nil)
	require.NoError(t, err)

	// Create false branch: returns constant 20
	falseBranch, err := mainFn.Closure()
	require.NoError(t, err)
	falseConst, err := falseBranch.Constant([]int32{20})
	require.NoError(t, err)
	err = falseBranch.Return([]backends.Value{falseConst}, nil)
	require.NoError(t, err)

	// Create predicate parameter
	pred, err := mainFn.Parameter("pred", shapes.Make(dtypes.Bool), nil)
	require.NoError(t, err)

	// Create If operation
	results, err := mainFn.If(pred, trueBranch, falseBranch)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Return the result
	err = mainFn.Return(results, nil)
	require.NoError(t, err)

	// Compile and execute with true
	exec, err := builder.Compile()
	require.NoError(t, err)

	trueInput := &Buffer{shape: shapes.Make(dtypes.Bool), flat: []bool{true}, inUse: true}
	outputs, err := exec.Execute([]backends.Buffer{trueInput}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	require.Equal(t, []int32{10}, outputs[0].(*Buffer).flat)

	// Execute with false
	falseInput := &Buffer{shape: shapes.Make(dtypes.Bool), flat: []bool{false}, inUse: true}
	outputs, err = exec.Execute([]backends.Buffer{falseInput}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	require.Equal(t, []int32{20}, outputs[0].(*Buffer).flat)
}

// TestWhileOperation tests the While control flow operation.
func TestWhileOperation(t *testing.T) {
	builder := backend.Builder("test_while")
	mainFn := builder.Main()

	// Create condition closure: counter < 5
	cond, err := mainFn.Closure()
	require.NoError(t, err)
	condCounter, err := cond.Parameter("counter", shapes.Make(dtypes.Int32), nil)
	require.NoError(t, err)
	condLimit, err := cond.Constant([]int32{5})
	require.NoError(t, err)
	condResult, err := cond.LessThan(condCounter, condLimit)
	require.NoError(t, err)
	err = cond.Return([]backends.Value{condResult}, nil)
	require.NoError(t, err)

	// Create body closure: counter + 1
	body, err := mainFn.Closure()
	require.NoError(t, err)
	bodyCounter, err := body.Parameter("counter", shapes.Make(dtypes.Int32), nil)
	require.NoError(t, err)
	bodyOne, err := body.Constant([]int32{1})
	require.NoError(t, err)
	bodyResult, err := body.Add(bodyCounter, bodyOne)
	require.NoError(t, err)
	err = body.Return([]backends.Value{bodyResult}, nil)
	require.NoError(t, err)

	// Create initial state
	initCounter, err := mainFn.Constant([]int32{0})
	require.NoError(t, err)

	// Create While operation
	results, err := mainFn.While(cond, body, initCounter)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Return the result
	err = mainFn.Return(results, nil)
	require.NoError(t, err)

	// Compile and execute
	exec, err := builder.Compile()
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	require.Equal(t, []int32{5}, outputs[0].(*Buffer).flat)
}

// TestSortOperation tests the Sort control flow operation.
func TestSortOperation(t *testing.T) {
	builder := backend.Builder("test_sort")
	mainFn := builder.Main()

	// Create comparator closure: lhs < rhs (ascending sort)
	comp, err := mainFn.Closure()
	require.NoError(t, err)
	lhs, err := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	rhs, err := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	compResult, err := comp.LessThan(lhs, rhs)
	require.NoError(t, err)
	err = comp.Return([]backends.Value{compResult}, nil)
	require.NoError(t, err)

	// Create input parameter
	input, err := mainFn.Parameter("input", shapes.Make(dtypes.Float32, 5), nil)
	require.NoError(t, err)

	// Create Sort operation
	results, err := mainFn.Sort(comp, 0, false, input)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Return the result
	err = mainFn.Return(results, nil)
	require.NoError(t, err)

	// Compile and execute
	exec, err := builder.Compile()
	require.NoError(t, err)

	inputBuf := &Buffer{
		shape: shapes.Make(dtypes.Float32, 5),
		flat:  []float32{5.0, 2.0, 8.0, 1.0, 3.0},
		inUse: true,
	}
	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	require.Equal(t, []float32{1.0, 2.0, 3.0, 5.0, 8.0}, outputs[0].(*Buffer).flat)
}

// TestClosureCaptureExecutionWithIf tests that captured values work correctly with If operations.
func TestClosureCaptureExecutionWithIf(t *testing.T) {
	builder := backend.Builder("test_closure_capture_if")
	mainFn := builder.Main()

	// Create a constant in the main function that will be captured
	capturedConst, err := mainFn.Constant([]float32{10.0, 20.0}, 2)
	require.NoError(t, err)

	// Create parameter for the predicate
	pred, err := mainFn.Parameter("pred", shapes.Make(dtypes.Bool), nil)
	require.NoError(t, err)

	// Create true branch that uses the captured constant
	trueBranch, err := mainFn.Closure()
	require.NoError(t, err)

	// In true branch: return capturedConst * 2
	two, err := trueBranch.Constant([]float32{2.0, 2.0}, 2)
	require.NoError(t, err)
	trueResult, err := trueBranch.Mul(capturedConst, two)
	require.NoError(t, err)
	err = trueBranch.Return([]backends.Value{trueResult}, nil)
	require.NoError(t, err)

	// Create false branch that uses the captured constant
	falseBranch, err := mainFn.Closure()
	require.NoError(t, err)

	// In false branch: return capturedConst / 2
	half, err := falseBranch.Constant([]float32{0.5, 0.5}, 2)
	require.NoError(t, err)
	falseResult, err := falseBranch.Mul(capturedConst, half)
	require.NoError(t, err)
	err = falseBranch.Return([]backends.Value{falseResult}, nil)
	require.NoError(t, err)

	// Create If operation
	ifOutputs, err := mainFn.If(pred, trueBranch, falseBranch)
	require.NoError(t, err)

	// Return the If result
	err = mainFn.Return(ifOutputs, nil)
	require.NoError(t, err)

	// Compile and execute
	exec, err := builder.Compile()
	require.NoError(t, err)

	// Test with pred = true
	trueInput := &Buffer{shape: shapes.Make(dtypes.Bool), flat: []bool{true}, inUse: true}
	outputs, err := exec.Execute([]backends.Buffer{trueInput}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	resultFlat := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{20.0, 40.0}, resultFlat, "True branch should return capturedConst * 2")

	// Test with pred = false
	falseInput := &Buffer{shape: shapes.Make(dtypes.Bool), flat: []bool{false}, inUse: true}
	outputs, err = exec.Execute([]backends.Buffer{falseInput}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	resultFlat = outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{5.0, 10.0}, resultFlat, "False branch should return capturedConst / 2")
}

// TestClosureCaptureExecutionWithWhile tests that captured values work correctly with While operations.
func TestClosureCaptureExecutionWithWhile(t *testing.T) {
	builder := backend.Builder("test_closure_capture_while")
	mainFn := builder.Main()

	// Create a constant in the main function that will be captured by the body (scalar)
	addAmount, err := mainFn.Constant([]float32{1.0})
	require.NoError(t, err)

	// Create a threshold constant for the condition (scalar)
	threshold, err := mainFn.Constant([]float32{5.0})
	require.NoError(t, err)

	// Create parameter for initial counter value (scalar)
	counter, err := mainFn.Parameter("counter", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)

	// Create condition: counter < threshold (returns scalar boolean)
	cond, err := mainFn.Closure()
	require.NoError(t, err)
	condCounter, err := cond.Parameter("counter", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	condResult, err := cond.LessThan(condCounter, threshold) // Uses captured threshold
	require.NoError(t, err)
	err = cond.Return([]backends.Value{condResult}, nil)
	require.NoError(t, err)

	// Create body: counter + addAmount (uses captured addAmount)
	body, err := mainFn.Closure()
	require.NoError(t, err)
	bodyCounter, err := body.Parameter("counter", shapes.Make(dtypes.Float32), nil)
	require.NoError(t, err)
	newCounter, err := body.Add(bodyCounter, addAmount) // Uses captured addAmount
	require.NoError(t, err)
	err = body.Return([]backends.Value{newCounter}, nil)
	require.NoError(t, err)

	// Create While operation
	whileOutputs, err := mainFn.While(cond, body, counter)
	require.NoError(t, err)

	// Return the While result
	err = mainFn.Return(whileOutputs, nil)
	require.NoError(t, err)

	// Compile and execute
	exec, err := builder.Compile()
	require.NoError(t, err)

	// Test with initial counter = 0 (scalar)
	counterInput := &Buffer{shape: shapes.Make(dtypes.Float32), flat: []float32{0.0}, inUse: true}
	outputs, err := exec.Execute([]backends.Buffer{counterInput}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	resultFlat := outputs[0].(*Buffer).flat.([]float32)
	// Should loop until counter >= 5.0, so 0+1+1+1+1+1 = 5
	require.Equal(t, []float32{5.0}, resultFlat, "While should loop until counter >= threshold")
}
