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
