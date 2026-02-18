// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	// Perform your setup logic here
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", backends.ConfigEnvVar, os.Getenv(backends.ConfigEnvVar))
	}
	backend = backends.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

// buildGraph compiles a backend graph from the given input shapes and build function,
// and creates input buffers from the provided data. Used by both test and benchmark helpers.
func buildGraph(inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f backends.Function, params []backends.Value) (backends.Value, error),
) (backends.Executable, []backends.Buffer, error) {
	builder := backend.Builder("test")
	mainFn := builder.Main()

	params := make([]backends.Value, len(inputShapes))
	for i, s := range inputShapes {
		p, err := mainFn.Parameter(fmt.Sprintf("x%d", i), s, nil)
		if err != nil {
			return nil, nil, err
		}
		params[i] = p
	}

	out, err := buildFn(mainFn, params)
	if err != nil {
		return nil, nil, err
	}

	if err := mainFn.Return([]backends.Value{out}, nil); err != nil {
		return nil, nil, err
	}

	exec, err := builder.Compile()
	if err != nil {
		return nil, nil, err
	}

	inputs := make([]backends.Buffer, len(inputDatas))
	for i, data := range inputDatas {
		buf, err := backend.BufferFromFlatData(0, data, inputShapes[i])
		if err != nil {
			return nil, nil, err
		}
		inputs[i] = buf
	}

	return exec, inputs, nil
}

// testBackend builds, compiles, and executes a single-input, single-output backend graph.
func testBackend(t *testing.T, inputShape shapes.Shape, inputData any,
	buildFn func(f backends.Function, param backends.Value) (backends.Value, error),
) *Buffer {
	t.Helper()
	return testBackendMultiInput(t, []shapes.Shape{inputShape}, []any{inputData},
		func(f backends.Function, params []backends.Value) (backends.Value, error) {
			return buildFn(f, params[0])
		},
	)
}

// testBackendMultiInput builds, compiles, and executes a multi-input, single-output backend graph.
func testBackendMultiInput(t *testing.T, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f backends.Function, params []backends.Value) (backends.Value, error),
) *Buffer {
	t.Helper()
	exec, inputBufs, err := buildGraph(inputShapes, inputDatas, buildFn)
	require.NoError(t, err)
	outputs, err := exec.Execute(inputBufs, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	return outputs[0].(*Buffer)
}

func TestDuplicatedOutputNodes(t *testing.T) {
	// Create a builder and a node
	builder := backend.Builder("test_duplicated_outputs")
	mainFn := builder.Main()
	node, err := mainFn.Constant([]float32{1.0, 2.0, 3.0}, 3)
	require.NoError(t, err)
	require.NotNil(t, node)

	// Compile with the same node duplicated as outputs
	// This should create Identity nodes for the duplicate
	err = mainFn.Return([]backends.Value{node, node}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)
	require.NotNil(t, exec)

	// Execute with no inputs (since we're using a constant)
	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)

	// Verify that the two output buffers are different (not the same pointer)
	output0 := outputs[0].(*Buffer)
	output1 := outputs[1].(*Buffer)
	require.NotSame(t, output0, output1, "duplicated output nodes should yield different buffers")

	// Verify that the underlying flat data slices are also different
	// (they may have the same values but should be different slices)
	flat0 := output0.flat.([]float32)
	flat1 := output1.flat.([]float32)
	require.NotSame(t, &flat0[0], &flat1[0], "duplicated output nodes should have different underlying data slices")

	// Verify that the values are correct (both should be [1.0, 2.0, 3.0])
	require.Equal(t, []float32{1.0, 2.0, 3.0}, flat0)
	require.Equal(t, []float32{1.0, 2.0, 3.0}, flat1)

	// Verify shapes are correct
	shape0, err := backend.BufferShape(outputs[0])
	require.NoError(t, err)
	require.True(t, shape0.Equal(shapes.Make(dtypes.Float32, 3)))

	shape1, err := backend.BufferShape(outputs[1])
	require.NoError(t, err)
	require.True(t, shape1.Equal(shapes.Make(dtypes.Float32, 3)))
}
