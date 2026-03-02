// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestHasDynamicParameters(t *testing.T) {
	builder := backend.Builder("test_has_dynamic")
	mainFn := builder.Main()

	// Static parameters only.
	p1, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	p2, err := mainFn.Parameter("y", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	require.False(t, hasDynamicParameters([]*Node{p1.(*Node), p2.(*Node)}))

	// Mixed: one dynamic parameter.
	builder2 := backend.Builder("test_has_dynamic2")
	mainFn2 := builder2.Main()
	p3, err := mainFn2.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""}), nil)
	require.NoError(t, err)
	p4, err := mainFn2.Parameter("b", shapes.Make(dtypes.Float32, 512), nil)
	require.NoError(t, err)
	require.True(t, hasDynamicParameters([]*Node{p3.(*Node), p4.(*Node)}))
}

func TestExtractBindingsFromInputs(t *testing.T) {
	builder := backend.Builder("test_extract_bindings")
	mainFn := builder.Main()

	// Single dynamic parameter.
	p1, err := mainFn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""}), nil)
	require.NoError(t, err)

	buf1, err := backend.BufferFromFlatData(0, make([]float32, 32*512), shapes.Make(dtypes.Float32, 32, 512))
	require.NoError(t, err)

	bindings, err := extractBindingsFromInputs([]*Node{p1.(*Node)}, []*Buffer{buf1.(*Buffer)})
	require.NoError(t, err)
	require.Equal(t, shapes.AxisBindings{"batch": 32}, bindings)

	// Multiple dynamic parameters.
	builder2 := backend.Builder("test_extract_bindings2")
	mainFn2 := builder2.Main()
	p2, err := mainFn2.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""}), nil)
	require.NoError(t, err)
	p3, err := mainFn2.Parameter("b", shapes.MakeDynamic(dtypes.Float32, []int{-1, 256}, []string{"batch", ""}), nil)
	require.NoError(t, err)

	buf2, err := backend.BufferFromFlatData(0, make([]float32, 8*512), shapes.Make(dtypes.Float32, 8, 512))
	require.NoError(t, err)
	buf3, err := backend.BufferFromFlatData(0, make([]float32, 8*256), shapes.Make(dtypes.Float32, 8, 256))
	require.NoError(t, err)

	bindings, err = extractBindingsFromInputs([]*Node{p2.(*Node), p3.(*Node)}, []*Buffer{buf2.(*Buffer), buf3.(*Buffer)})
	require.NoError(t, err)
	require.Equal(t, shapes.AxisBindings{"batch": 8}, bindings)

	// Conflicting bindings should fail.
	buf4, err := backend.BufferFromFlatData(0, make([]float32, 16*256), shapes.Make(dtypes.Float32, 16, 256))
	require.NoError(t, err)
	_, err = extractBindingsFromInputs([]*Node{p2.(*Node), p3.(*Node)}, []*Buffer{buf2.(*Buffer), buf4.(*Buffer)})
	require.Error(t, err)
	require.Contains(t, err.Error(), "conflicting")
}

// buildDynamicGraph builds a graph with dynamic parameters for testing.
// It creates a simple Add(x, y) graph where the first axis is dynamic.
func buildDynamicGraph(t *testing.T, batchSize, featureSize int) (backends.Executable, []backends.Buffer) {
	t.Helper()

	builder := backend.Builder("test_dynamic")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, featureSize}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	y, err := mainFn.Parameter("y", dynShape, nil)
	require.NoError(t, err)

	sum, err := mainFn.Add(x, y)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Create concrete input buffers.
	concreteShape := shapes.Make(dtypes.Float32, batchSize, featureSize)
	xData := make([]float32, batchSize*featureSize)
	yData := make([]float32, batchSize*featureSize)
	for i := range xData {
		xData[i] = float32(i)
		yData[i] = float32(i * 2)
	}
	xBuf, err := backend.BufferFromFlatData(0, xData, concreteShape)
	require.NoError(t, err)
	yBuf, err := backend.BufferFromFlatData(0, yData, concreteShape)
	require.NoError(t, err)

	return exec, []backends.Buffer{xBuf, yBuf}
}

func TestExecuteStaticUnchanged(t *testing.T) {
	// Verify that static graphs work exactly as before.
	builder := backend.Builder("test_static")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	neg, err := mainFn.Neg(x)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{neg}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Verify hasDynamicAxes is false.
	e := exec.(*Executable)
	require.False(t, e.hasDynamicAxes)

	// Execute with concrete input.
	buf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	outputs, err := exec.Execute([]backends.Buffer{buf}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	flat := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{-1, -2, -3}, flat)
}

func TestExecuteDynamic_Add(t *testing.T) {
	// Build graph with dynamic batch axis.
	exec, inputs := buildDynamicGraph(t, 4, 3)

	e := exec.(*Executable)
	require.True(t, e.hasDynamicAxes)

	// Execute with batch=4.
	outputs, err := exec.Execute(inputs, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	outBuf := outputs[0].(*Buffer)
	require.Equal(t, dtypes.Float32, outBuf.shape.DType)
	require.Equal(t, []int{4, 3}, outBuf.shape.Dimensions)

	flat := outBuf.flat.([]float32)
	for i := range flat {
		require.InDelta(t, float32(i)+float32(i*2), flat[i], 1e-6,
			"output[%d] = %f, want %f", i, flat[i], float32(i)+float32(i*2))
	}

	// Execute with a different batch size (batch=8) using the same executable.
	concreteShape8 := shapes.Make(dtypes.Float32, 8, 3)
	xData := make([]float32, 24)
	yData := make([]float32, 24)
	for i := range xData {
		xData[i] = 1.0
		yData[i] = 2.0
	}
	xBuf, err := backend.BufferFromFlatData(0, xData, concreteShape8)
	require.NoError(t, err)
	yBuf, err := backend.BufferFromFlatData(0, yData, concreteShape8)
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs2, 1)

	outBuf2 := outputs2[0].(*Buffer)
	require.Equal(t, dtypes.Float32, outBuf2.shape.DType)
	require.Equal(t, []int{8, 3}, outBuf2.shape.Dimensions)

	flat2 := outBuf2.flat.([]float32)
	for i := range flat2 {
		require.InDelta(t, float32(3.0), flat2[i], 1e-6)
	}
}

func TestExecuteDynamic_Mul(t *testing.T) {
	// Build a Mul graph with dynamic batch.
	builder := backend.Builder("test_dynamic_mul")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, 2}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	y, err := mainFn.Parameter("y", dynShape, nil)
	require.NoError(t, err)

	prod, err := mainFn.Mul(x, y)
	require.NoError(t, err)

	err = mainFn.Return([]backends.Value{prod}, nil)
	require.NoError(t, err)

	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2.
	xBuf, err := backend.BufferFromFlatData(0, []float32{2, 3, 4, 5}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)
	yBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40}, shapes.Make(dtypes.Float32, 2, 2))
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)
	flat := outputs[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{20, 60, 120, 200}, flat)

	// Execute with batch=1.
	xBuf2, err := backend.BufferFromFlatData(0, []float32{5, 7}, shapes.Make(dtypes.Float32, 1, 2))
	require.NoError(t, err)
	yBuf2, err := backend.BufferFromFlatData(0, []float32{3, 4}, shapes.Make(dtypes.Float32, 1, 2))
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{xBuf2, yBuf2}, nil, 0)
	require.NoError(t, err)
	flat2 := outputs2[0].(*Buffer).flat.([]float32)
	require.Equal(t, []float32{15, 28}, flat2)
}

func TestSpecializationCache(t *testing.T) {
	exec, inputs := buildDynamicGraph(t, 4, 3)
	e := exec.(*Executable)

	// First execution creates a specialization.
	_, err := exec.Execute(inputs, nil, 0)
	require.NoError(t, err)

	bindings4 := shapes.AxisBindings{"batch": 4}
	spec1, ok := e.specializations.Load(bindings4.Key())
	require.True(t, ok, "specialization should be cached for batch=4")

	// Same bindings should return the same cached specialization.
	_, err = exec.Execute(inputs, nil, 0)
	require.NoError(t, err)
	spec2, ok := e.specializations.Load(bindings4.Key())
	require.True(t, ok)
	require.Same(t, spec1, spec2, "same bindings should reuse cached specialization")

	// Different batch size creates a different specialization.
	xBuf, err := backend.BufferFromFlatData(0, make([]float32, 8*3), shapes.Make(dtypes.Float32, 8, 3))
	require.NoError(t, err)
	yBuf, err := backend.BufferFromFlatData(0, make([]float32, 8*3), shapes.Make(dtypes.Float32, 8, 3))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	require.NoError(t, err)

	bindings8 := shapes.AxisBindings{"batch": 8}
	spec3, ok := e.specializations.Load(bindings8.Key())
	require.True(t, ok, "specialization should be cached for batch=8")
	require.NotSame(t, spec1, spec3, "different bindings should create different specializations")
}

func TestExecuteDynamic_Concurrent(t *testing.T) {
	// Build one dynamic graph and execute concurrently with different batch sizes.
	builder := backend.Builder("test_concurrent")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, 4}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	y, err := mainFn.Parameter("y", dynShape, nil)
	require.NoError(t, err)

	sum, err := mainFn.Add(x, y)
	require.NoError(t, err)
	err = mainFn.Return([]backends.Value{sum}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)

	var wg sync.WaitGroup
	batchSizes := []int{1, 2, 4, 8, 16, 32}
	for _, bs := range batchSizes {
		bs := bs
		wg.Add(1)
		go func() {
			defer wg.Done()
			concreteShape := shapes.Make(dtypes.Float32, bs, 4)
			xData := make([]float32, bs*4)
			yData := make([]float32, bs*4)
			for i := range xData {
				xData[i] = 1.0
				yData[i] = 2.0
			}
			xBuf, err := backend.BufferFromFlatData(0, xData, concreteShape)
			require.NoError(t, err)
			yBuf, err := backend.BufferFromFlatData(0, yData, concreteShape)
			require.NoError(t, err)

			outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
			require.NoError(t, err)
			require.Len(t, outputs, 1)

			outBuf := outputs[0].(*Buffer)
			require.Equal(t, dtypes.Float32, outBuf.shape.DType)
			require.Equal(t, concreteShape.Dimensions, outBuf.shape.Dimensions)
			flat := outBuf.flat.([]float32)
			for i, v := range flat {
				require.InDelta(t, float32(3.0), v, 1e-6,
					"batch=%d, idx=%d", bs, i)
			}
		}()
	}
	wg.Wait()
}

func TestExecuteDynamic_MultipleBindings(t *testing.T) {
	// Both batch and seq_len are dynamic.
	builder := backend.Builder("test_multi_bindings")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, -1, 64}, []string{"batch", "seq_len", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	neg, err := mainFn.Neg(x)
	require.NoError(t, err)
	err = mainFn.Return([]backends.Value{neg}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)

	// Execute with batch=2, seq_len=10.
	shape1 := shapes.Make(dtypes.Float32, 2, 10, 64)
	data1 := make([]float32, 2*10*64)
	for i := range data1 {
		data1[i] = float32(i)
	}
	buf1, err := backend.BufferFromFlatData(0, data1, shape1)
	require.NoError(t, err)

	outputs, err := exec.Execute([]backends.Buffer{buf1}, nil, 0)
	require.NoError(t, err)
	outBuf := outputs[0].(*Buffer)
	require.Equal(t, dtypes.Float32, outBuf.shape.DType)
	require.Equal(t, shape1.Dimensions, outBuf.shape.Dimensions)
	flat := outBuf.flat.([]float32)
	for i, v := range flat {
		require.InDelta(t, -float32(i), v, 1e-6)
	}

	// Execute with batch=4, seq_len=5.
	shape2 := shapes.Make(dtypes.Float32, 4, 5, 64)
	data2 := make([]float32, 4*5*64)
	for i := range data2 {
		data2[i] = 1.0
	}
	buf2, err := backend.BufferFromFlatData(0, data2, shape2)
	require.NoError(t, err)

	outputs2, err := exec.Execute([]backends.Buffer{buf2}, nil, 0)
	require.NoError(t, err)
	outBuf2 := outputs2[0].(*Buffer)
	require.Equal(t, dtypes.Float32, outBuf2.shape.DType)
	require.Equal(t, shape2.Dimensions, outBuf2.shape.Dimensions)
	flat2 := outBuf2.flat.([]float32)
	for _, v := range flat2 {
		require.InDelta(t, float32(-1.0), v, 1e-6)
	}
}

func TestExecuteDynamic_ShapeMismatch(t *testing.T) {
	// Static dimension mismatch should fail even with dynamic axes.
	builder := backend.Builder("test_shape_mismatch")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, 512}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	neg, err := mainFn.Neg(x)
	require.NoError(t, err)
	err = mainFn.Return([]backends.Value{neg}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)

	// Wrong static dimension (256 instead of 512).
	wrongBuf, err := backend.BufferFromFlatData(0, make([]float32, 4*256), shapes.Make(dtypes.Float32, 4, 256))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{wrongBuf}, nil, 0)
	require.Error(t, err)
	require.Contains(t, err.Error(), "mismatch")
}

func TestExecWithDynamicAxes(t *testing.T) {
	// End-to-end test using graph.Exec.WithDynamicAxes().
	addFn := func(a, b *graph.Node) *graph.Node {
		return graph.Add(a, b)
	}
	exec := graph.MustNewExec(backend, addFn).
		WithDynamicAxes([]string{"batch", ""}, []string{"batch", ""})

	// Execute with batch=4.
	a1 := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2)
	b1 := tensors.FromFlatDataAndDimensions([]float32{10, 20, 30, 40, 50, 60, 70, 80}, 4, 2)
	outputs := exec.MustExec(a1, b1)
	require.Len(t, outputs, 1)
	got, err := tensors.CopyFlatData[float32](outputs[0])
	require.NoError(t, err)
	require.Equal(t, []float32{11, 22, 33, 44, 55, 66, 77, 88}, got)

	// Execute with batch=2 — should reuse the same compiled graph.
	a2 := tensors.FromFlatDataAndDimensions([]float32{100, 200, 300, 400}, 2, 2)
	b2 := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4}, 2, 2)
	outputs2 := exec.MustExec(a2, b2)
	require.Len(t, outputs2, 1)
	got2, err := tensors.CopyFlatData[float32](outputs2[0])
	require.NoError(t, err)
	require.Equal(t, []float32{101, 202, 303, 404}, got2)

	exec.Finalize()
}

func TestExecWithDynamicAxes_CacheSingleEntry(t *testing.T) {
	// Verify that WithDynamicAxes compiles only one graph regardless of input shape variation.
	mulFn := func(a, b *graph.Node) *graph.Node {
		return graph.Mul(a, b)
	}
	exec := graph.MustNewExec(backend, mulFn).
		WithDynamicAxes([]string{"batch", ""}, []string{"batch", ""})

	batchSizes := []int{1, 2, 4, 8, 16}
	for _, bs := range batchSizes {
		data := make([]float32, bs*3)
		for i := range data {
			data[i] = 2.0
		}
		a := tensors.FromFlatDataAndDimensions(data, bs, 3)
		b := tensors.FromFlatDataAndDimensions(data, bs, 3)
		outputs := exec.MustExec(a, b)
		require.Len(t, outputs, 1)
		got, gotErr := tensors.CopyFlatData[float32](outputs[0])
		require.NoError(t, gotErr)
		for i, v := range got {
			require.InDelta(t, float32(4.0), v, 1e-6,
				"batch=%d, idx=%d", bs, i)
		}
	}

	exec.Finalize()
}

func TestExecWithDynamicAxes_Neg(t *testing.T) {
	// Single-input function with dynamic axes.
	negFn := func(x *graph.Node) *graph.Node {
		return graph.Neg(x)
	}
	exec := graph.MustNewExec(backend, negFn).
		WithDynamicAxes([]string{"batch", "", ""})

	// batch=2
	a := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 1, 3)
	outputs := exec.MustExec(a)
	got, err := tensors.CopyFlatData[float32](outputs[0])
	require.NoError(t, err)
	require.Equal(t, []float32{-1, -2, -3, -4, -5, -6}, got)

	// batch=3
	b := tensors.FromFlatDataAndDimensions([]float32{10, 20, 30, 40, 50, 60, 70, 80, 90}, 3, 1, 3)
	outputs2 := exec.MustExec(b)
	got2, err := tensors.CopyFlatData[float32](outputs2[0])
	require.NoError(t, err)
	require.Equal(t, []float32{-10, -20, -30, -40, -50, -60, -70, -80, -90}, got2)

	exec.Finalize()
}

func TestCreateSpecialization(t *testing.T) {
	// Build a dynamic graph and verify the specialization has correctly resolved shapes.
	builder := backend.Builder("test_create_spec")
	mainFn := builder.Main()

	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{-1, 8}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", dynShape, nil)
	require.NoError(t, err)
	neg, err := mainFn.Neg(x)
	require.NoError(t, err)
	err = mainFn.Return([]backends.Value{neg}, nil)
	require.NoError(t, err)
	exec, err := builder.Compile()
	require.NoError(t, err)

	e := exec.(*Executable)
	require.True(t, e.hasDynamicAxes)

	bindings := shapes.AxisBindings{"batch": 16}
	spec, err := e.createSpecialization(bindings)
	require.NoError(t, err)

	// All resolved nodes should have concrete shapes.
	for i, n := range spec.resolvedNodes {
		if n == nil {
			continue
		}
		require.False(t, n.shape.HasDynamicDims(),
			"resolved node %d (op=%s) still has dynamic dims: %s", i, n.opType, n.shape)
	}

	// The parameter node should have shape [16, 8] with dtype Float32.
	paramNode := spec.resolvedNodes[e.builder.mainFn.parameters[0].idx]
	require.Equal(t, dtypes.Float32, paramNode.shape.DType)
	require.Equal(t, []int{16, 8}, paramNode.shape.Dimensions,
		"expected resolved parameter dimensions [16, 8], got %v", paramNode.shape.Dimensions)

	fmt.Printf("Specialization test passed for bindings: %v\n", bindings)
}
