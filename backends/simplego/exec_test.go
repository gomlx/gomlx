package simplego

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
)

func TestBuilder_Compile(t *testing.T) {
	// backend must be exclusive (not shared across tests) for this test to work.
	builder := backend.Builder("test")
	x, err := builder.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	require.NoError(t, err)
	require.NotNil(t, x)
	x, err = builder.Neg(x)
	require.NoError(t, err)
	require.NotNil(t, x)
	c, err := builder.Constant([]int64{1, 2, 3}, 3)
	require.NoError(t, err)
	require.NotNil(t, c)

	exec, err := builder.Compile([]backends.Op{x, c}, nil)
	require.NoError(t, err)
	require.NotNil(t, exec)

	// Check that it fails if fed the wrong number of parameters.
	i0, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	i1, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{i0, i1}, []bool{true, true}, 0)
	require.Error(t, err)

	// Check that it fails if fed incompatible parameters.
	i0, err = backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{i0}, []bool{true}, 0)
	require.Error(t, err)

	i0, err = backend.BufferFromFlatData(0, []uint32{1, 2, 3}, shapes.Make(dtypes.Uint32, 3))
	require.NoError(t, err)
	_, err = exec.Execute([]backends.Buffer{i0}, []bool{true}, 0)
	require.Error(t, err)

	// Checks correct execution with donated inputs, and that the output reused the input buffer.
	i0, err = backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	i0Data := i0.(*Buffer).flat.([]float32)
	outputs, err := exec.Execute([]backends.Buffer{i0}, []bool{true}, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)
	require.True(t, &i0Data[0] == &(outputs[0].(*Buffer).flat.([]float32))[0])
	outputShape, err := backend.BufferShape(outputs[1])
	require.NoError(t, err)
	require.True(t, outputShape.Equal(shapes.Make(dtypes.Int64, 3)))

	// Checks correct execution without donated inputs.
	// Notice the inputs were donated in the last iteration, so we have to set them again.
	i0, err = backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3))
	require.NoError(t, err)
	outputs, err = exec.Execute([]backends.Buffer{i0}, []bool{false}, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)
	require.True(t, i0.(*Buffer) != outputs[0].(*Buffer))
	outputShape, err = backend.BufferShape(outputs[1])
	require.NoError(t, err)
	require.True(t, outputShape.Equal(shapes.Make(dtypes.Int64, 3)))
}

func TestGomlxIntegration(t *testing.T) {
	// Makes sure we get a SimpleGo backend.
	backend, err := backends.NewWithConfig(BackendName)
	require.NoError(t, err)
	require.NotPanics(t, func() { _ = backend.(*Backend) })

	// Checks that basic graph building and execution works.
	y := graph.MustExecOnce(backend, graph.Neg, float32(7))
	fmt.Printf("\ty=-x: x=7, y=%s\n", y.GoStr())
	require.Equal(t, float32(-7), y.Value())

	ctx := context.New()
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *graph.Graph) *graph.Node {
		counterVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("counter", shapes.Make(dtypes.Int64))
		counter := counterVar.ValueGraph(g)
		counterVar.SetValueGraph(graph.OnePlus(counter))
		return counter
	})
	for ii := range 10 {
		got := exec.MustExec()[0]
		require.Equal(t, int64(ii), got.Value())
	}
}
