package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestBuilder_Compile(t *testing.T) {
	// backend must be exclusive (not shared across tests) for this test to work.
	backend := New("")
	builder := backend.Builder("test")
	var x, c backends.Op
	require.NotPanics(t, func() {
		x = builder.Parameter("x", shapes.Make(dtypes.Float32, 3))
		x = builder.Neg(x)
		c = builder.Constant([]int64{1, 2, 3}, 3)
	})
	require.NotNil(t, x)
	require.NotNil(t, c)
	var exec *Executable
	require.NotPanics(t, func() { exec = builder.Compile(x, c).(*Executable) })
	require.NotNil(t, exec)

	// Check that it fails if fed the wrong number of parameters.
	require.Panics(t, func() {
		inputs := []backends.Buffer{
			backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3)),
			backend.BufferFromFlatData(0, []int64{1, 2, 3}, shapes.Make(dtypes.Float32, 3)),
		}
		exec.Execute(inputs, []bool{true, true})
	})

	// Check that it fails if fed incompatible parameters.
	require.Panics(t, func() {
		inputs := []backends.Buffer{
			backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4)),
		}
		exec.Execute(inputs, []bool{true})
	})
	require.Panics(t, func() {
		inputs := []backends.Buffer{
			backend.BufferFromFlatData(0, []int64{1, 2, 3}, shapes.Make(dtypes.Float32, 3)),
		}
		exec.Execute(inputs, []bool{true})
	})

	// Checks correct execution with donated inputs.
	inputs := []backends.Buffer{
		backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3)),
	}
	inputsData := inputs[0].(*Buffer).flat.([]float32)
	outputs := exec.Execute(inputs, []bool{true})
	require.Len(t, outputs, 2)
	require.True(t, &inputsData[0] == &(outputs[0].(*Buffer).flat.([]float32))[0])
	require.True(t, backend.BufferShape(outputs[1]).Equal(shapes.Make(dtypes.Int64, 3)))

	// Save reference to buffer before finalizing it: it should be re-used on the next call.
	oldOutput1 := outputs[1].(*Buffer)
	backend.BufferFinalize(outputs[1]) // Return buffer: we want it re-used at the next call.

	// Checks correct execution without donated inputs.
	// Notice the inputs were donated in the last interation, so we have to set them again.
	inputs = []backends.Buffer{
		backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3)),
	}
	outputs = exec.Execute(inputs, []bool{false})
	require.Len(t, outputs, 2)
	require.True(t, inputs[0].(*Buffer) != outputs[0].(*Buffer))
	require.True(t, backend.BufferShape(outputs[1]).Equal(shapes.Make(dtypes.Int64, 3)))

	// Checks that the output buffer for output1 was reused from the pool.
	newOutput1 := outputs[1].(*Buffer)
	require.True(t, oldOutput1 == newOutput1)
}

func TestGomlxIntegration(t *testing.T) {
	// Makes sure we get a SimpleGo backend.
	backend := backends.NewWithConfig(BackendName)
	require.NotPanics(t, func() { _ = backend.(*Backend) })

	// Checks that basic graph building and execution works.
	y := graph.ExecOnce(backend, func(x *graph.Node) *graph.Node { return graph.Neg(x) }, float32(7))
	fmt.Printf("\ty=-x: x=7, y=%s\n", y.GoStr())
	require.Equal(t, float32(-7), y.Value())
}
