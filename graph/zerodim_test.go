package graph_test

import (
	"math"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

// TestZeroDim runs several tests to check that the backend correctly supports zero-dimensional
// tensors.
func TestZeroDim(t *testing.T) {
	graphtest.RunTestGraphFn(t, "UnaryOps", func(g *Graph) (inputs []*Node, outputs []*Node) {
		x := Iota(g, shapes.Make(dtypes.Float32, 3, 0, 3), -1)
		minusX := Neg(x)
		inputs, outputs = []*Node{x}, []*Node{minusX}
		return
	}, []any{
		shapes.Make(dtypes.Float32, 3, 0, 3),
	}, 0)

	graphtest.RunTestGraphFn(t, "BinaryOps", func(g *Graph) (inputs []*Node, outputs []*Node) {
		x := Iota(g, shapes.Make(dtypes.Float32, 3, 0, 3), -1)
		twoX := Add(x, x)
		inputs, outputs = []*Node{x}, []*Node{twoX}
		return
	}, []any{
		shapes.Make(dtypes.Float32, 3, 0, 3),
	}, 0)

	graphtest.RunTestGraphFn(t, "Reduce", func(g *Graph) (inputs []*Node, outputs []*Node) {
		x := Iota(g, shapes.Make(dtypes.Float32, 3, 0, 3), -1)
		sumDefault := ReduceAllSum(x)
		maxDefault := ReduceAllMax(x)
		reduceToZeroDim := ReduceSum(x, -1)
		require.NotPanics(t, func() {
			reduceToZeroDim.AssertDims(3, 0)
		})
		inputs, outputs = []*Node{x}, []*Node{sumDefault, maxDefault, reduceToZeroDim}
		return
	}, []any{
		float32(0),
		float32(math.Inf(-1)),
		shapes.Make(dtypes.Float32, 3, 0),
	}, 0)
}
