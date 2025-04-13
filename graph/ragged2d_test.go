package graph_test

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestRagged2D(t *testing.T) {
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	graphtest.RunTestGraphFn(t, "Ragged2D", func(g *Graph) (inputs, outputs []*Node) {
		r := MakeRagged2D(5, Const(g, []float32{1, 3, 5, 7, 11, 13}), Const(g, []int32{0, 0, 0, 1, 1, 3}))
		inputs = []*Node{r.Flat, r.RowIDs}
		require.Equal(t, dtypes.F32, r.DType())
		require.Equal(t, g, r.Graph())
		outputs = []*Node{
			r.ReduceSumCols(),
			r.ReduceMaxCols(),
			r.ReduceMinCols(),
		}
		return
	}, []any{
		[]float32{1 + 3 + 5, 7 + 11, 0, 13, 0},
		[]float32{5, 11, negInf, 13, negInf},
		[]float32{1, 7, posInf, 13, posInf},
	}, 0)
}

func TestRagged2D_ReduceSumCols(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	// The inputs as fed as parameters, which prevents constant subexpression optimization.
	// This was used to point out a difference in CPU/GPU versions.
	output := ExecOnce(backend, func(flat, rowIDs *Node) *Node {
		r := MakeRagged2D(3, flat, rowIDs)
		output := r.ReduceSumCols()
		output.SetLogged("#full Ragged2D.ReduceSumCols")
		return output
	},
		[]float32{1, 3, 5, 7, 11, 13, 17, 19},
		[]int32{0, 0, 0, 0, 0, 1, 1, 1},
	)
	require.Equal(t, []float32{1 + 3 + 5 + 7 + 11, 13 + 17 + 19, 0}, output.Value().([]float32))
}

func TestRagged2DSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Ragged2DSoftmax", func(g *Graph) (inputs, outputs []*Node) {
		r := MakeRagged2D(5, Const(g, []float32{1, 3, 5, 7, 11, 13}), Const(g, []int32{0, 0, 0, 1, 1, 3}))
		inputs = []*Node{r.Flat, r.RowIDs}
		result := r.Softmax()
		outputs = []*Node{result.Flat}
		return
	}, []any{
		[]float32{0.015876241, 0.11731043, 0.86681336, 0.01798621, 0.98201376, 1},
	}, 1e-3)
}
