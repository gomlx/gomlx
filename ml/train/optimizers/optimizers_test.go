package optimizers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestMonotonicProjection(t *testing.T) {
	graphtest.RunTestGraphFn(t, "MonotonicProjection()", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{
			Const(g, []float32{0.0, 1.0, 1.0, 2.0}),
			Const(g, []float32{1.0, 1.0}),
			Const(g, []float32{1.0, 1.0, 1.0}),
			Const(g, [][]float32{{0.0}, {1.1}, {0.9}, {2.0}}),
		}
		margin := Scalar(g, inputs[0].DType(), 0.1)
		outputs = []*Node{
			MonotonicProjection(inputs[0], margin, -1),
			MonotonicProjection(inputs[1], margin, -1),
			MonotonicProjection(inputs[2], margin, -1),
			MonotonicProjection(inputs[3], margin, 0),
		}
		return
	}, []any{
		[]float32{0.0, 0.95, 1.05, 2.0},
		[]float32{0.95, 1.05},
		[]float32{0.9, 1, 1.1},
		[][]float32{{0.0}, {0.95}, {1.05}, {2.0}},
	}, 1e-4)

}
