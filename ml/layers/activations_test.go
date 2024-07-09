package layers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"testing"
)

func TestRelu(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Relu",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{Relu(x)}
			return
		}, []any{
			[]float32{0, 0, 2, 0, 4, 0, 6},
		}, xslices.Epsilon)
}

func TestLeakyReluWithAlpha(t *testing.T) {
	graphtest.RunTestGraphFn(t, "LeakyReluWithAlpha",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{LeakyReluWithAlpha(x, 0.1)}
			return
		}, []any{
			[]float32{0, -0.1, 2, -0.3, 4, -0.5, 6},
		}, xslices.Epsilon)
}

func TestSwish(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Swish",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{Swish(x)}
			return
		}, []any{
			[]float32{0, -0.26894143, 1.7615942, -0.14227763, 3.928055, -0.03346425, 5.9851646},
		}, xslices.Epsilon)
}

func TestSelu(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Swish",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{Selu(x)}
			return
		}, []any{
			[]float32{0., -1.1113307, 2.101402, -1.6705687, 4.202804, -1.7462534, 6.304206},
		}, xslices.Epsilon)
}
