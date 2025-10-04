package activations

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/support/xslices"

	_ "github.com/gomlx/gomlx/backends/default"
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

func TestGelu(t *testing.T) {
	// Values generated using jax.nn.Gelu(approximate=False) on GPU (on cpus it varies a bit).
	graphtest.RunTestGraphFn(t, "Gelu (Exact)",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{Gelu(x)}
			return
		}, []any{
			[]float32{0, -0.15865526, 1.9544997, -4.0496886e-03, 3.9998736, -1.3411045e-06, 6},
		}, 1e-5)

	// Values generated using jax.nn.Gelu(approximate=True) on GPU (on cpus it varies a bit).
	graphtest.RunTestGraphFn(t, "GeluApproximate",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{0, -1, 2, -3, 4, -5, 6})
			inputs = []*Node{x}
			outputs = []*Node{GeluApproximate(x)}
			return
		}, []any{
			[]float32{0, -0.15880796, 1.9545977, -3.6375225e-03, 3.9999294, 0, 6},
		}, 0.01)
}
