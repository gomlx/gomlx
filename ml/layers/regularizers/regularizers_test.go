package regularizers

import (
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/ctxtest"
	"github.com/gomlx/gomlx/ml/train"
	"testing"
)

func TestConstantL1(t *testing.T) {
	ctxtest.RunTestGraphFn(t, "ConstantL1 regularizer", func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
		wVar := ctx.VariableWithValue("w", [][]float32{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}})
		w := wVar.ValueGraph(g)
		ConstantL1(0.1)(ctx, g, wVar)
		loss := train.GetLosses(ctx, g)
		inputs = []*Node{w}
		outputs = []*Node{loss, Gradient(loss, w)[0]}
		return
	}, []any{
		float32(8 * 0.1), // Total regularization loss.
		[][]float32{{-0.1, 0, 0, 0, 0.1}, {-0.1, 0, 0, 0, 0.1}}, // Gradient.
	}, 1e-4)
}
