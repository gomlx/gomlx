package train

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
)

func TestTrainer_AccumulateGradients(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()
		predictionVar := ctx.In("model").
			VariableWithValue("prediction", float32(0))
		return []*Node{predictionVar.ValueGraph(g)}
	}
	lossFn := losses.MeanAbsoluteError
	optimizer := optimizers.StochasticGradientDescent

	trainer := NewTrainer(backend, ctx, modelFn, lossFn, initializers.Constant(0.0))
}
