package train

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
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
	optimizer := optimizers.StochasticGradientDescent().WithDecay(false).WithLearningRate(0.1).Done()
	trainer := NewTrainer(backend, ctx, modelFn, lossFn, optimizer, nil, nil)
	input := tensors.FromScalar(float32(0))
	label := tensors.FromScalar(float32(10))

	// Check that it accumulates for 3 steps:
	numSteps := 3
	err := trainer.AccumulateGradients(3)
	numTrainerMetrics := len(trainer.TrainMetrics())
	require.NoError(t, err)
	for ii := range numSteps {
		metrics := trainer.TrainStep(nil, []*tensors.Tensor{input}, []*tensors.Tensor{label})
		require.Len(t, metrics, numTrainerMetrics)

		// Since the gradient hasn't been applied yet, the loss should be 10.0.
		loss := metrics[0].Value().(float32)
		require.Equal(t, float32(10), loss)

		_ = ii
	}
}
