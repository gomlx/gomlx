package train

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
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
	learningRate := 0.1
	optimizer := optimizers.StochasticGradientDescent().
		WithDecay(false).WithLearningRate(learningRate).Done()
	trainer := NewTrainer(backend, ctx, modelFn, lossFn, optimizer, nil, nil)
	input := tensors.FromScalar(float32(0))
	label := tensors.FromScalar(float32(10))

	// Check that it accumulates for 3 steps:
	numSteps := 3
	err := trainer.AccumulateGradients(3)
	numTrainerMetrics := len(trainer.TrainMetrics())
	require.NoError(t, err)
	for ii := range numSteps {
		metrics, err := trainer.TrainStep(nil, []*tensors.Tensor{input}, []*tensors.Tensor{label})
		require.NoError(t, err)
		require.Len(t, metrics, numTrainerMetrics)

		// Since the gradient hasn't been applied yet, the loss should be 10.0.
		loss := metrics[0].Value().(float32)
		require.Equal(t, float32(10), loss)

		predictionVar := ctx.GetVariableByScopeAndName("/model", "prediction")
		require.NotNil(t, predictionVar)
		accPredictionVar := ctx.GetVariableByScopeAndName("/"+AccumulatedGradientsScope+"/model", "prediction")
		require.NotNil(t, accPredictionVar)

		prediction := predictionVar.MustValue().Value().(float32)
		accPrediction := accPredictionVar.MustValue().Value().(float32)
		fmt.Printf("\tIterator #%d: prediction=%g, accumulated gradient=%g\n", ii, prediction, accPrediction)

		if ii < numSteps-1 {
			// Gradients have not yet been applied:
			// - prediction is still 0.
			require.Equal(t, float32(0), prediction)
			// - gradient is always -1, and it has accumulated ii+1 times:
			require.Equal(t, float32(-(ii + 1)), accPrediction)
		} else {
			// Gradients have been applied, and accumulator is reset to 0.
			require.Equal(t, float32(0), accPrediction)
			// The mean gradient was -1, and we took one -learningRate in that direction.
			require.Equal(t, float32(learningRate), prediction)
		}
	}
}
