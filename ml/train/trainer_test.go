// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"testing"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStoreErrors(t *testing.T) {
	backend := testutil.BuildTestBackend()
	g := graph.NewGraph(backend, "test")

	// GetTrainLastStepVar should panic when graph has no associated store.
	assert.Panics(t, func() {
		GetTrainLastStepVar(g)
	})

	// ExecPerStepUpdateGraphFn should panic when graph has no associated store.
	assert.Panics(t, func() {
		ExecPerStepUpdateGraphFn(g)
	})
}

func TestTrainer_WithMainLossMetric(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()

	modelFn := func(scope *model.Scope, spec any, inputs []*graph.Node) []*graph.Node {
		g := inputs[0].Graph()
		predictionVar := scope.In("model").
			VariableWithValue("prediction", float32(0))
		return []*graph.Node{predictionVar.NodeValue(g)}
	}
	lossFn := loss.MeanAbsoluteError
	opt := optimizer.StochasticGradientDescent().
		WithDecay(false).WithLearningRate(0.1).Done()

	trainer := NewTrainer(backend, store, modelFn, lossFn, opt, nil, nil)

	// By default, only two training metrics and one eval metric.
	assert.Len(t, trainer.TrainMetrics(), 2)
	assert.Equal(t, "Loss", trainer.TrainMetrics()[0].Name())
	assert.Equal(t, "Moving Average Loss", trainer.TrainMetrics()[1].Name())

	assert.Len(t, trainer.EvalMetrics(), 1)
	assert.Equal(t, "Mean Loss", trainer.EvalMetrics()[0].Name())

	// Call WithMainLossMetric:
	trainer.WithMainLossMetric()

	// Now there should be three training metrics and two eval metrics.
	assert.Len(t, trainer.TrainMetrics(), 3)
	assert.Equal(t, "Loss", trainer.TrainMetrics()[0].Name())
	assert.Equal(t, "Moving Average Loss", trainer.TrainMetrics()[1].Name())
	assert.Equal(t, "Moving Average Loss (no-reg)", trainer.TrainMetrics()[2].Name())

	assert.Len(t, trainer.EvalMetrics(), 2)
	assert.Equal(t, "Mean Loss", trainer.EvalMetrics()[0].Name())
	assert.Equal(t, "Mean Loss (no-reg)", trainer.EvalMetrics()[1].Name())
}

func TestTrainer_DynamicShapes(t *testing.T) {
	t.Run("TrainerDynamicShapes", func(t *testing.T) {
		// Use simple Go backend
		simpleBackend, err := gobackend.New("")
		require.NoError(t, err)

		store := model.NewStore()
		store.SetParam(model.ParamInitialSeed, int64(42))

		// Input is x: [batch, 3]
		// Label is y: [batch, 1]
		// Output is prediction: [batch, 1]
		modelFn := func(scope *model.Scope, spec any, inputs []*graph.Node) []*graph.Node {
			x := inputs[0]
			g := x.Graph()
			inputDim := x.Shape().Dimensions[x.Shape().Rank()-1]
			outputDim := 1
			w := scope.VariableWithShape("weights", shapes.Make(x.DType(), inputDim, outputDim)).NodeValue(g)
			b := scope.VariableWithShape("biases", shapes.Make(x.DType(), outputDim)).NodeValue(g)
			y := graph.MatMul(x, w)
			return []*graph.Node{graph.Add(y, graph.ExpandLeftToRank(b, y.Rank()))}
		}

		lossFn := loss.MeanAbsoluteError
		opt := optimizer.StochasticGradientDescent().
			WithDecay(false).WithLearningRate(0.1).Done()

		trainer := NewTrainer(simpleBackend, store, modelFn, lossFn, opt, nil, nil).
			WithInputsDynamicAxes([]string{"batch", ""}).
			WithLabelsDynamicAxes([]string{"batch", ""})

		// Run with batch = 2
		x2Tensor := tensors.MustFromAnyValue([][]float32{{1, 2, 3}, {4, 5, 6}})
		y2Tensor := tensors.MustFromAnyValue([][]float32{{2}, {5}})
		metrics2, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{x2Tensor}, Labels: []*tensors.Tensor{y2Tensor}})
		require.NoError(t, err)
		require.NotEmpty(t, metrics2)

		// Run with batch = 3 (should not recompile, should just execute and update weights)
		x3Tensor := tensors.MustFromAnyValue([][]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}})
		y3Tensor := tensors.MustFromAnyValue([][]float32{{1}, {2}, {3}})
		metrics3, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{x3Tensor}, Labels: []*tensors.Tensor{y3Tensor}})
		require.NoError(t, err)
		require.NotEmpty(t, metrics3)
	})
}

func TestTrainerAliases(t *testing.T) {
	backend := testutil.BuildTestBackend()

	// 1. func(scope *model.Scope, x *graph.Node) *graph.Node
	{
		store := model.NewStore()
		modelFn := func(scope *model.Scope, x *graph.Node) *graph.Node {
			return graph.AddScalar(x, 1)
		}
		trainer := NewTrainer(backend, store, modelFn, loss.MeanAbsoluteError, optimizer.StochasticGradientDescent().Done(), nil, nil)
		xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		yTensor := tensors.MustFromAnyValue([]float32{2, 3, 4})
		metrics, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{xTensor}, Labels: []*tensors.Tensor{yTensor}})
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	}

	// 2. func(scope *model.Scope, spec any, x *graph.Node) *graph.Node
	{
		store := model.NewStore()
		modelFn := func(scope *model.Scope, spec any, x *graph.Node) *graph.Node {
			scale := spec.(float32)
			return graph.MulScalar(x, scale)
		}
		trainer := NewTrainer(backend, store, modelFn, loss.MeanAbsoluteError, optimizer.StochasticGradientDescent().Done(), nil, nil)
		xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		yTensor := tensors.MustFromAnyValue([]float32{2, 4, 6})
		metrics, err := trainer.TrainStep(Batch{Spec: float32(2.0), Inputs: []*tensors.Tensor{xTensor}, Labels: []*tensors.Tensor{yTensor}})
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	}

	// 3. func(scope *model.Scope, spec any, inputs []*graph.Node) *graph.Node
	{
		store := model.NewStore()
		modelFn := func(scope *model.Scope, spec any, inputs []*graph.Node) *graph.Node {
			return graph.Add(inputs[0], inputs[1])
		}
		trainer := NewTrainer(backend, store, modelFn, loss.MeanAbsoluteError, optimizer.StochasticGradientDescent().Done(), nil, nil)
		x1Tensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		x2Tensor := tensors.MustFromAnyValue([]float32{3, 2, 1})
		yTensor := tensors.MustFromAnyValue([]float32{4, 4, 4})
		metrics, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{x1Tensor, x2Tensor}, Labels: []*tensors.Tensor{yTensor}})
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	}

	// 4. func(scope *model.Scope, x *graph.Node) (*graph.Node, *graph.Node)
	{
		store := model.NewStore()
		modelFn := func(scope *model.Scope, x *graph.Node) (*graph.Node, *graph.Node) {
			return x, x
		}
		customLossFn := func(labels, predictions []*graph.Node) *graph.Node {
			return graph.Add(loss.MeanAbsoluteError(labels[:1], predictions[:1]), loss.MeanAbsoluteError(labels[1:], predictions[1:]))
		}
		trainer := NewTrainer(backend, store, modelFn, customLossFn, optimizer.StochasticGradientDescent().Done(), nil, nil)
		xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		y1Tensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		y2Tensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		metrics, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{xTensor}, Labels: []*tensors.Tensor{y1Tensor, y2Tensor}})
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	}

	// 5. func(scope *model.Scope, x *graph.Node) []*graph.Node
	{
		store := model.NewStore()
		modelFn := func(scope *model.Scope, x *graph.Node) []*graph.Node {
			return []*graph.Node{x}
		}
		trainer := NewTrainer(backend, store, modelFn, loss.MeanAbsoluteError, optimizer.StochasticGradientDescent().Done(), nil, nil)
		xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		yTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
		metrics, err := trainer.TrainStep(Batch{Inputs: []*tensors.Tensor{xTensor}, Labels: []*tensors.Tensor{yTensor}})
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	}
}

func TestBatchClone(t *testing.T) {
	xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
	yTensor := tensors.MustFromAnyValue([]float32{4, 5, 6})
	batch := Batch{
		Inputs: []*tensors.Tensor{xTensor},
		Labels: []*tensors.Tensor{yTensor},
		Spec:   "test-spec",
	}
	cloned, err := batch.Clone()
	require.NoError(t, err)
	assert.Equal(t, batch.Spec, cloned.Spec)
	require.Len(t, cloned.Inputs, 1)
	require.Len(t, cloned.Labels, 1)
	assert.Equal(t, batch.Inputs[0].Shape(), cloned.Inputs[0].Shape())
	assert.Equal(t, batch.Labels[0].Shape(), cloned.Labels[0].Shape())

	// Clean up
	require.NoError(t, batch.Finalize())
	require.NoError(t, cloned.Finalize())
}

func TestBatchOnDeviceClone(t *testing.T) {
	backend := testutil.BuildTestBackend()
	xTensor := tensors.MustFromAnyValue([]float32{1, 2, 3})
	yTensor := tensors.MustFromAnyValue([]float32{4, 5, 6})
	batch := Batch{
		Inputs: []*tensors.Tensor{xTensor},
		Labels: []*tensors.Tensor{yTensor},
		Spec:   "test-spec",
	}
	cloned, err := batch.OnDeviceClone(backend, 0)
	require.NoError(t, err)
	assert.Equal(t, batch.Spec, cloned.Spec)
	require.Len(t, cloned.Inputs, 1)
	require.Len(t, cloned.Labels, 1)
	assert.Equal(t, batch.Inputs[0].Shape(), cloned.Inputs[0].Shape())
	assert.Equal(t, batch.Labels[0].Shape(), cloned.Labels[0].Shape())

	// Clean up
	require.NoError(t, batch.Finalize())
	require.NoError(t, cloned.Finalize())
}
