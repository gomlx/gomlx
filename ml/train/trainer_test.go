// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
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
