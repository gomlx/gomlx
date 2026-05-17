// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package cosineschedule_test

import (
	"math"
	"path"
	"testing"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestCosineAnnealingSchedule(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const periodInSteps = 100
	const minLearningRate = 0.001
	const baseLearningRate = 1.0

	t.Run("periodSteps", func(t *testing.T) {
		store := model.NewStore()
		cosineExec, err := model.NewExec(backend, store, func(scope *model.Scope, graph *Graph) *Node {
			scope.SetTraining(graph, true)
			cosineschedule.New(scope, graph, dtypes.Float32).
				PeriodSteps(periodInSteps).
				LearningRate(baseLearningRate).
				MinLearningRate(minLearningRate).
				Done()
			return optimizers.LearningRateVar(scope.Store().Scope(scope.Scope()), dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range 2 * periodInSteps {
			lrT, err := cosineExec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			// Checks the correct step number is set.
			stepVar := store.GetVariable(
				path.Join("/", optimizers.Scope, cosineschedule.Scope, optimizers.GlobalStepVariableName),
			)
			if stepVar == nil {
				t.Fatalf(
					"Learning rate variable not created in scope %q, name %q",
					"/optimizers/cosine",
					optimizers.GlobalStepVariableName,
				)
			}
			step := stepVar.MustValue().Value().(int64)
			assert.Equal(t, int64(ii+1), step)

			// Check learning rate is following cosine formulation.
			lr := tensors.ToScalar[float32](lrT)
			cycle := float64(ii) / float64(periodInSteps)
			wantLR := (math.Cos((cycle-math.Floor(cycle))*math.Pi) + 1.0) / 2.0
			wantLR = wantLR*(baseLearningRate-minLearningRate) + minLearningRate
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "step=%d", ii)
		}
	})

	t.Run("periodSteps with warmUp", func(t *testing.T) {
		store := model.NewStore()
		const warmUpSteps = 10
		cosineExec, err := model.NewExec(backend, store, func(scope *model.Scope, graph *Graph) *Node {
			scope.SetTraining(graph, true)
			cosineschedule.New(scope, graph, dtypes.Float32).
				PeriodSteps(periodInSteps).
				LearningRate(baseLearningRate).
				MinLearningRate(minLearningRate).
				WarmUpSteps(warmUpSteps).
				Done()
			return optimizers.LearningRateVar(scope.Store().Scope(scope.Scope()), dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)
		for ii := range 2*periodInSteps + warmUpSteps {
			lrT, err := cosineExec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			// Check learning rate is following cosine formulation.
			lr := tensors.ToScalar[float32](lrT)
			var ratio float64
			if ii < warmUpSteps {
				ratio = float64(ii) / float64(warmUpSteps)
			} else {
				cycle := float64(ii-warmUpSteps) / float64(periodInSteps)
				ratio = (math.Cos((cycle-math.Floor(cycle))*math.Pi) + 1.0) / 2.0
			}
			wantLR := ratio*(baseLearningRate-minLearningRate) + minLearningRate
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "wantLR=%g, lr=%g, step=%d", wantLR, lr, ii)
		}
	})

	t.Run("numCycles with warmUp+context configuration", func(t *testing.T) {
		store := model.NewStore()
		const warmUpSteps = 10
		const numCycles = 2
		const stepsPerCycle = 100
		const numSteps = numCycles*stepsPerCycle + warmUpSteps
		store.SetParam("/", optimizers.ParamLearningRate, baseLearningRate)
		store.SetParam("/", cosineschedule.ParamCycles, numCycles)
		store.SetParam("/", cosineschedule.ParamWarmUpSteps, warmUpSteps)
		store.SetParam("/", cosineschedule.ParamMinLearningRate, minLearningRate)
		lastStepVar := train.GetTrainLastStepVar(store.RootScope())
		lastStepVar.MustSetValue(tensors.FromScalar(int64(numSteps)))
		cosineExec, err := model.NewExec(backend, store, func(scope *model.Scope, graph *Graph) *Node {
			scope.SetTraining(graph, true)
			cosineschedule.New(scope, graph, dtypes.Float32).FromContext().Done()
			return optimizers.LearningRateVar(scope.Store().Scope(scope.Scope()), dtypes.Float32, 1e3).ValueGraph(graph)
		})

		require.NoError(t, err)
		for ii := range numSteps {
			lrT, err := cosineExec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			// Check learning rate is following cosine formulation.
			lr := tensors.ToScalar[float32](lrT)
			var ratio float64
			if ii < warmUpSteps {
				ratio = float64(ii) / float64(warmUpSteps)
			} else {
				cycle := float64(ii-warmUpSteps) / float64(stepsPerCycle)
				ratio = (math.Cos((cycle-math.Floor(cycle))*math.Pi) + 1.0) / 2.0
			}
			wantLR := ratio*(baseLearningRate-minLearningRate) + minLearningRate
			require.InDeltaf(t, float32(wantLR), lr, 1e-4, "wantLR=%g, lr=%g, step=%d", wantLR, lr, ii)
		}
	})
}
