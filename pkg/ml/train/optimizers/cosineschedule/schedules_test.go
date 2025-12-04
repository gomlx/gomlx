/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package cosineschedule_test

import (
	"fmt"
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestCosineAnnealingSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const periodInSteps = 100
	const minLearningRate = 0.001
	const baseLearningRate = 1.0

	t.Run("periodSteps", func(t *testing.T) {
		ctx := context.New().Checked(false)
		cosineExec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			cosineschedule.New(ctx, graph, dtypes.Float32).
				PeriodSteps(periodInSteps).
				LearningRate(baseLearningRate).
				MinLearningRate(minLearningRate).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
		})
		require.NoError(t, err)

		for ii := range 2 * periodInSteps {
			lrT, err := cosineExec.Exec1()
			require.NoErrorf(t, err, "failed for step %d", ii)

			// Checks the correct step number is set.
			stepVar := ctx.GetVariableByScopeAndName(
				fmt.Sprintf("/%s/%s", optimizers.Scope, cosineschedule.Scope),
				optimizers.GlobalStepVariableName,
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
		ctx := context.New().Checked(false)
		const warmUpSteps = 10
		cosineExec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			cosineschedule.New(ctx, graph, dtypes.Float32).
				PeriodSteps(periodInSteps).
				LearningRate(baseLearningRate).
				MinLearningRate(minLearningRate).
				WarmUpSteps(warmUpSteps).
				Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
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
		ctx := context.New().Checked(false)
		const warmUpSteps = 10
		const numCycles = 2
		const stepsPerCycle = 100
		const numSteps = numCycles*stepsPerCycle + warmUpSteps
		ctx.SetParam(optimizers.ParamLearningRate, baseLearningRate)
		ctx.SetParam(cosineschedule.ParamCycles, numCycles)
		ctx.SetParam(cosineschedule.ParamWarmUpSteps, warmUpSteps)
		ctx.SetParam(cosineschedule.ParamMinLearningRate, minLearningRate)
		lastStepVar := train.GetTrainLastStepVar(ctx)
		lastStepVar.MustSetValue(tensors.FromScalar(int64(numSteps)))
		cosineExec, err := context.NewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
			ctx.SetTraining(graph, true)
			cosineschedule.New(ctx, graph, dtypes.Float32).FromContext().Done()
			return optimizers.LearningRateVar(ctx, dtypes.Float32, 1e3).ValueGraph(graph)
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
