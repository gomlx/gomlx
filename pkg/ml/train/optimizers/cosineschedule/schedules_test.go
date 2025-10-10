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

package cosineschedule

import (
	"fmt"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestCosineAnnealingSchedule(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	periodInSteps := 100
	ctx := context.New().Checked(false)
	cosineExec := context.MustNewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
		ctx.SetTraining(graph, true)
		New(ctx, graph, dtypes.Float32).
			PeriodInSteps(periodInSteps).
			LearningRate(1.0).
			MinLearningRate(0.001).
			Done()
		return optimizers.LearningRateVar(ctx, dtypes.Float32, 0.001).ValueGraph(graph)
	})

	for ii := 0; ii < 2*periodInSteps; ii++ {
		require.NotPanicsf(t, func() { _ = cosineExec.Call() }, "cosineExec.Call failed to execute graph for ii=%d", ii)

		// Checks correct step number.
		stepVar := ctx.GetVariableByScopeAndName(fmt.Sprintf("/%s/%s", optimizers.Scope, Scope), optimizers.GlobalStepVariableName)
		if stepVar == nil {
			t.Fatalf("Learning rate variable not created in scope %q, name %q", "/optimizers/cosine", optimizers.GlobalStepVariableName)
		}
		step := stepVar.Value().Value().(int64)
		assert.Equal(t, int64(ii+1), step)

		// Check learning rate is following cosine formulation.
		lrVar := ctx.GetVariableByScopeAndName("/optimizers", optimizers.ParamLearningRate)
		if lrVar == nil {
			t.Fatalf("Learning rate variable not created in scope %q, name %q", "/optimiziers", optimizers.ParamLearningRate)
		}
		lr := lrVar.Value().Value().(float32)
		cycle := float64(ii) / float64(periodInSteps)
		wantLR := (math.Cos((cycle-math.Floor(cycle))*math.Pi) + 1.0) / 2.0
		wantLR = wantLR*(1.0-0.001) + 0.001
		assert.InDelta(t, float32(wantLR), lr, 0.001)
		fmt.Printf("Step %d: %f\n", ii, wantLR)
	}

	cosineExec = context.MustNewExec(backend, ctx, func(ctx *context.Context, graph *Graph) *Node {
		ctx.SetTraining(graph, false)
		New(ctx, graph, dtypes.Float32).
			PeriodInSteps(50).
			LearningRate(1.0).
			MinLearningRate(0.001).
			Done()
		return optimizers.LearningRateVar(ctx, dtypes.Float32, 0.001).ValueGraph(graph)
	})
	require.NotPanics(t, func() { _ = cosineExec.Call() }, "cosineExec.Call failed to execute graph when ctx.IsTraining() == false")
}
