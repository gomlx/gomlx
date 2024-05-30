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

package optimizers

import (
	"math"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"google3/third_party/golang/testify/assert/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/xla/cpu"
)

func TestCosineAnnealingSchedule(t *testing.T) {
	manager := BuildManager().Platform("Host").Done()
	ctx := context.NewContext(manager).Checked(false)
	cosineExec := context.NewExec(manager, ctx, func(ctx *context.Context, graph *Graph) {
		CosineAnnealingSchedule(ctx, graph, shapes.Float32).
			PeriodInSteps(50).
			LearningRate(1.0).
			MinLearningRate(0.001).
			Done()
	})

	for ii := int64(0); ii < 100; ii++ {
		require.NotPanicsf(t, func() { _ = cosineExec.Call() }, "cosineExec.Call failed to execute graph for ii=%d", ii)

		// Checks correct step number.
		stepVar := ctx.InspectVariable("/optimizers/cosine", GlobalStepVariableName)
		if stepVar == nil {
			t.Fatalf("Learning rate variable not created in scope %q, name %q", "/optimizers/cosine", GlobalStepVariableName)
		}
		step := stepVar.Value().Value().(int64)
		assert.Equal(t, ii+1, step)

		// Check learning rate is following cosine formulation.
		lrVar := ctx.InspectVariable("/optimizers", LearningRateKey)
		if lrVar == nil {
			t.Fatalf("Learning rate variable not created in scope %q, name %q", "/optimiziers", LearningRateKey)
		}
		lr := lrVar.Value().Value().(float32)
		cycle := float64(ii) / 50.0
		wantLR := (math.Cos((cycle-math.Floor(cycle))*math.Pi) + 1.0) / 2.0
		wantLR = wantLR*(1.0-0.001) + 0.001
		assert.InDelta(t, float32(wantLR), lr, 0.001)

	}
}
