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

package checkpoints

// TODO:
// * Test that previously loaded variables -- not used by the Context -- are also saved.
// * Test what happens with saving/loading of objects in Params: do they need to be filtered?

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestCheckpoints(t *testing.T) {
	manager := graphtest.BuildTestManager()

	// Graph function to test: it simply creates, increments and returns the global step.
	testGraphFn := func(ctx *context.Context, g *Graph) *Node {
		return optimizers.IncrementGlobalStepGraph(ctx, g, shapes.Float64)
	}
	var dir string
	{
		// Build model, checkpoint a few times.
		ctx := context.NewContext(manager)
		ctx.SetParam("learning_rate", 0.01)
		ctx.SetParam(layers.L2RegularizationKey, 0.001)
		ctx.In("layer_1").SetParam(layers.L2RegularizationKey, 0.004)
		checkpoint := Build(ctx).TempDir("", "test_checkpoints_").Keep(3).MustDone()
		dir = checkpoint.Dir()
		fmt.Printf("Checkpoint directory: %s\n", dir)
		e := context.NewExec(manager, ctx, testGraphFn)
		for ii := 0; ii < 10; ii++ {
			results := e.Call()
			globalStep := results[0].Local().Value().(float64)
			assert.Equal(t, float64(ii)+1, globalStep, "LoopStep")
			assert.NoError(t, checkpoint.Save(), "Saving checkpoint")
		}

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpoint.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
	}

	// Test loading of values
	{
		// Build model, checkpoint a few times.
		ctx := context.NewContext(manager)
		ctx.SetParam("learning_rate", 5.0) // Value should be overwritten when loading.
		checkpoint := Build(ctx).Dir(dir).Keep(3).MustDone()

		lr, found := ctx.GetParam("learning_rate")
		assert.True(t, found, "learning_rate should be set")
		assert.Equal(t, 0.01, lr.(float64), "Params[learning_rate]")

		var l2 any
		l2, found = ctx.GetParam(layers.L2RegularizationKey)
		assert.Truef(t, found, "%s should have been set", layers.L2RegularizationKey)
		assert.Equal(t, 0.001, l2.(float64), "(Scope=%s) Params[%s]", ctx.Scope(), layers.L2RegularizationKey)
		l2, found = ctx.In("layer_1").GetParam(layers.L2RegularizationKey)
		assert.Truef(t, found, "%s should have been set", layers.L2RegularizationKey)
		assert.Equal(t, 0.004, l2.(float64), "Params[%s]", layers.L2RegularizationKey)

		// Re-execute testGraphFn: it should load global step at 10, increment and return it at 11.
		e := context.NewExec(manager, ctx, testGraphFn)
		results := e.Call()
		globalStep := results[0].Local().Value().(float64)
		assert.Equal(t, 11.0, globalStep, "Re-loaded global step")
		assert.NoError(t, checkpoint.Save(), "Saving checkpoint")

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpoint.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
	}

	// Remove test directory.
	assert.NoErrorf(t, os.RemoveAll(dir), "Removing directory used for testing %q", dir)
}

func TestMergedCheckpoints(t *testing.T) {
	manager := graphtest.BuildTestManager()
	var dir string
	{
		ctx := context.NewContext(manager).Checked(false)
		checkpoint := Build(ctx).TempDir("", "test_checkpoints_").Keep(2).MustDone()
		dir = checkpoint.Dir()
		globalStepV := optimizers.GetGlobalStepVar(ctx)
		globalStepV.SetValue(tensor.FromValue(1))
		xV := ctx.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		yV := ctx.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
		require.NoError(t, checkpoint.Save())

		globalStepV.SetValue(tensor.FromValue(10))
		xV.SetValue(tensor.FromValue([]float64{3.0, 3.0, 3.0}))
		yV.SetValue(tensor.FromValue([][]float32{{6.0}, {6.0}}))
		require.NoError(t, checkpoint.Save())
	}
	{
		// Check that the values were averaged:
		ctx := context.NewContext(manager).Checked(false)
		_ = Build(ctx).Dir(dir).Keep(2).TakeMean(-1).MustDone()
		globalStepV := optimizers.GetGlobalStepVar(ctx)
		assert.Equal(t, 10, globalStepV.Value().Value(), "GlobalStep")
		xV := ctx.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		// Assume X will be loaded with the mean of the previous 2 checkpoints:
		assert.Equal(t, []float64{2.0, 2.0, 2.0}, xV.Value().Value(), "X")
		yV := ctx.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
		assert.Equal(t, [][]float32{{5.0}, {5.0}}, yV.Value().Value(), "Y")
	}
}
