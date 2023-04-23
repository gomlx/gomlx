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
	"github.com/stretchr/testify/assert"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"os"
	"testing"
)

func TestCheckpoints(t *testing.T) {
	manager := BuildManager().MustDone()

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
			result := e.Call()[0]
			assert.NoError(t, result.Error(), "Executing test graph")
			globalStep := tensor.ToScalar[float64](result.Local())
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
		result := e.Call()[0]
		assert.NoError(t, result.Error(), "Executing test graph")
		globalStep := tensor.ToScalar[float64](result.Local())
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
