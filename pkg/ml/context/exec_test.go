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

package context_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func oneLayerGraph(ctx *context.Context, x *Node) *Node {
	return layers.DenseWithBias(ctx.In("dense0"), x, 1)
}

func oneLayerManyInputsGraph(ctx *context.Context, inputs []*Node) *Node {
	return layers.DenseWithBias(ctx.In("dense0"), Concatenate(inputs, -1), 1)
}

func TestExec(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	t.Run("DenseLayer", func(t *testing.T) {
		oneLayer, err := context.NewExecAny(backend, nil, oneLayerGraph)
		if err != nil {
			t.Fatalf("Failed to create context.Exec for oneLayer: %+v", err)
		}

		// First graph build, variable should be initialized.
		x := [][]float64{{1, 1, 1}}
		got := oneLayer.MustExec(x)[0].Value().([][]float64)[0][0]
		assert.NotEqual(
			t,
			0.0,
			got,
			"Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized",
			x,
		)

		// The second call should reuse the graph and yield the same result.
		got2 := oneLayer.MustExec(x)[0].Value().([][]float64)[0][0]
		assert.InDeltaf(
			t,
			got,
			got2,
			1e-3,
			"Second call to oneLayer(%v) returned %f, but first returned %f",
			x,
			got,
			got2,
		)

		// Different batch size: it should generate a new graph but yield the same result.
		x = [][]float64{{1, 1, 1}, {2, 2, 2}}
		got3 := oneLayer.MustExec(x)[0].Value().([][]float64)[0][0]
		assert.InDeltaf(t, got, got3, 1e-3,
			"Exec to oneLayer(%v) on a larger batch returned %f, but original call returned %f",
			x, got, got3)

		// If X changes the inner dimension, then the layers.DenseWithBias would require different shaped variables, which should
		// fail to build.
		x = [][]float64{{1, 1, 1, 1}}
		require.Panics(t, func() { _ = oneLayer.MustExec(x) },
			"oneLayer(%v) leads to a graph with differently shaped variables should have failed", x)
	})

	t.Run("WithSlices", func(t *testing.T) {
		oneLayer, err := context.NewExecAny(backend, nil, oneLayerManyInputsGraph)
		if err != nil {
			t.Fatalf("Failed to create context.Exec for oneLayer: %+v", err)
		}

		// First execution builds graph and should initialize variables (weights) randomly.
		x := [][]float64{{1, 1, 1}}
		got := oneLayer.MustExec(x)[0].Value().([][]float64)
		fmt.Printf("\toneLayer(%v)=%v\n", x, got)
		if got[0][0] == 0 {
			t.Fatalf("Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized", x)
		}

		// Second execution: two tensors that if concatenated will be the same as the previous run. Since the context
		// and hence the variables are the same, it should evaluate to exact same value.
		oneElementOfX := [][]float64{{1}}
		got2 := oneLayer.MustExec(oneElementOfX, oneElementOfX, oneElementOfX)[0].Value().([][]float64)
		fmt.Printf("\toneLayer(%v, %v, %v)=%v\n", oneElementOfX, oneElementOfX, oneElementOfX, got)
		require.Truef(t, math.Abs(got[0][0]-got2[0][0]) < 1.0e-3,
			"oneLayer(%v) should have been the same as oneLayer(%v, %v, %v), but got %v and %v",
			x, oneElementOfX, oneElementOfX, oneElementOfX, got, got2)
	})

	t.Run("VariableUpdates", func(t *testing.T) {
		ctx := context.New()
		counter := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			dtype := dtypes.Int64
			counterVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("counter", shapes.Make(dtype))
			counterNode := counterVar.ValueGraph(g)
			counterNode = Add(counterNode, OnesLike(counterNode))
			counterVar.SetValueGraph(counterNode)
			return counterNode
		})
		results := counter.MustExec()
		gotTensor := results[0]
		got := gotTensor.Value().(int64)
		if got != 1 {
			t.Fatalf("Wanted first counter value to be 1, got %d instead", got)
		}
		results = counter.MustExec()
		gotTensor = results[0]
		got = gotTensor.Value().(int64)
		if got != 2 {
			t.Fatalf("Wanted second counter value to be 2, got %d instead", got)
		}
		counterVar := ctx.GetVariableByScopeAndName(ctx.Scope(), "counter")
		require.NotNil(t, counterVar)
		require.Equal(t, int64(2), tensors.ToScalar[int64](counterVar.MustValue()))
	})
}

func TestDistributedExec(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	numDevices := backend.NumDevices()
	if numDevices < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", numDevices)
	}
	meshFor2 := must.M1(distributed.NewDeviceMesh([]int{2}, []string{"shards"}))

	t.Run("AutoSharding-Batch", func(t *testing.T) {
		oneLayerExec, err := context.NewExec(backend, nil, oneLayerGraph)
		require.NoError(t, err)
		batchSpec := must.M1(distributed.BuildSpec(meshFor2).S("shards").Done())
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		oneLayerExec = oneLayerExec.AutoSharding(meshFor2).
			WithInputShardingSpecs(batchSpec).
			WithOutputShardingSpecs(batchSpec)
		err = oneLayerExec.SetDefaultShardingSpec(replicatedSpec)
		require.NoError(t, err)

		// With two devices, we want 2 results, one per device.
		x := [][]float64{{1, 1, 1}, {2, 2, 2}}
		distributedX, err := distributed.ShardTensor(batchSpec, tensors.FromValue(x))
		require.NoError(t, err)
		shardedResults, err := oneLayerExec.Exec(distributedX)
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		distributedResult, err := distributed.NewTensor(batchSpec, shardedResults)
		require.NoError(t, err)
		distributedResult.Shape().Equal(shapes.Make(dtypes.Float64, 2, 1))
		gotValue1 := must.M1(distributedResult.Merge()).Value().([][]float64)
		assert.NotEqualf(
			t, 0.0, gotValue1[0][0],
			"Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized", x)
		assert.Equal(t, 2*gotValue1[0][0], gotValue1[1][0])

		// The second call should reuse the graph and yield the same result.
		shardedResults, err = oneLayerExec.Exec(distributedX)
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		distributedResult, err = distributed.NewTensor(batchSpec, shardedResults)
		require.NoError(t, err)
		distributedResult.Shape().Equal(shapes.Make(dtypes.Float64, 2, 1))
		gotValue2 := must.M1(distributedResult.Merge()).Value().([][]float64)
		assert.Equal(t, gotValue1, gotValue2,
			"results of second call of oneLayer(%v) should be the same as the first call", x)

		// Different batch size: it should generate a new graph but yield the same result.
		shardedResults, err = oneLayerExec.Exec(distributedX)
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		distributedResult, err = distributed.NewTensor(batchSpec, shardedResults)
		require.NoError(t, err)
		distributedResult.Shape().Equal(shapes.Make(dtypes.Float64, 4, 1))
		gotValue3 := must.M1(distributedResult.Merge()).Value().([][]float64)
		assert.Equal(t, gotValue3[:2], gotValue1,
			"results of third call of oneLayer(%v) should be the same as the first call", x)
	})
}
