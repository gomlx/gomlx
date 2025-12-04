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

	// Checks that Context.RNGState is auto-initialized correctly.
	t.Run("Context.RNGState initialization", func(t *testing.T) {
		ctx := context.New()
		result, err := context.ExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			return Add(ctx.RandomUniform(g, shapes.Make(dtypes.Float32, 10)), Const(g, float32(0.0001)))
		})
		require.NoError(t, err)
		assert.NotNil(t, result)
		fmt.Printf("- Random uniform: %s\n", result)
		values := result.Value().([]float32)
		for _, v := range values {
			if v == 0 {
				fmt.Printf("Expected non-zero value, got %v", values)
				t.FailNow()
			}
		}
	})

	// Checks that Context.RNGState is auto-initialized correctly.
	t.Run("variable initialization", func(t *testing.T) {
		ctx := context.New()
		result, err := context.ExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			v := ctx.WithInitializer(initializers.RandomUniformFn(ctx, 0.0001, 1.0)).
				VariableWithShape("v", shapes.Make(dtypes.Float32, 10))
			return v.ValueGraph(g)
		})
		require.NoError(t, err)
		assert.NotNil(t, result)
		fmt.Printf("- Uniformly initialized variable: %s\n", result)
		values := result.Value().([]float32)
		for _, v := range values {
			if v == 0 {
				fmt.Printf("Expected non-zero value, got %v", values)
				t.FailNow()
			}
		}
	})

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

func TestAutoSharding(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	numDevices := backend.NumDevices()
	if numDevices < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", numDevices)
	}
	meshFor2 := must.M1(distributed.NewDeviceMesh([]int{2}, []string{"shards"}))

	// Replicating variable: initializing it's value non-sharde should trigger automatic replication.
	// for each device.
	t.Run("replicated variable", func(t *testing.T) {
		ctx := context.New()
		xVar := ctx.VariableWithValue("x", []float32{1.0, 2.0, 3.0})
		e := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			return xVar.ValueGraph(g)
		})
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(replicatedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, ctx.ResetRNGState())
		shardedResults, err := e.Exec()
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		fmt.Println("x:")
		fmt.Printf("\t- [Shard #0]: %s\n\t- [Shard #1]: %s\n", shardedResults[0], shardedResults[1])
		require.Equalf(t, shardedResults[0].Value(), shardedResults[1].Value(),
			"Expected replicated variable to be the same in every device.")
		for shardIdx, shardT := range shardedResults {
			shard := shardT.Value().([]float32)
			var hasNonZero bool
			for _, v := range shard {
				if v != 0 {
					hasNonZero = true
					break
				}
			}
			if !hasNonZero {
				t.Errorf("Shard #%d is all zeroes", shardIdx)
			}
		}
	})

	// RNGState is special: it must be initialized as "replicated", but the values are different
	// for each device.
	t.Run("RandomUniform", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParam(context.ParamInitialSeed, int64(42))
		e := context.MustNewExec(backend, ctx, func(ctx *context.Context, g *Graph) (*Node, *Node) {
			a := ctx.RandomUniform(g, shapes.Make(dtypes.Float32, 2, 3))
			b := ctx.RandomUniform(g, shapes.Make(dtypes.Float32, 2, 3))
			return a, b
		})
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		shardedSpec := must.M1(distributed.BuildSpec(meshFor2).S("shards").Done())
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(shardedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, ctx.ResetRNGState())
		shardedResults, err := e.Exec()
		require.NoError(t, err)
		require.Len(t, shardedResults, 4)
		a := must.M1(distributed.NewTensor(shardedSpec, []*tensors.Tensor{shardedResults[0], shardedResults[2]}))
		b := must.M1(distributed.NewTensor(shardedSpec, []*tensors.Tensor{shardedResults[1], shardedResults[3]}))
		fmt.Println("RandomUniform([2, 3]):")
		fmt.Printf("\t- A [Shard #0]: %s\n\t- A [Shard #1]: %s\n", a.Shards()[0], a.Shards()[1])
		fmt.Printf("\t- B [Shard #0]: %s\n\t- B [Shard #1]: %s\n", b.Shards()[0], b.Shards()[1])
		require.NotEqualf(t, a.Shards()[0].Value(), a.Shards()[1].Value(),
			"Expected sharded random values of A to be the different in every device.")
		require.NotEqualf(t, b.Shards()[0].Value(), b.Shards()[1].Value(),
			"Expected sharded random values of B to be the different in every device.")
		require.NotEqualf(t, a.Shards()[0].Value(), b.Shards()[0].Value(),
			"Expected sharded random values of A and B to be the different in every device.")
		require.NotEqualf(t, a.Shards()[1].Value(), b.Shards()[1].Value(),
			"Expected sharded random values of A and B to be the different in every device.")
	})

	t.Run("variable initialization", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParam(context.ParamInitialSeed, int64(42))
		e, err := context.NewExec(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			// v := ctx.WithInitializer(initializers.RandomUniformFn(ctx, 0.0001, 1.0)).VariableWithShape("v", shapes.Make(dtypes.Float32, 10))
			// return v.ValueGraph(g)
			// return ctx.RandomUniform(g, shapes.Make(dtypes.Float32, 10))
			return ctx.GetVariable(context.RNGStateVariableName).ValueGraph(g)
		})
		require.NoError(t, err)
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(replicatedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, ctx.ResetRNGState())

		shardedResults, err := e.Exec()
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		fmt.Println("\tVariable:")
		fmt.Printf("\t- [Shard #0]: %s\n\t- [Shard #1]: %s\n", shardedResults[0], shardedResults[1])
		require.Equalf(t, shardedResults[0].Value(), shardedResults[1].Value(),
			"Expected replicated initialization to be the same in every device.")
	})

	t.Run("batch sharding", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParam(context.ParamInitialSeed, int64(42))
		oneLayerExec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x *Node) (*Node, *Node) {
			g := x.Graph()
			wVar := ctx.VariableWithShape("w", shapes.Make(dtypes.F64, 3, 1))
			w := wVar.ValueGraph(g)
			bVar := ctx.VariableWithShape("b", shapes.Make(dtypes.F64, 1))
			b := bVar.ValueGraph(g)
			y := MatMul(x, w)
			y = Add(y, ExpandLeftToRank(b, y.Rank()))
			return y, w
		})
		require.NoError(t, err)
		batchSpec := must.M1(distributed.BuildSpec(meshFor2).S("shards").Done())
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		oneLayerExec = oneLayerExec.AutoSharding(meshFor2).
			WithInputShardingSpecs(batchSpec).
			WithOutputShardingSpecs(batchSpec, replicatedSpec)
		err = oneLayerExec.SetDefaultShardingSpec(replicatedSpec)
		require.NoError(t, err)

		// With two devices, we want 2 results, one per device.
		// x := [][]float64{{1, 1, 1}, {10, 10, 10}, {1, 1, 1}, {10, 10, 10}}
		x := [][]float64{{1, 1, 1}, {10, 10, 10}}
		distributedX, err := distributed.ShardTensor(batchSpec, tensors.FromValue(x))
		fmt.Printf("\tInput x: shape=%s, shardShape=%s\n", distributedX.Shape(), distributedX.ShardShape())
		fmt.Printf("\t- [Shard #0]: %s\n\t- [Shard #1]: %s\n", distributedX.Shards()[0], distributedX.Shards()[1])
		require.NoError(t, err)
		results, err := oneLayerExec.DistributedExec(distributedX)
		require.NoError(t, err)
		require.Len(t, results, 2)
		y0, w := results[0], results[1]
		fmt.Printf("\ty_{t=0} = %s\n", must.M1(y0.Merge()))
		fmt.Printf("\tw       = %s\n", must.M1(w.Merge()))

		// The second call should reuse the graph and yield the same result.
		results, err = oneLayerExec.DistributedExec(distributedX)
		require.NoError(t, err)
		require.Len(t, results, 2)
		y1 := results[0]
		fmt.Printf("\ty_{t=1} = %s\n", must.M1(y1.Merge()))

		assert.Equal(t, must.M1(y0.Merge()).Value(), must.M1(y1.Merge()).Value())
	})
}
