// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func denseWithBias(scope *model.Scope, x *Node, outputDim int) *Node {
	g := x.Graph()
	inputDim := x.Shape().Dimensions[x.Shape().Rank()-1]
	w := scope.VariableWithShape("weights", shapes.Make(x.DType(), inputDim, outputDim)).NodeValue(g)
	b := scope.VariableWithShape("biases", shapes.Make(x.DType(), outputDim)).NodeValue(g)
	y := MatMul(x, w)
	return Add(y, ExpandLeftToRank(b, y.Rank()))
}

func oneLayerGraph(scope *model.Scope, x *Node) *Node {
	return denseWithBias(scope.In("dense0"), x, 1)
}

func oneLayerManyInputsGraph(scope *model.Scope, inputs []*Node) *Node {
	return denseWithBias(scope.In("dense0"), Concatenate(inputs, -1), 1)
}

func TestExec(t *testing.T) {
	backend := testutil.BuildTestBackend()

	// Checks that Scope.RNGState is auto-initialized correctly.
	t.Run("Scope.RNGState initialization", func(t *testing.T) {
		scope := model.NewStore()
		result, err := model.CallOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
			return Add(scope.RandomUniform(g, shapes.Make(dtypes.Float32, 10)), Const(g, float32(0.0001)))
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

	// Checks that Scope.RNGState is auto-initialized correctly.
	t.Run("variable initialization", func(t *testing.T) {
		scope := model.NewStore()
		result, err := model.CallOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
			v := scope.WithInitializer(initializer.RandomUniformFn(scope, 0.0001, 1.0)).
				VariableWithShape("v", shapes.Make(dtypes.Float32, 10))
			return v.NodeValue(g)
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
		store := model.NewStore()
		oneLayer, err := model.NewExec(backend, store, oneLayerGraph)
		if err != nil {
			t.Fatalf("Failed to create model.Exec for oneLayer: %+v", err)
		}

		// First graph build, variable should be initialized.
		x := [][]float64{{1, 1, 1}}
		got := oneLayer.MustCall(x)[0].Value().([][]float64)[0][0]
		assert.NotEqual(
			t,
			0.0,
			got,
			"Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized",
			x,
		)

		// The second call should reuse the graph and yield the same result.
		got2 := oneLayer.MustCall(x)[0].Value().([][]float64)[0][0]
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
		got3 := oneLayer.MustCall(x)[0].Value().([][]float64)[0][0]
		assert.InDeltaf(t, got, got3, 1e-3,
			"Exec to oneLayer(%v) on a larger batch returned %f, but original call returned %f",
			x, got, got3)

		// If X changes the inner dimension, then the layers.DenseWithBias would require different shaped variables, which should
		// fail to build.
		x = [][]float64{{1, 1, 1, 1}}
		require.Panics(t, func() { _ = oneLayer.MustCall(x) },
			"oneLayer(%v) leads to a graph with differently shaped variables should have failed", x)
	})

	t.Run("WithSlices", func(t *testing.T) {
		oneLayer, err := model.NewExec(backend, model.NewStore(), oneLayerManyInputsGraph)
		if err != nil {
			t.Fatalf("Failed to create model.Exec for oneLayer: %+v", err)
		}

		// First execution builds graph and should initialize variables (weights) randomly.
		x := [][]float64{{1, 1, 1}}
		got := oneLayer.MustCall(x)[0].Value().([][]float64)
		fmt.Printf("\toneLayer(%v)=%v\n", x, got)
		if got[0][0] == 0 {
			t.Fatalf("Failed evaluating oneLayer(%v) returned 0, but weights should have been randomly initialized", x)
		}

		// Second execution: two tensors that if concatenated will be the same as the previous run. Since the scope
		// and hence the variables are the same, it should evaluate to exact same value.
		oneElementOfX := [][]float64{{1}}
		got2 := oneLayer.MustCall(oneElementOfX, oneElementOfX, oneElementOfX)[0].Value().([][]float64)
		fmt.Printf("\toneLayer(%v, %v, %v)=%v\n", oneElementOfX, oneElementOfX, oneElementOfX, got)
		require.Truef(t, math.Abs(got[0][0]-got2[0][0]) < 1.0e-3,
			"oneLayer(%v) should have been the same as oneLayer(%v, %v, %v), but got %v and %v",
			x, oneElementOfX, oneElementOfX, oneElementOfX, got, got2)
	})

	t.Run("VariableUpdates", func(t *testing.T) {
		scope := model.NewStore()
		counter := model.MustNewExec(backend, scope, func(scope *model.Scope, g *Graph) *Node {
			dtype := dtypes.Int64
			counterVar := scope.WithInitializer(initializer.Zero).VariableWithShape("counter", shapes.Make(dtype))
			counterNode := counterVar.NodeValue(g)
			counterNode = Add(counterNode, OnesLike(counterNode))
			counterVar.SetNodeValue(counterNode)
			return counterNode
		})
		results := counter.MustCall()
		gotTensor := results[0]
		got := gotTensor.Value().(int64)
		if got != 1 {
			t.Fatalf("Wanted first counter value to be 1, got %d instead", got)
		}
		results = counter.MustCall()
		gotTensor = results[0]
		got = gotTensor.Value().(int64)
		if got != 2 {
			t.Fatalf("Wanted second counter value to be 2, got %d instead", got)
		}
		counterVar := scope.GetVariable("/counter")
		require.NotNil(t, counterVar)
		require.Equal(t, int64(2), tensors.ToScalar[int64](counterVar.MustValue()))
	})

	t.Run("DynamicShapes", func(t *testing.T) {
		// Use simple Go backend
		simpleBackend, err := gobackend.New("")
		require.NoError(t, err)

		store := model.NewStore()
		store.SetParam(model.ParamInitialSeed, int64(42))

		// Function that defines variables and performs a dense-bias operation with dynamic shape.
		// Input shape is [batch, 3], output is [batch, 2].
		exec, err := model.NewExec(simpleBackend, store, func(scope *model.Scope, x *Node) *Node {
			return denseWithBias(scope.In("dense"), x, 2)
		})
		require.NoError(t, err)
		exec.WithDynamicAxes([]string{"batch", ""})

		// Run with batch=2
		x2 := [][]float64{{1, 2, 3}, {4, 5, 6}}
		out2, err := exec.Exec(x2)
		require.NoError(t, err)
		require.Len(t, out2, 1)
		val2 := out2[0].Value().([][]float64)
		assert.Equal(t, 2, len(val2))
		assert.Equal(t, 2, len(val2[0]))

		// Run with batch=3
		x3 := [][]float64{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}
		out3, err := exec.Exec(x3)
		require.NoError(t, err)
		require.Len(t, out3, 1)
		val3 := out3[0].Value().([][]float64)
		assert.Equal(t, 3, len(val3))
		assert.Equal(t, 2, len(val3[0]))
	})
}

func TestAutoSharding(t *testing.T) {
	backend := testutil.BuildTestBackend()
	numDevices := backend.NumDevices()
	if numDevices < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", numDevices)
	}
	meshFor2 := must.M1(distributed.NewDeviceMesh([]int{2}, []string{"shards"}))

	// Replicating variable: initializing it's value non-sharde should trigger automatic replication.
	// for each device.
	t.Run("replicated variable", func(t *testing.T) {
		store := model.NewStore()
		xVar := store.RootScope().VariableWithValue("x", []float32{1.0, 2.0, 3.0})
		e := model.MustNewExec(backend, store, func(scope *model.Scope, g *Graph) *Node {
			return xVar.NodeValue(g)
		})
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(replicatedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, store.ResetRNGState())
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
		store := model.NewStore()
		store.SetParam(model.ParamInitialSeed, int64(42))
		e := model.MustNewExec(backend, store, func(scope *model.Scope, g *Graph) (*Node, *Node) {
			a := scope.RandomUniform(g, shapes.Make(dtypes.Float32, 2, 3))
			b := scope.RandomUniform(g, shapes.Make(dtypes.Float32, 2, 3))
			return a, b
		})
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		shardedSpec := must.M1(distributed.BuildSpec(meshFor2).S("shards").Done())
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(shardedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, store.ResetRNGState())
		shardedResults, err := e.Exec()
		require.NoError(t, err)
		require.Len(t, shardedResults, 4)
		a := must.M1(dtensor.NewTensor(shardedSpec, []*tensors.Tensor{shardedResults[0], shardedResults[2]}))
		b := must.M1(dtensor.NewTensor(shardedSpec, []*tensors.Tensor{shardedResults[1], shardedResults[3]}))
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
		store := model.NewStore()
		store.SetParam(model.ParamInitialSeed, int64(42))
		e, err := model.NewExec(backend, store, func(scope *model.Scope, g *Graph) *Node {
			return scope.Store().GetVariable(model.RNGStateVariableName).NodeValue(g)
		})
		require.NoError(t, err)
		replicatedSpec := distributed.NewReplicatedShardingSpec(meshFor2)
		e = e.AutoSharding(meshFor2).
			WithOutputShardingSpecs(replicatedSpec)
		require.NoError(t, e.SetDefaultShardingSpec(replicatedSpec))
		require.NoError(t, store.ResetRNGState())

		shardedResults, err := e.Exec()
		require.NoError(t, err)
		require.Len(t, shardedResults, 2)
		fmt.Println("\tVariable:")
		fmt.Printf("\t- [Shard #0]: %s\n\t- [Shard #1]: %s\n", shardedResults[0], shardedResults[1])
		require.Equalf(t, shardedResults[0].Value(), shardedResults[1].Value(),
			"Expected replicated initialization to be the same in every device.")
	})

	t.Run("batch sharding", func(t *testing.T) {
		store := model.NewStore()
		store.SetParam(model.ParamInitialSeed, int64(42))
		oneLayerExec, err := model.NewExec(backend, store, func(scope *model.Scope, x *Node) (*Node, *Node) {
			g := x.Graph()
			wVar := scope.VariableWithShape("w", shapes.Make(dtypes.F64, 3, 1))
			w := wVar.NodeValue(g)
			bVar := scope.VariableWithShape("b", shapes.Make(dtypes.F64, 1))
			b := bVar.NodeValue(g)
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
		distributedX, err := dtensor.ShardTensor(batchSpec, tensors.FromValue(x))
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

func TestExecWrappers(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("OneOutput", func(t *testing.T) {
		store := model.NewStore()
		graphFn := func(scope *model.Scope, x *Node) *Node {
			return AddScalar(x, 1)
		}
		exec1, err := model.NewExec1(backend, store, graphFn)
		require.NoError(t, err)

		res, err := exec1.Call(1.0)
		require.NoError(t, err)
		assert.Equal(t, 2.0, res.Value())

		resMust := exec1.MustCall(2.0)
		assert.Equal(t, 3.0, resMust.Value())

		// Test MustNewExec1 as well.
		exec1Must := model.MustNewExec1(backend, store, graphFn)
		assert.Equal(t, 5.0, exec1Must.MustCall(4.0).Value())
	})

	t.Run("TwoOutputs", func(t *testing.T) {
		store := model.NewStore()
		graphFn := func(scope *model.Scope, x *Node) (*Node, *Node) {
			return AddScalar(x, 1), AddScalar(x, 2)
		}
		exec2, err := model.NewExec2(backend, store, graphFn)
		require.NoError(t, err)

		r1, r2, err := exec2.Call(1.0)
		require.NoError(t, err)
		assert.Equal(t, 2.0, r1.Value())
		assert.Equal(t, 3.0, r2.Value())

		r1Must, r2Must := exec2.MustCall(2.0)
		assert.Equal(t, 3.0, r1Must.Value())
		assert.Equal(t, 4.0, r2Must.Value())

		// Test MustNewExec2 as well.
		exec2Must := model.MustNewExec2(backend, store, graphFn)
		mr1, mr2 := exec2Must.MustCall(4.0)
		assert.Equal(t, 5.0, mr1.Value())
		assert.Equal(t, 6.0, mr2.Value())
	})

	t.Run("ThreeOutputs", func(t *testing.T) {
		store := model.NewStore()
		graphFn := func(scope *model.Scope, x *Node) (*Node, *Node, *Node) {
			return AddScalar(x, 1), AddScalar(x, 2), AddScalar(x, 3)
		}
		exec3, err := model.NewExec3(backend, store, graphFn)
		require.NoError(t, err)

		r1, r2, r3, err := exec3.Call(1.0)
		require.NoError(t, err)
		assert.Equal(t, 2.0, r1.Value())
		assert.Equal(t, 3.0, r2.Value())
		assert.Equal(t, 4.0, r3.Value())

		r1Must, r2Must, r3Must := exec3.MustCall(2.0)
		assert.Equal(t, 3.0, r1Must.Value())
		assert.Equal(t, 4.0, r2Must.Value())
		assert.Equal(t, 5.0, r3Must.Value())

		// Test MustNewExec3 as well.
		exec3Must := model.MustNewExec3(backend, store, graphFn)
		mr1, mr2, mr3 := exec3Must.MustCall(4.0)
		assert.Equal(t, 5.0, mr1.Value())
		assert.Equal(t, 6.0, mr2.Value())
		assert.Equal(t, 7.0, mr3.Value())
	})
}

