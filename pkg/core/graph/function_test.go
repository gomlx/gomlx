package graph_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFunction(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("IsMain", func(t *testing.T) {
		g := graph.NewGraph(backend, t.Name())
		_ = graph.Parameter(g, "x", shapes.Make(dtypes.F32))
		assert.True(t, g.IsMainFunc())
		assert.True(t, g.CurrentFunc().IsMain())
		g.Finalize()
	})

	t.Run("NewFunctionWithSharding", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		if backend.NumDevices() < 2 {
			t.Skip("Test requires at least 2 devices for sharding")
		}

		t.Run("ValidSharding", func(t *testing.T) {
			g := graph.NewGraph(backend, "ValidSharding")
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"data"})
			require.NoError(t, err)
			require.NoError(t, g.SetAutoSharding(mesh))

			f := graph.NewFunctionWithSharding(g, "shard_fn", func(g *graph.Graph) ([]*graph.Node, []*distributed.ShardingSpec) {
				p := graph.Parameter(g, "p", shapes.Make(dtypes.F32, 4))
				spec, err := distributed.NewShardingSpec(mesh, []string{"data"}) // Split 4 into 2 shards of 2
				require.NoError(t, err)
				return []*graph.Node{p}, []*distributed.ShardingSpec{spec}
			})
			require.NotNil(t, f)
			assert.Equal(t, "shard_fn", f.Name())
		})

		t.Run("RankMismatch", func(t *testing.T) {
			g := graph.NewGraph(backend, "RankMismatch")
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"data"})
			require.NoError(t, err)
			require.NoError(t, g.SetAutoSharding(mesh))

			assert.Panics(t, func() {
				graph.NewFunctionWithSharding(g, "fail_fn", func(g *graph.Graph) ([]*graph.Node, []*distributed.ShardingSpec) {
					p := graph.Parameter(g, "p", shapes.Make(dtypes.F32, 4, 1))      // Rank 2
					spec, err := distributed.NewShardingSpec(mesh, []string{"data"}) // Rank 1
					require.NoError(t, err)
					return []*graph.Node{p}, []*distributed.ShardingSpec{spec}
				})
			})
		})

		t.Run("NotDivisible", func(t *testing.T) {
			g := graph.NewGraph(backend, "NotDivisible")
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"data"})
			require.NoError(t, err)
			require.NoError(t, g.SetAutoSharding(mesh))

			assert.Panics(t, func() {
				graph.NewFunctionWithSharding(g, "fail_fn", func(g *graph.Graph) ([]*graph.Node, []*distributed.ShardingSpec) {
					p := graph.Parameter(g, "p", shapes.Make(dtypes.F32, 3)) // 3 not divisible by 2
					spec, err := distributed.NewShardingSpec(mesh, []string{"data"})
					require.NoError(t, err)
					return []*graph.Node{p}, []*distributed.ShardingSpec{spec}
				})
			})
		})

		t.Run("LengthMismatch", func(t *testing.T) {
			g := graph.NewGraph(backend, "LengthMismatch")
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"data"})
			require.NoError(t, err)
			require.NoError(t, g.SetAutoSharding(mesh))

			assert.Panics(t, func() {
				graph.NewFunctionWithSharding(g, "fail_fn", func(g *graph.Graph) ([]*graph.Node, []*distributed.ShardingSpec) {
					p := graph.Parameter(g, "p", shapes.Make(dtypes.F32, 4))
					spec, err := distributed.NewShardingSpec(mesh, []string{"data"})
					require.NoError(t, err)
					return []*graph.Node{p}, []*distributed.ShardingSpec{spec, spec} // 2 specs, 1 output
				})
			})
		})
	})

	t.Run("Call", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "Call")

		// Define a simple function: f(a, b) = a + b
		addFn := graph.NewFunction(g, "add", func(g *graph.Graph) []*graph.Node {
			a := graph.Parameter(g, "a", shapes.Make(dtypes.F32))
			b := graph.Parameter(g, "b", shapes.Make(dtypes.F32))
			return []*graph.Node{graph.Add(a, b)}
		})

		// Call it from main
		a := graph.Const(g, float32(10))
		b := graph.Const(g, float32(32))
		sum := addFn.Call(a, b)[0]

		g.Compile(sum)
		result := g.Run()[0]
		assert.Equal(t, float32(42), result.Value().(float32))
	})

	t.Run("InvalidCall", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "Call")

		// Define a simple function: f(a, b) = a + b
		addFn := graph.NewFunction(g, "add", func(g *graph.Graph) []*graph.Node {
			a := graph.Parameter(g, "a", shapes.Make(dtypes.F32))
			b := graph.Parameter(g, "b", shapes.Make(dtypes.F32))
			return []*graph.Node{graph.Add(a, b)}
		})

		// Test invalid call (wrong number of arguments)
		a := graph.Const(g, float32(10))
		assert.Panics(t, func() {
			addFn.Call(a)
		})

		// Test invalid call (wrong rank)
		v := graph.Const(g, []float32{1.0})
		assert.Panics(t, func() {
			addFn.Call(v, a)
		})
	})
}
