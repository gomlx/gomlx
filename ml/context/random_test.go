package context

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestRandomBernoulli(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := New()
	ctx.RngStateFromSeed(42) // Always the same result.
	gotT := ExecOnce(backend, ctx, func(ctx *Context, g *Graph) *Node {
		ctx.SetTraining(g, true)
		values := ctx.RandomBernoulli(graph.Const(g, 0.13), shapes.Make(dtypes.Float32, 100, 100, 100))
		require.NoError(t, values.Shape().CheckDims(100, 100, 100))
		return graph.ReduceAllMean(values)
	})
	require.InDelta(t, float32(0.13), gotT.Value(), 0.01)
}
