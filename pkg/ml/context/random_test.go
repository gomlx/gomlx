package context_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestRandomBernoulli(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42) // Always the same result.
	gotT := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *graph.Graph) *graph.Node {
		ctx.SetTraining(g, true)
		values := ctx.RandomBernoulli(graph.Const(g, 0.13), shapes.Make(dtypes.Float32, 100, 100, 100))
		require.NoError(t, values.Shape().CheckDims(100, 100, 100))
		return graph.ReduceAllMean(values)
	})
	require.InDelta(t, float32(0.13), gotT.Value(), 0.01)
}
