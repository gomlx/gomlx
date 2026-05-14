// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package context_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestRandomBernoulli(t *testing.T) {
	backend := testutil.BuildTestBackend()
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
