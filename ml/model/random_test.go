// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestRandomBernoulli(t *testing.T) {
	backend := testutil.BuildTestBackend()
	scope := model.NewStore()
	scope.SetRNGStateFromSeed(42) // Always the same result.
	gotT := model.MustCallOnce(backend, scope, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		scope.Store().SetTraining(g, true)
		values := scope.RandomBernoulli(graph.Const(g, 0.13), shapes.Make(dtypes.Float32, 100, 100, 100))
		require.NoError(t, values.Shape().CheckDims(100, 100, 100))
		return graph.ReduceAllMean(values)
	})
	require.InDelta(t, float32(0.13), gotT.Value(), 0.01)
}

func TestStorePerAlgorithmRNGState(t *testing.T) {
	goBackend := testutil.GetOfficialBackend("go")
	xlaBackend := testutil.GetOfficialBackend("xla:cpu")

	// 1. Backends resolving RNGDefault to the same concrete algorithm (xla:cpu and go both use Philox)
	// share /#rngState_philox and continue the exact same RNG sequence seamlessly.
	store := model.NewStore()
	require.NoError(t, store.SetRNGStateFromSeed(42)) // Lazy: before any backend is attached.

	step1XLA := model.MustCallOnce(xlaBackend, store, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	step2Go := model.MustCallOnce(goBackend, store, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	require.NotNil(t, store.GetVariable("/#rngState_philox"))
	require.NotEqual(t, step1XLA.Value(), step2Go.Value())

	// Cross-check against running both steps on the Go backend alone from the same seed.
	refStore := model.NewStore()
	require.NoError(t, refStore.SetRNGStateFromSeed(42))
	ref1 := model.MustCallOnce(goBackend, refStore, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	ref2 := model.MustCallOnce(goBackend, refStore, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	require.Equal(t, ref1.Value(), step1XLA.Value())
	require.Equal(t, ref2.Value(), step2Go.Value())

	// 2. Switching RNGAlgorithm on the same Store allocates a separate per-algorithm state variable
	// initialized from the Store's seed, without clobbering /#rngState_philox.
	philoxStateBefore := store.GetVariable("/#rngState_philox").MustValue().Value()
	store.WithRNGAlgorithm(graph.RNGThreeFry)
	tfXLA := model.MustCallOnce(xlaBackend, store, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	require.NotNil(t, store.GetVariable("/#rngState_three_fry"))
	require.Equal(t, philoxStateBefore, store.GetVariable("/#rngState_philox").MustValue().Value())

	// 3. String hyperparameter configuration (ParamRNGAlgorithm = "cha_cha8") + graph fallback on XLA.
	store.SetParam(model.ParamRNGAlgorithm, "cha_cha8")
	chachaXLA := model.MustCallOnce(xlaBackend, store, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	require.NotNil(t, store.GetVariable("/#rngState_cha_cha8"))
	require.NotEqual(t, tfXLA.Value(), chachaXLA.Value())

	// Re-seeding resets all materialized per-algorithm variables in place.
	require.NoError(t, store.SetRNGStateFromSeed(42))
	chachaReset := model.MustCallOnce(xlaBackend, store, func(scope *model.Scope, g *graph.Graph) *graph.Node {
		return scope.RandomUniform(g, shapes.Make(dtypes.Float32, 4))
	})
	require.Equal(t, chachaXLA.Value(), chachaReset.Value())
}
