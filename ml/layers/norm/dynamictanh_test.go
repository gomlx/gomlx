// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"testing"

	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestDynamicTanh(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()

	exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		return DynamicTanh(scope, x).WithAlpha(1.0).Done()
	})

	x := [][][]float32{{{1, 2}, {3, 4}}}
	got := exec.MustCall(x)[0]
	require.NoError(t, got.Shape().CheckDims(1, 2, 2), "shape should not have changed")

	want := [][][]float32{{
		{0.7615942, 0.9640276}, {0.9950547, 0.9993292},
	}}
	if ok, diff := testutil.IsInDelta(want, got.Value(), 1e-3); !ok {
		t.Errorf("DynamicTanh mismatch (-want +got):\n%s", diff)
	}

	alphaVar := store.GetVariable("/dynamic_tanh/alpha")
	require.NotNil(t, alphaVar)
	require.NoError(t, alphaVar.Shape().CheckDims())

	gammaVar := store.GetVariable("/dynamic_tanh/gamma")
	require.NotNil(t, gammaVar)
	require.NoError(t, gammaVar.Shape().CheckDims(2))

	betaVar := store.GetVariable("/dynamic_tanh/beta")
	require.NotNil(t, betaVar)
	require.NoError(t, betaVar.Shape().CheckDims(2))
}
