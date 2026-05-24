// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestRMSNorm(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		return RMSNorm(scope, x).
			WithNormalizationAxes(-1, -2).
			Done()
	})

	x := [][][]float32{
		{{3, 4}, {5, 6}, {7, 8}},
		{{3 * 2, 4 * 2}, {5 * 2, 6 * 2}, {7 * 2, 8 * 2}},
	}
	got := exec.MustCall(x)[0]
	require.NoError(t, got.Shape().CheckDims(2, 3, 2), "shape should not have changed")
	want := [][][]float32{{
		{0.52091914, 0.6945589}, {0.86819863, 1.0418383}, {1.2154781, 1.3891178},
	}, {
		{0.52091914, 0.6945589}, {0.86819863, 1.0418383}, {1.2154781, 1.3891178},
	}}
	if ok, diff := testutil.IsInDelta(want, got.Value(), 1e-3); !ok {
		t.Errorf("RMSNorm mismatch (-want +got):\n%s", diff)
	}

	scaleVar := store.GetVariable("/rms_norm/scale")
	require.NotNil(t, scaleVar)
	require.NoError(t, scaleVar.Shape().CheckDims(3, 2))
}
