// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

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

func TestRMSNormDType(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("Float16NumericalStabilityAndDefaultVar", func(t *testing.T) {
		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).Done()
		})
		// Squares (90,000 + 160,000) overflow Float16 max (65504), so normalization must
		// run in Float32 to stay finite and accurate: rms = sqrt((300^2 + 400^2)/2) = 353.5534.
		in := [][]float16.Float16{float16.FromFloat32s(300, 400)}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, dtypes.Float16, store.GetVariable("/rms_norm/scale").DType())

		got := out.Value().([][]float16.Float16)
		v0 := float64(got[0][0].Float32())
		v1 := float64(got[0][1].Float32())
		require.False(t, math.IsNaN(v0) || math.IsInf(v0, 0))
		require.False(t, math.IsNaN(v1) || math.IsInf(v1, 0))
		require.InDelta(t, 300.0/math.Sqrt((300*300+400*400)/2.0), v0, 1e-2)
		require.InDelta(t, 400.0/math.Sqrt((300*300+400*400)/2.0), v1, 1e-2)
	})

	t.Run("UnexpectedStoreDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("rms_norm").VariableWithShape("scale", shapes.Make(dtypes.Float32, 2))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).Done()
		})
		in := [][]bfloat16.BFloat16{{bfloat16.FromFloat32(3), bfloat16.FromFloat32(4)}}
		require.Panics(t, func() { exec.MustCall(in) })
	})

	t.Run("BFloat16ExplicitWithDTypeFloat32Scale", func(t *testing.T) {
		store := model.NewStore()
		scaleVar := store.RootScope().In("rms_norm").VariableWithShape("scale", shapes.Make(dtypes.Float32, 2))
		require.NoError(t, scaleVar.SetValue(tensors.FromValue([]float32{2.0, 0.5})))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).WithScaleOffset(1.0).WithDType(dtypes.Float32).Done()
		})
		in := [][]bfloat16.BFloat16{{bfloat16.FromFloat32(3), bfloat16.FromFloat32(4)}}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.BFloat16, out.DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/rms_norm/scale").DType())

		rms := math.Sqrt((9.0+16.0)/2.0 + 1e-6)
		got := out.Value().([][]bfloat16.BFloat16)
		require.InDelta(t, (3.0/rms)*(2.0+1.0), float64(got[0][0].Float32()), 2e-2)
		require.InDelta(t, (4.0/rms)*(0.5+1.0), float64(got[0][1].Float32()), 2e-2)
	})

	t.Run("ExplicitWithDTypeAndNormalizationDType", func(t *testing.T) {
		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).
				WithDType(dtypes.Float32).
				WithNormalizationDType(dtypes.Float32).
				Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(3, 4)}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/rms_norm/scale").DType())
	})

	t.Run("ExplicitWithDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("rms_norm").VariableWithShape("scale", shapes.Make(dtypes.Float32, 2))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).WithDType(dtypes.Float16).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(3, 4)}
		require.Panics(t, func() { exec.MustCall(in) })
	})

	t.Run("ExistingScaleShapeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("rms_norm").VariableWithShape("scale", shapes.Make(dtypes.Float32, 4))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
			return RMSNorm(scope, x).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(3, 4)}
		require.Panics(t, func() { exec.MustCall(in) })
	})
}
