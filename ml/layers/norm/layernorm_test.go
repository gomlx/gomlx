// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

type Shape = shapes.Shape

func IotaP1Initializer(g *Graph, shape Shape) *Node {
	return AddScalar(Iota(g, shape, 0), 1.0)
}

func testSimpleFunc(t *testing.T, name string, input any,
	fn func(scope *model.Scope, input *Node) *Node, want any) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
		return fn(scope.WithInitializer(IotaP1Initializer), input)
	})
	var outputs []*tensors.Tensor
	require.NotPanicsf(t, func() { outputs = exec.MustCall(input) }, "%s: failed to exec graph", name)
	fmt.Printf("\t%s(%v) = %s\n", name, input, outputs[0].GoStr())
	require.Truef(t, xslices.SlicesInDelta(outputs[0].Value(), want, xslices.Epsilon),
		"%s(%v): want=%v, got=%v", name, input, want, outputs[0].GoStr())
}

func testSimpleFuncMany(t *testing.T, name string, inputs []any,
	fn func(scope *model.Scope, inputs []*Node) *Node, want any) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, inputs []*Node) *Node {
		return fn(scope.WithInitializer(IotaP1Initializer), inputs)
	})
	var outputs []*tensors.Tensor
	require.NotPanicsf(t, func() { outputs = exec.MustCall(inputs...) }, "%s: failed to exec graph", name)
	parts := make([]string, len(inputs))
	for ii, input := range inputs {
		parts[ii] = fmt.Sprintf("%v", input)
	}
	inputsStr := strings.Join(parts, ", ")
	fmt.Printf("\t%s(%s) = %s\n", name, inputsStr, outputs[0].GoStr())
	require.Truef(t, xslices.SlicesInDelta(outputs[0].Value(), want, xslices.Epsilon),
		"%s(%s): want=%v, got=%v", name, inputsStr, want, outputs[0].GoStr())
}

func TestLayerNorm(t *testing.T) {
	testSimpleFunc(t, "LayerNorm()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).LearnedOffset(false).LearnedGain(false).Epsilon(0).Done()
		},
		[][]float32{{-1, 1}, {-1, 1}, {-1, 1}},
	)

	testSimpleFunc(t, "LayerNorm()",
		[][]float32{{0, 10}, {20, 30}, {40, 50}},
		func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).LearnedOffset(false).LearnedGain(false).Epsilon(0).ScaleNormalization(false).Done()
		},
		[][]float32{{-5, 5}, {-5, 5}, {-5, 5}},
	)
	testSimpleFuncMany(t, "LayerNorm()",
		[]any{
			[][]float32{{0, 10, 5}, {20, 30, 0}, {0, 30, 50}, {0, 0, 0}},
			[][]bool{{true, true, true}, {true, true, false}, {true, false, true}, {false, false, false}},
		},
		func(scope *model.Scope, inputs []*Node) *Node {
			return LayerNorm(scope, inputs[0], -1).Mask(inputs[1]).
				LearnedOffset(false).LearnedGain(false).Epsilon(0).
				ScaleNormalization(false).Done()
		},
		[][]float32{{-5, 5, 0}, {-5, 5, 0}, {-25, 0, 25}, {0, 0, 0}},
	)
}

func TestLayerNormDType(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("Float16DefaultVars", func(t *testing.T) {
		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).Epsilon(0).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10), float16.FromFloat32s(20, 30)}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, dtypes.Float16, store.GetVariable("/layer_normalization/gain").DType())
		require.Equal(t, dtypes.Float16, store.GetVariable("/layer_normalization/offset").DType())

		got := out.Value().([][]float16.Float16)
		require.InDelta(t, -1.0, float64(got[0][0].Float32()), 1e-2)
		require.InDelta(t, 1.0, float64(got[0][1].Float32()), 1e-2)
		require.InDelta(t, -1.0, float64(got[1][0].Float32()), 1e-2)
		require.InDelta(t, 1.0, float64(got[1][1].Float32()), 1e-2)
	})

	t.Run("UnexpectedStoreDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("layer_normalization").VariableWithShape("gain", shapes.Make(dtypes.Float32, 2))
		store.RootScope().In("layer_normalization").VariableWithShape("offset", shapes.Make(dtypes.Float32, 2))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10)}
		require.Panics(t, func() { exec.MustCall(in) })
	})

	t.Run("Float16ExplicitWithDTypeFloat32", func(t *testing.T) {
		store := model.NewStore()
		gainVar := store.RootScope().In("layer_normalization").VariableWithShape("gain", shapes.Make(dtypes.Float32, 2))
		require.NoError(t, gainVar.SetValue(tensors.FromValue([]float32{2.0, 3.0})))
		offsetVar := store.RootScope().In("layer_normalization").VariableWithShape("offset", shapes.Make(dtypes.Float32, 2))
		require.NoError(t, offsetVar.SetValue(tensors.FromValue([]float32{0.5, -0.5})))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).Epsilon(0).WithDType(dtypes.Float32).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10)}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_normalization/gain").DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_normalization/offset").DType())

		got := out.Value().([][]float16.Float16)
		// [-1*2 + 0.5, 1*3 - 0.5] = [-1.5, 2.5]
		require.InDelta(t, -1.5, float64(got[0][0].Float32()), 1e-2)
		require.InDelta(t, 2.5, float64(got[0][1].Float32()), 1e-2)
	})

	t.Run("ExplicitWithDType", func(t *testing.T) {
		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).WithDType(dtypes.Float32).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10)}
		out := exec.MustCall(in)[0]
		require.Equal(t, dtypes.Float16, out.DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_normalization/gain").DType())
		require.Equal(t, dtypes.Float32, store.GetVariable("/layer_normalization/offset").DType())
	})

	t.Run("ExplicitWithDTypeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("layer_normalization").VariableWithShape("gain", shapes.Make(dtypes.Float32, 2))
		store.RootScope().In("layer_normalization").VariableWithShape("offset", shapes.Make(dtypes.Float32, 2))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).WithDType(dtypes.Float16).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10)}
		require.Panics(t, func() { exec.MustCall(in) })
	})

	t.Run("ExistingVarShapeMismatchPanics", func(t *testing.T) {
		store := model.NewStore()
		store.RootScope().In("layer_normalization").VariableWithShape("gain", shapes.Make(dtypes.Float32, 4))
		store.RootScope().In("layer_normalization").VariableWithShape("offset", shapes.Make(dtypes.Float32, 4))

		exec := model.MustNewExec(backend, store, func(scope *model.Scope, input *Node) *Node {
			return LayerNorm(scope, input, -1).Done()
		})
		in := [][]float16.Float16{float16.FromFloat32s(0, 10)}
		require.Panics(t, func() { exec.MustCall(in) })
	})
}
