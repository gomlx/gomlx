// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"fmt"
	"strings"
	"testing"

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
	require.NotPanicsf(t, func() { outputs = exec.MustExec(input) }, "%s: failed to exec graph", name)
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
	require.NotPanicsf(t, func() { outputs = exec.MustExec(inputs...) }, "%s: failed to exec graph", name)
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
