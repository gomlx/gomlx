// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

func TestSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			logits := graph.Const(g, [][]float64{{-1, 0, 1.}, {-1, 0, 0}})
			inputs = []*graph.Node{logits}
			outputs = []*graph.Node{nn.Softmax(logits)}
			return
		}, []any{
			[][]float64{
				{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
				{0.15536240349696362, 0.4223187982515182, 0.4223187982515182}},
		}, xslices.Epsilon)
}

func TestMaskedSoftmax(t *testing.T) {
	// Values checked with Tensorflow's `tf.nn.softmax()` function.
	graphtest.RunTestGraphFn(t, "TestMaskedSoftmax()",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			logits := graph.Const(g, [][]float64{{-1, 0, 1.}, {-1, 5, 10}})
			mask := graph.Const(g, [][]bool{{true, true, true}, {true, false, false}})
			inputs = []*graph.Node{logits, mask}
			outputs = []*graph.Node{nn.MaskedSoftmax(logits, mask, -1)}
			return
		}, []any{
			[][]float64{{0.09003057317038046, 0.24472847105479764, 0.6652409557748218}, {1, 0, 0}},
		}, xslices.Epsilon)
}

func TestLogSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			logits := graph.Const(g, [][]float32{{-1, 0, 1.}, {-1, 0, 0}})
			inputs = []*graph.Node{logits}
			outputs = []*graph.Node{nn.LogSoftmax(logits)}
			return
		}, []any{
			[][]float32{
				{-2.4076061, -1.407606, -0.40760604},
				{-1.8619947, -0.8619948, -0.8619948}},
		}, xslices.Epsilon)
}

func TestMaskedLogSoftmax(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TestSoftmax()",
		func(g *graph.Graph) (inputs, outputs []*graph.Node) {
			logits := graph.Const(g, [][]float64{{-1, 0, 1., 1000}, {-1, 0, 1000, 0}})
			mask := graph.Const(g, [][]bool{{true, true, true, false}, {true, true, false, true}})
			inputs = []*graph.Node{logits}
			outputs = []*graph.Node{nn.MaskedLogSoftmax(logits, mask)}
			return
		}, []any{
			[][]float64{
				{-2.4076061, -1.407606, -0.40760604, math.Inf(-1)},
				{-1.8619947, -0.8619948, math.Inf(-1), -0.8619948}},
		}, xslices.Epsilon)
}
