// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gnn

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

func TestPoolMessagesWithFixedShape(t *testing.T) {
	ctx := context.New()
	ctx.SetParam(ParamPoolingType, "sum|max")
	graphtest.RunTestGraphFn(
		t, "poolMessagesWithFixedShape()",
		func(g *Graph) (inputs, outputs []*Node) {
			mask := Const(g, [][]bool{
				{true, false, true},
				{false, false, false}})
			x := IotaFull(g, shapes.Make(dtypes.Float32, append(mask.Shape().Dimensions, 5)...))
			degree := Const(g, [][]float32{{10}, {7}})
			outputNoDegree := poolMessagesWithFixedShape(ctx, x, mask, nil)
			outputWithDegree := poolMessagesWithFixedShape(ctx, x, mask, degree)
			inputs = []*Node{x, mask}
			outputs = []*Node{outputNoDegree, outputWithDegree}
			return
		}, []any{
			[][]float32{
				{ /* sum */ 10, 12, 14, 16, 18 /* max */, 10, 11, 12, 13, 14},
				{ /* sum */ 0, 0, 0, 0, 0 /* max */, 0, 0, 0, 0, 0},
			},
			[][]float32{
				{ /* sum */ 50, 60, 70, 80, 90 /* max */, 10, 11, 12, 13, 14},
				{ /* sum */ 0, 0, 0, 0, 0 /* max */, 0, 0, 0, 0, 0},
			},
		}, xslices.Epsilon)
}

func TestPoolMessagesWithAdjacency(t *testing.T) {
	ctx := context.New()
	ctx.SetParam(ParamPoolingType, "sum|mean")
	graphtest.RunTestGraphFn(
		t, "poolMessagesWithAdjacency()",
		func(g *Graph) (inputs, outputs []*Node) {
			// 4 source nodes.
			source := IotaFull(g, shapes.Make(dtypes.Float32, 4, 2))
			// Edges: source/target pairs.
			edgesSource := Const(g, []int{0, 1, 1, 2})
			edgesTarget := Const(g, []int{0, 2, 3, 3})

			// 4 target nodes.
			targetSize := 4

			degree := Const(g, [][]float32{{1}, {1000}, {10}, {100}})
			outputNoDegree := poolMessagesWithAdjacency(ctx, source, edgesSource, edgesTarget, targetSize, nil)
			outputWithDegree := poolMessagesWithAdjacency(ctx, source, edgesSource, edgesTarget, targetSize, degree)
			inputs = []*Node{source, edgesSource, edgesTarget, degree}
			outputs = []*Node{outputNoDegree, outputWithDegree}
			return
		}, []any{
			[][]float32{
				/* Sum | Mean */
				{0, 1, 0, 1},
				{0, 0, 0, 0},
				{2, 3, 2, 3},
				{6, 8, 3, 4},
			},
			[][]float32{
				{0, 1, 0, 1},
				{0, 0, 0, 0},
				{20, 30, 2, 3},
				{300, 400, 3, 4},
			},
		}, xslices.Epsilon)
}
