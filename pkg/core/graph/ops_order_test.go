// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

func TestSort_Descending(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sort: descending",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{5, 2, 8, 1, 9, 3})
			result := Sort(x, 0, false)
			return nil, []*Node{result}
		},
		[]any{[]float32{9, 8, 5, 3, 2, 1}},
		0,
	)
}

func TestSortFunc_WithComparator(t *testing.T) {
	graphtest.RunTestGraphFn(t, "SortFunc: custom comparator",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{5, 2, 8, 1, 9, 3})

			comparator := NewClosure(g, func(g *Graph) []*Node {
				lhs := Parameter(g, "lhs", shapes.Scalar[float32]())
				rhs := Parameter(g, "rhs", shapes.Scalar[float32]())
				return []*Node{LessThan(lhs, rhs)}
			})

			results := SortFunc(comparator, 0, false, x)
			return nil, []*Node{results[0]}
		},
		[]any{[]float32{1, 2, 3, 5, 8, 9}},
		0,
	)
}

func TestTopK_1D(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TopK: 1D largest",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
			values, indices := TopK(x, 3, 0)
			return nil, []*Node{values, indices}
		},
		[]any{
			[]float32{9, 6, 5},
			[]int32{5, 7, 4},
		},
		0,
	)
}

func TestTopK_2D(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TopK: 2D along axis 1",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{
				{3, 1, 4},
				{1, 5, 9},
			})
			values, indices := TopK(x, 2, 1)
			return nil, []*Node{values, indices}
		},
		[]any{
			[][]float32{{4, 3}, {9, 5}},
			[][]int32{{2, 0}, {2, 1}},
		},
		0,
	)
}

func TestBottomK_1D(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BottomK: 1D smallest",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
			values, indices := BottomK(x, 3, 0)
			return nil, []*Node{values, indices}
		},
		[]any{
			[]float32{1, 1, 2},
			[]int32{1, 3, 6},
		},
		0,
	)
}

func TestTopK_ArgMax(t *testing.T) {
	// TopK with k=1 can be used as ArgMax
	graphtest.RunTestGraphFn(t, "TopK: argmax behavior",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{
				{3, 1, 4},
				{1, 5, 9},
				{2, 6, 5},
			})
			_, indices := TopK(x, 1, 1)
			return nil, []*Node{indices}
		},
		[]any{
			[][]int32{{2}, {2}, {1}},
		},
		0,
	)
}

func TestBottomK_ArgMin(t *testing.T) {
	// BottomK with k=1 can be used as ArgMin
	graphtest.RunTestGraphFn(t, "BottomK: argmin behavior",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, [][]float32{
				{3, 1, 4},
				{1, 5, 9},
				{2, 6, 5},
			})
			_, indices := BottomK(x, 1, 1)
			return nil, []*Node{indices}
		},
		[]any{
			[][]int32{{1}, {0}, {0}},
		},
		0,
	)
}
