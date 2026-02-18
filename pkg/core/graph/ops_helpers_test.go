// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

const margin = 1e-4

// buildTestBackend and sets backends.DefaultConfig to "xla:cpu" -- it can be overwritten by GOMLX_BACKEND environment variable.
func buildTestBackend() backends.Backend {
	backends.DefaultConfig = "xla:cpu"
	return backends.MustNew()
}

type graphFnOneInputToTest func(g *Graph) (input, output *Node)

func testFuncOneInput(t *testing.T, testName string, graphFn graphFnOneInputToTest, want any) {
	require.NotPanics(t, func() {
		fmt.Printf("%s\n", testName)
		backend := buildTestBackend()
		g := NewGraph(backend, testName)
		input, output := graphFn(g)
		g.Compile(input, output)
		outputs := g.Run()
		fmt.Printf("\t%s(%s) = %s\n", testName, outputs[0].GoStr(), outputs[1].GoStr())
		wantTensor := tensors.FromAnyValue(want)
		require.Truef(t, wantTensor.InDelta(outputs[1], margin), "%s(%v): want=%v, got=%v", testName, outputs[0], wantTensor, outputs[1])
	})
}

func TestBackendSlice(t *testing.T) {
	backend := buildTestBackend()
	g := NewGraph(backend, "iota0")
	numbers := Iota(g, shapes.Make(dtypes.Float64, 9), 0)
	numbers = ReshapeWithShape(numbers, shapes.Make(dtypes.Float64, 3, 3))
	slice := backendSlice(numbers, []int{1, 1}, []int{2, 3}, []int{1, 1})
	g.Compile(slice)
	got := g.Run()[0]
	want := [][]float64{{4, 5}}
	require.Equalf(t, want, got.Value(), "Iota: want %v, got %v", want, got)
}

func TestBackendGather(t *testing.T) {
	testFuncOneInput(t, t.Name(), func(g *Graph) (input, output *Node) {
		// input=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
		input = ReshapeWithShape(Iota(g, shapes.Make(dtypes.Float64, 5*3), 0), shapes.Make(dtypes.Float64, 5, 3))
		indices := Const(g, [][]int{{2}, {0}})
		output = backendGather(input, indices, 1,
			/* offsetDims */ []int{1},
			/* collapsed_slice_dims */ []int{0},
			/* start_index_map */ []int{0},
			/* slice_sizes */ []int{1, 3}, false)
		return
	}, [][]float64{{6, 7, 8}, {0, 1, 2}})
}

func TestCheckedSelectAndScatter(t *testing.T) {
	testFuncOneInput(t, "checkedSelectAndScatter()",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(dtypes.Float64, 1, 6, 1))
			source := Add(IotaFull(g, shapes.Make(dtypes.Float64, 1, 2, 1)), Const(g, 1.0))
			output = checkedSelectAndScatter(input, source, backends.ReduceOpMax, []int{1, 3, 1}, []int{1, 3, 1}, nil)
			return
		}, [][][]float64{{{0}, {0}, {1}, {0}, {0}, {2}}})
}
