// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

const margin = 1e-4

func buildTestBackend() compute.Backend {
	compute.DefaultConfig = "xla:cpu"
	return compute.MustNew()
}

type graphFnOneInputToTest func(g *GraphImpl) (input, output *NodeImpl)

func testFuncOneInput(t *testing.T, testName string, graphFn graphFnOneInputToTest, want any) {
	require.NotPanics(t, func() {
		fmt.Printf("%s\n", testName)
		backend := buildTestBackend()
		g := NewGraphImpl(backend, testName)
		input, output := graphFn(g)
		err := g.Compile(input, output)
		require.NoError(t, err)
		outputs := g.Run()
		fmt.Printf("\t%s(%s) = %s\n", testName, outputs[0].GoStr(), outputs[1].GoStr())
		wantTensor := tensors.MustFromAnyValue(want)
		require.Truef(t, wantTensor.InDelta(outputs[1], margin), "%s(%v): want=%v, got=%v", testName, outputs[0], wantTensor, outputs[1])
	})
}

func TestBackendSlice(t *testing.T) {
	backend := buildTestBackend()
	g := NewGraphImpl(backend, "iota0")
	numbers := BackendIota(g, shapes.Make(dtypes.Float64, 9), 0)
	numbers = BackendReshape(numbers, 3, 3)
	slice := BackendSlice(numbers, []int{1, 1}, []int{2, 3}, []int{1, 1})
	err := g.Compile(slice)
	require.NoError(t, err)
	got := g.Run()[0]
	want := [][]float64{{4, 5}}
	require.Equalf(t, want, got.Value(), "Iota: want %v, got %v", want, got)
}

func TestBackendGather(t *testing.T) {
	testFuncOneInput(t, t.Name(), func(g *GraphImpl) (input, output *NodeImpl) {
		input = BackendReshape(BackendIota(g, shapes.Make(dtypes.Float64, 5*3), 0), 5, 3)
		indices := Const(g, [][]int{{2}, {0}})
		output = BackendGather(input, indices, 1,
			/* offsetDims */ []int{1},
			/* collapsed_slice_dims */ []int{0},
			/* start_index_map */ []int{0},
			/* slice_sizes */ []int{1, 3}, false)
		return
	}, [][]float64{{6, 7, 8}, {0, 1, 2}})
}
