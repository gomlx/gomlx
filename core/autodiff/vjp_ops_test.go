// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package autodiff

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestCheckedSelectAndScatter(t *testing.T) {
	require.NotPanics(t, func() {
		backend := testutil.BuildTestBackend()
		g := graph.NewGraph(backend, "checkedSelectAndScatter()")
		input := graph.IotaFull(g, shapes.Make(dtypes.Float64, 1, 6, 1))
		source := graph.Add(graph.IotaFull(g, shapes.Make(dtypes.Float64, 1, 2, 1)), graph.Const(g, 1.0))
		output := checkedSelectAndScatter(input, source, compute.ReduceOpMax, []int{1, 3, 1}, []int{1, 3, 1}, nil)
		err := g.Compile(input, output)
		require.NoError(t, err)
		outputs := g.Run()
		fmt.Printf("\tcheckedSelectAndScatter()(%s) = %s\n", outputs[0].GoStr(), outputs[1].GoStr())
		wantTensor := tensors.MustFromAnyValue([][][]float64{{{0}, {0}, {1}, {0}, {0}, {2}}})
		require.Truef(t, wantTensor.InDelta(outputs[1], 1e-4), "want=%v, got=%v", wantTensor, outputs[1])
	})
}
