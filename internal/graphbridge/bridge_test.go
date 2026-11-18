// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphbridge_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestConstBuffer(t *testing.T) {
	backend := testutil.BuildTestBackend()
	shape := shapes.Make(dtypes.Float32, 2, 2)

	for _, donate := range []bool{false, true} {
		g := NewGraph(backend, "")
		buf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shape)
		require.NoError(t, err)
		c := graphbridge.ConstBuffer(g, buf, donate)
		p := Parameter(g, "x", shape)
		out := Add(c, p)
		require.NoError(t, g.Compile(out))

		in := tensors.FromFlatDataAndDimensions([]float32{10, 20, 30, 40}, 2, 2)
		got := g.Run(in)[0]
		require.Equal(t, [][]float32{{11, 22}, {33, 44}}, got.Value())
		g.Finalize()
		if !donate {
			require.NoError(t, buf.Finalize())
		}
	}
}

func TestBatchNormForInference(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BatchNormInference()",
		func(g *Graph) (inputs, outputs []*Node) {
			input := Iota(g, shapes.Make(dtypes.Float32, 7, 3), 0)
			inputs = []*Node{input}
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{10.0, 100.0, 1000.0})
			mean := Const(g, []float32{0.5, 0.5, 1.0})
			variance := Const(g, []float32{1.0, 1.0, 10.0})
			output := graphbridge.BatchNormForInference(input, scale, offset, mean, variance, 1e-7, -1)
			outputs = []*Node{output}
			return
		}, []any{[][]float32{
			{9.5, 99, 999.05133},
			{10.5, 101, 1000},
			{11.5, 103, 1000.94867},
			{12.5, 105, 1001.8974},
			{13.5, 107, 1002.84607},
			{14.5, 109, 1003.79474},
			{15.5, 111, 1004.7434},
		}}, 1e-4)
}

func TestBatchNormForTraining(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BatchNormTraining()",
		func(g *Graph) (inputs, outputs []*Node) {
			input := Iota(g, shapes.Make(dtypes.Float32, 7, 3), 0)
			inputs = []*Node{input}
			scale := Const(g, []float32{1.0, 2.0, 3.0})
			offset := Const(g, []float32{10.0, 100.0, 1000.0})
			normalized, batchMean, batchVariance := graphbridge.BatchNormForTraining(input, scale, offset, 1e-7, -1)
			normalized2 := graphbridge.BatchNormForInference(input, scale, offset, batchMean, batchVariance, 1e-7, -1)
			outputs = []*Node{normalized, normalized2, batchMean, batchVariance}
			return
		}, []any{
			[][]float32{
				{8.5, 97, 995.5},
				{9, 98, 997},
				{9.5, 99, 998.5},
				{10, 100, 1000},
				{10.5, 101, 1001.5},
				{11, 102, 1003},
				{11.5, 103, 1004.5},
			},
			[][]float32{
				{8.5, 97, 995.5},
				{9, 98, 997},
				{9.5, 99, 998.5},
				{10, 100, 1000},
				{10.5, 101, 1001.5},
				{11, 102, 1003},
				{11.5, 103, 1004.5},
			},
			[]float32{3, 3, 3},
			[]float32{4, 4, 4},
		}, 1e-4)
}
