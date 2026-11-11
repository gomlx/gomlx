// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
)

func TestRuntimeAudioConvolutionAndScatter(t *testing.T) {
	backend, err := gobackend.New("")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Finalize()
	t.Run("convolution geometry", func(t *testing.T) {
		exec := MustNewExec(backend, func(x, length *Node) *Node {
			x = DynamicReshape(x, StaticDim(1), NamedDynamicDim("samples", length), StaticDim(1))
			return Convolve(x, Const(x.Graph(), [][][]float32{{{1}}, {{1}}, {{1}}})).PadSame().Strides(2).Done()
		})
		defer exec.Finalize()
		out := exec.MustExec1([]float32{1, 2, 3, 4, 5}, int64(5))
		defer out.FinalizeAll()
		want := [][][]float32{{{3}, {9}, {9}}}
		if !reflect.DeepEqual(out.Value(), want) {
			t.Fatalf("got %v, want %v", out.Value(), want)
		}
	})
	t.Run("empty runtime scatter", func(t *testing.T) {
		exec := MustNewExec(backend, func(x, length *Node) *Node {
			indices := DynamicIota(x.Graph(), dtypes.Int64, 0, NamedDynamicDim("updates", length))
			updates := ConvertDType(indices, dtypes.Float32)
			return ScatterUpdate(x, ExpandAxes(indices, -1), updates, true, true)
		})
		defer exec.Finalize()
		out := exec.MustExec1([]float32{1, 2, 3}, int64(0))
		defer out.FinalizeAll()
		if !reflect.DeepEqual(out.Value(), []float32{1, 2, 3}) {
			t.Fatalf("got %v", out.Value())
		}
	})
}
