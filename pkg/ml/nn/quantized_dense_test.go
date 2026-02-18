// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

func TestQuantizedDense_Int8(t *testing.T) {
	// M=2, K=4, N=3, groupSize=3 → numGroups=1, scales [4, 1].
	K, N, groupSize := 4, 3, 3

	xData := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	weightsData := [][]int8{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}
	numGroups := (N + groupSize - 1) / groupSize
	scalesData := make([][]float32, K)
	for k := range K {
		scalesData[k] = make([]float32, numGroups)
		for g := range numGroups {
			scalesData[k][g] = 1.0
		}
	}
	biasData := []float32{0.1, 0.2, 0.3}

	// Expected: x @ float32(weights) + bias
	// x[0]=[1,0,0,0] → y[0]=[1,2,3]+[0.1,0.2,0.3]=[1.1,2.2,3.3]
	// x[1]=[0,1,0,0] → y[1]=[4,5,6]+[0.1,0.2,0.3]=[4.1,5.2,6.3]
	graphtest.RunTestGraphFn(t, "with_bias", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, xData)
		w := Const(g, weightsData)
		s := Const(g, scalesData)
		b := Const(g, biasData)
		y := nn.QuantizedDense(x, w, s, b, backends.QuantInt8, groupSize, N)
		return []*Node{x}, []*Node{y}
	}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)

	graphtest.RunTestGraphFn(t, "no_bias", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, xData)
		w := Const(g, weightsData)
		s := Const(g, scalesData)
		y := nn.QuantizedDense(x, w, s, nil, backends.QuantInt8, groupSize, N)
		return []*Node{x}, []*Node{y}
	}, []any{[][]float32{{1, 2, 3}, {4, 5, 6}}}, 1e-4)

	// All output values are positive, so ReLU acts as identity.
	graphtest.RunTestGraphFn(t, "with_relu", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, xData)
		w := Const(g, weightsData)
		s := Const(g, scalesData)
		b := Const(g, biasData)
		y := nn.QuantizedDense(x, w, s, b, backends.QuantInt8, groupSize, N, activations.TypeRelu)
		return []*Node{x}, []*Node{y}
	}, []any{[][]float32{{1.1, 2.2, 3.3}, {4.1, 5.2, 6.3}}}, 1e-4)
}

func TestQuantizedDense_NF4(t *testing.T) {
	// M=1, K=2, N=4, groupSize=4 → numGroups=1.
	//
	// Nibble indices (low nibble = even col, high nibble = odd col):
	//   row 0: [0, 15, 7, 7] → NF4: [-1.0, 1.0, 0.0, 0.0]
	//   row 1: [15, 0, 7, 7] → NF4: [ 1.0, -1.0, 0.0, 0.0]
	//
	// Packed bytes (low nibble first):
	//   row 0: byte 0: low=0,high=15 → 0xF0; byte 1: low=7,high=7 → 0x77
	//   row 1: byte 0: low=15,high=0 → 0x0F; byte 1: low=7,high=7 → 0x77
	N, groupSize := 4, 4
	packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
	scalesData := [][]float32{{1.0}, {1.0}}
	xData := [][]float32{{1.0, 2.0}}

	// Expected: x @ dequant_weights
	// = 1*[-1, 1, 0, 0] + 2*[1, -1, 0, 0] = [1, -1, 0, 0]
	graphtest.RunTestGraphFn(t, "basic", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, xData)
		packed := Const(g, packedData)
		s := Const(g, scalesData)
		y := nn.QuantizedDense(x, packed, s, nil, backends.QuantNF4, groupSize, N)
		return []*Node{x}, []*Node{y}
	}, []any{[][]float32{{1.0, -1.0, 0.0, 0.0}}}, 1e-4)
}

func TestQuantizedDense_Int4(t *testing.T) {
	// Same packed layout as the NF4 test, but Int4 dequant: (nibble - 8).
	//
	// Nibble indices:
	//   row 0: [0, 15, 7, 7] → Int4: (-8, 7, -1, -1)
	//   row 1: [15, 0, 7, 7] → Int4: ( 7, -8, -1, -1)
	N, groupSize := 4, 4
	packedData := [][]uint8{{0xF0, 0x77}, {0x0F, 0x77}}
	scalesData := [][]float32{{1.0}, {1.0}}
	xData := [][]float32{{1.0, 2.0}}

	// Expected: x @ dequant_weights
	// = 1*[-8, 7, -1, -1] + 2*[7, -8, -1, -1] = [6, -9, -3, -3]
	graphtest.RunTestGraphFn(t, "basic", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, xData)
		packed := Const(g, packedData)
		s := Const(g, scalesData)
		y := nn.QuantizedDense(x, packed, s, nil, backends.QuantInt4, groupSize, N)
		return []*Node{x}, []*Node{y}
	}, []any{[][]float32{{6.0, -9.0, -3.0, -3.0}}}, 1e-4)
}
