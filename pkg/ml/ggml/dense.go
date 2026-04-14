// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// DenseDecomposed performs quantized dense (linear) transformation using graph-level
// dequantization of GGML block-format weights.
//
// It computes y = activation(x @ dequant(weights) + bias).
//
// Parameters:
//   - x: [batch..., K] Float32 input activations.
//   - weights: [N, bytesPerRow] Uint8 in native GGML block layout.
//   - ggmlType: the GGML block format (Q4_0, Q8_0, IQ4_NL).
//   - bias: [N] Float32 (nil for no bias).
//   - activation: optional activation function.
func DenseDecomposed(x, weights *Node, ggmlType backends.GGMLQuantType,
	bias *Node, activation ...activations.Type) *Node {

	act := activations.TypeNone
	if len(activation) > 0 {
		act = activation[0]
	}

	// Dequantize weights → [N, K] Float32.
	N := weights.Shape().Dimensions[0]
	dequantW := Dequantize(weights, ggmlType, N)
	K := dequantW.Shape().Dimensions[1]

	// Flatten x to [M, K] if needed.
	xShape := x.Shape()
	if xShape.Dim(-1) != K {
		exceptions.Panicf("ggml.DenseDecomposed expects x to be shaped [batch..., K] and (decomposed) weights [N, K], "+
			"but got x.shape=%s and weight.shape=%s", xShape, dequantW.Shape())
	}
	x2d := Reshape(x, -1)

	// Tansposed matmul: [M, K] x [N, K] → [M, N]
	y := Dot(x2d, dequantW).EinsumAxes([][2]int{{1, 1}}, nil) // Contract the axis 1 of both sides.

	// Reshape back if x had more than 2 dimensions.
	if xShape.Rank() > 2 {
		outDims := make([]int, xShape.Rank())
		copy(outDims, xShape.Dimensions[:xShape.Rank()-1])
		outDims[xShape.Rank()-1] = N
		y = Reshape(y, outDims...)
	}

	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	if act != activations.TypeNone {
		y = activations.Apply(act, y)
	}
	return y
}
