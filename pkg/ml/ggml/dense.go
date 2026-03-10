// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
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

	xShape := x.Shape()
	N := weights.Shape().Dimensions[0]

	// Dequantize weights → [N, K] Float32.
	dequantW := Dequant(weights, ggmlType, N)

	// Transpose to [K, N] for matmul.
	dequantW = Transpose(dequantW, 0, 1) // [K, N]

	K := dequantW.Shape().Dimensions[0]

	// Flatten x to [M, K] if needed.
	M := xShape.Size() / K
	x2d := x
	if xShape.Rank() > 2 {
		x2d = Reshape(x, M, K)
	}

	// Matmul: [M, K] @ [K, N] → [M, N]
	y := Dot(x2d, dequantW).Product()

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
