// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml

import (
	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/support/exceptions"
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
func DenseDecomposed(x, weights *Node, ggmlType compute.GGMLQuantType,
	bias *Node, optionalActivation ...activation.Type) *Node {

	act := activation.TypeNone
	if len(optionalActivation) > 0 {
		act = optionalActivation[0]
	}

	// Dequantize weights → [N, K] Float32.
	N := weights.Shape().Dimensions[0]
	dequantW := Dequantize(weights, ggmlType, N)
	K := dequantW.Shape().Dimensions[1]

	// Flatten x to [M, K] if needed, or contract along x's last axis directly.
	xShape := x.Shape()
	if xShape.Dim(-1) != K {
		exceptions.Panicf("ggml.DenseDecomposed expects x to be shaped [batch..., K] and (decomposed) weights [N, K], "+
			"but got x.shape=%s and weight.shape=%s", xShape, dequantW.Shape())
	}
	if xShape.DType != dequantW.DType() {
		dequantW = ConvertDType(dequantW, xShape.DType)
	}

	// Transposed matmul: [batch..., K] x [N, K] → [batch..., N].
	y := DotGeneral(x, []int{xShape.Rank() - 1}, nil, dequantW, []int{1}, nil)

	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	if act != activation.TypeNone {
		y = activation.Apply(act, y)
	}
	return y
}
