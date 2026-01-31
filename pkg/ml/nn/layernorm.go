// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// LayerNorm performs layer normalization on x over the given axes with the
// specified epsilon. gamma and beta are optional (nil means skip scale/shift).
// mask is optional (nil means no masking). When a mask is provided, the
// decomposed path is always used (fused ops don't support masking).
//
// If the backend supports fused LayerNorm, the optimized native implementation
// is used; otherwise the operation is decomposed into primitives. Fallback is
// handled automatically via FusedOpCaller.
func LayerNorm(x *Node, axes []int, epsilon float64, gamma, beta, mask *Node) *Node {
	decomposed := func() *Node {
		return layerNormDecomposed(x, axes, epsilon, gamma, beta, mask)
	}

	// Only try fused when there's no mask (fused ops don't support masking).
	if mask == nil {
		return FusedOpCaller(
			func() *Node { return FusedLayerNorm(x, axes, epsilon, gamma, beta) },
			func() *Node { return layerNormDecomposed(x, axes, epsilon, gamma, beta, nil) },
		)
	}

	return decomposed()
}

// layerNormDecomposed implements LayerNorm using primitive graph ops.
func layerNormDecomposed(x *Node, axes []int, epsilon float64, gamma, beta, mask *Node) *Node {
	var mean *Node
	if mask == nil {
		mean = ReduceAndKeep(x, ReduceMean, axes...)
	} else {
		mean = MaskedReduceAndKeep(x, mask, MaskedReduceMean, axes...)
	}
	xCentered := Sub(x, mean)
	if mask != nil {
		xCentered = Where(mask, xCentered, ZerosLike(xCentered))
	}
	var variance *Node
	if mask == nil {
		variance = ReduceAndKeep(Square(xCentered), ReduceMean, axes...)
	} else {
		variance = MaskedReduceAndKeep(Square(xCentered), mask, MaskedReduceMean, axes...)
	}
	eps := ConstAs(x, epsilon)
	normalized := Div(xCentered, Sqrt(Add(variance, eps)))
	if gamma != nil {
		normalized = Mul(normalized, gamma)
	}
	if beta != nil {
		normalized = Add(normalized, beta)
	}
	return normalized
}
