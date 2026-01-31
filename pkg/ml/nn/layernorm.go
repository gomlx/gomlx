// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// LayerNorm performs layer normalization on x over the given axes with the
// specified epsilon. gamma and beta are optional (nil means skip scale/shift).
// mask is optional (nil means no masking). When a mask is provided, the
// decomposed path is always used (fused ops don't support masking).
//
// If the backend supports fused LayerNorm (backends.OpTypeFusedLayerNorm) and
// no mask is provided, the optimized native implementation is used; otherwise
// the operation is decomposed into primitives.
func LayerNorm(x *Node, axes []int, epsilon float64, gamma, beta, mask *Node) *Node {
	// Try native fused LayerNorm when there's no mask.
	if mask == nil && x.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedLayerNorm] {
		return FusedLayerNorm(x, axes, epsilon, gamma, beta)
	}

	// Fall back to decomposition.
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
