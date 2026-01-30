// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// LayerNorm performs layer normalization on x over the given axes with the
// specified epsilon. gamma and beta are optional (nil means skip scale/shift).
//
// If the backend supports fused LayerNorm (backends.OpTypeFusedLayerNorm), the
// optimized native implementation is used; otherwise the operation is
// decomposed into primitives.
func LayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	// Try native fused LayerNorm.
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedLayerNorm] {
		return FusedLayerNorm(x, axes, epsilon, gamma, beta)
	}

	// Fall back to decomposition.
	mean := ReduceAndKeep(x, ReduceMean, axes...)
	xCentered := Sub(x, mean)
	variance := ReduceAndKeep(Square(xCentered), ReduceMean, axes...)
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
