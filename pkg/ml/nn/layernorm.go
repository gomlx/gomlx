// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// LayerNorm performs layer normalization on x over the given axes with the
// specified epsilon. gamma and beta are optional (nil means skip scale/shift).
//
// If the backend supports fused LayerNorm (backends.OpTypeLayerNorm), the
// optimized native implementation is used; otherwise the operation is
// decomposed into primitives.
func LayerNorm(x *graph.Node, axes []int, epsilon float64, gamma, beta *graph.Node) *graph.Node {
	// Try native fused LayerNorm.
	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeLayerNorm] {
		return graph.LayerNorm(x, axes, epsilon, gamma, beta)
	}

	// Fall back to decomposition.
	mean := graph.ReduceAndKeep(x, graph.ReduceMean, axes...)
	xCentered := graph.Sub(x, mean)
	variance := graph.ReduceAndKeep(graph.Square(xCentered), graph.ReduceMean, axes...)
	eps := graph.ConstAs(x, epsilon)
	normalized := graph.Div(xCentered, graph.Sqrt(graph.Add(variance, eps)))
	if gamma != nil {
		normalized = graph.Mul(normalized, gamma)
	}
	if beta != nil {
		normalized = graph.Add(normalized, beta)
	}
	return normalized
}
