// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	. "github.com/gomlx/gomlx/core/graph"
)

// SoftCap performs a smooth (soft) capping/limiting of the input values within the range [-cap, cap]
// using the hyperbolic tangent function:
//
//	output = cap * tanh(input / cap)
//
// This function is commonly used in Gemma models (Gemma-2, Gemma-3, and Gemma-4) to soft-cap attention scores,
// preventing them from growing too large and stabilizing softmax/gradients.
func SoftCap(input *Node, cap float64) *Node {
	if cap <= 0 {
		return input
	}
	input = DivScalar(input, cap)
	input = Tanh(input)
	return MulScalar(input, cap)
}
