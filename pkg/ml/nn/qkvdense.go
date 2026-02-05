// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// QKVDense performs a fused QKV projection: a single large matmul followed by
// split into separate Q, K, V outputs with optional per-projection bias.
//
// x: [..., inFeatures]
// wQKV: [inFeatures, qDim+2*kvDim] with Q, K, V weights concatenated along the last axis
// biasQ, biasK, biasV: optional biases (nil for no bias)
// qDim: output dimension for Q projection
// kvDim: output dimension for K and V projections
//
// If the backend supports fused QKVDense, the optimized native implementation is
// used; otherwise the operation is decomposed into primitives. Fallback is
// handled automatically via FusedOpCallerMulti.
func QKVDense(x, wQKV, biasQ, biasK, biasV *Node, qDim, kvDim int) (q, k, v *Node) {
	results := FusedOpCallerMulti(
		func() []*Node {
			q, k, v := FusedQKVDense(x, wQKV, biasQ, biasK, biasV, qDim, kvDim)
			return []*Node{q, k, v}
		},
		func() []*Node {
			return QKVDenseDecomposed(x, wQKV, biasQ, biasK, biasV, qDim, kvDim)
		},
	)
	return results[0], results[1], results[2]
}
