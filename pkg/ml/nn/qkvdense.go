// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// AttentionQKVProjection performs a fused Query-Key-Value projection: a single large matmul
// followed by split into separate query, key, value outputs with optional per-projection bias.
//
// x: [..., inFeatures]
// wQKV: [inFeatures, queryDim+2*keyValueDim] with Q, K, V weights concatenated along the last axis
// biasQ, biasK, biasV: optional biases (nil for no bias)
// queryDim: output dimension for query projection
// keyValueDim: output dimension for key and value projections
//
// If the backend supports the fused attention QKV projection, the optimized native implementation
// is used; otherwise the operation is decomposed into primitives. Fallback is
// handled automatically via InternalFusedOpCallerMulti.
func AttentionQKVProjection(x, wQKV, biasQ, biasK, biasV *Node, queryDim, keyValueDim int) (query, key, value *Node) {
	results := InternalFusedOpCallerMulti(
		func() []*Node {
			query, key, value := BackendFusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV, queryDim, keyValueDim)
			return []*Node{query, key, value}
		},
		func() []*Node {
			return AttentionQKVProjectionDecomposed(x, wQKV, biasQ, biasK, biasV, queryDim, keyValueDim)
		},
	)
	return results[0], results[1], results[2]
}
