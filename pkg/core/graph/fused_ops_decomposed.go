// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

// AttentionQKVProjectionDecomposed implements Query-Key-Value projection using primitive ops.
//
// Internal: normal users will want to use MultiHeadAttention instead. This is mostly used
// for converting existing models (that already have fused operations like this).
//
// x: [..., inFeatures]
// wQKV: [inFeatures, queryDim+2*keyValueDim] with Q, K, V weights concatenated along the last axis
// biasQ: [queryDim] (optional, nil for no bias)
// biasK: [keyValueDim] (optional, nil for no bias)
// biasV: [keyValueDim] (optional, nil for no bias)
//
// Returns: [query, key, value] where query=[..., queryDim], key=[..., keyValueDim], value=[..., keyValueDim]
func AttentionQKVProjectionDecomposed(x, wQKV, biasQ, biasK, biasV *Node, queryDim, keyValueDim int) []*Node {
	totalOut := queryDim + 2*keyValueDim

	// Flatten x to 2D if needed for Dot.
	xShape := x.Shape()
	xRank := xShape.Rank()
	inFeat := xShape.Dimensions[xRank-1]
	xBatchSize := xShape.Size() / inFeat
	x2d := x
	if xRank > 2 {
		x2d = Reshape(x, xBatchSize, inFeat)
	}

	// Single matmul: [batch, inFeatures] @ [inFeatures, queryDim+2*keyValueDim] → [batch, totalOut]
	combined := Dot(x2d, wQKV)

	// Slice the combined result into query, key, value along the last axis.
	query := Slice(combined, AxisRange(), AxisRange(0, queryDim))
	key := Slice(combined, AxisRange(), AxisRange(queryDim, queryDim+keyValueDim))
	value := Slice(combined, AxisRange(), AxisRange(queryDim+keyValueDim, totalOut))

	// Reshape back to [..., outDim] if needed.
	if xRank > 2 {
		batchDims := xShape.Dimensions[:xRank-1]
		qDims := append(append([]int{}, batchDims...), queryDim)
		kvDims := append(append([]int{}, batchDims...), keyValueDim)
		query = Reshape(query, qDims...)
		key = Reshape(key, kvDims...)
		value = Reshape(value, kvDims...)
	}

	if biasQ != nil {
		query = Add(query, ExpandLeftToRank(biasQ, query.Rank()))
	}
	if biasK != nil {
		key = Add(key, ExpandLeftToRank(biasK, key.Rank()))
	}
	if biasV != nil {
		value = Add(value, ExpandLeftToRank(biasV, value.Rank()))
	}

	return []*Node{query, key, value}
}
