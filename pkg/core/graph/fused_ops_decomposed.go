// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

// QKVDenseDecomposed implements QKV projection using primitive ops.
//
// x: [..., inFeatures]
// wQKV: [inFeatures, qDim+2*kvDim] with Q, K, V weights concatenated along the last axis
// biasQ: [qDim] (optional, nil for no bias)
// biasK: [kvDim] (optional, nil for no bias)
// biasV: [kvDim] (optional, nil for no bias)
//
// Returns: [q, k, v] where q=[..., qDim], k=[..., kvDim], v=[..., kvDim]
func QKVDenseDecomposed(x, wQKV, biasQ, biasK, biasV *Node, qDim, kvDim int) []*Node {
	totalOut := qDim + 2*kvDim

	// Flatten x to 2D if needed for Dot.
	xShape := x.Shape()
	xRank := xShape.Rank()
	inFeat := xShape.Dimensions[xRank-1]
	xBatchSize := xShape.Size() / inFeat
	x2d := x
	if xRank > 2 {
		x2d = Reshape(x, xBatchSize, inFeat)
	}

	// Single matmul: [batch, inFeatures] @ [inFeatures, qDim+2*kvDim] → [batch, totalOut]
	combined := Dot(x2d, wQKV)

	// Slice the combined result into Q, K, V along the last axis.
	q := Slice(combined, AxisRange(), AxisRange(0, qDim))
	k := Slice(combined, AxisRange(), AxisRange(qDim, qDim+kvDim))
	v := Slice(combined, AxisRange(), AxisRange(qDim+kvDim, totalOut))

	// Reshape back to [..., outDim] if needed.
	if xRank > 2 {
		batchDims := xShape.Dimensions[:xRank-1]
		qDims := append(append([]int{}, batchDims...), qDim)
		kvDims := append(append([]int{}, batchDims...), kvDim)
		q = Reshape(q, qDims...)
		k = Reshape(k, kvDims...)
		v = Reshape(v, kvDims...)
	}

	if biasQ != nil {
		q = Add(q, ExpandLeftToRank(biasQ, q.Rank()))
	}
	if biasK != nil {
		k = Add(k, ExpandLeftToRank(biasK, k.Rank()))
	}
	if biasV != nil {
		v = Add(v, ExpandLeftToRank(biasV, v.Rank()))
	}

	return []*Node{q, k, v}
}
