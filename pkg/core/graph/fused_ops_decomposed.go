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
	inFeatures := wQKV.Shape().Dimensions[0]

	// Slice wQKV into wQ, wK, wV along the last axis.
	wQ := backendSlice(wQKV, []int{0, 0}, []int{inFeatures, qDim}, nil)
	wK := backendSlice(wQKV, []int{0, qDim}, []int{inFeatures, qDim + kvDim}, nil)
	wV := backendSlice(wQKV, []int{0, qDim + kvDim}, []int{inFeatures, qDim + 2*kvDim}, nil)

	// Flatten x to 2D if needed for Dot.
	xShape := x.Shape()
	xRank := xShape.Rank()
	inFeat := xShape.Dimensions[xRank-1]
	xBatchSize := xShape.Size() / inFeat
	x2d := x
	if xRank > 2 {
		x2d = Reshape(x, xBatchSize, inFeat)
	}

	q := Dot(x2d, wQ)
	k := Dot(x2d, wK)
	v := Dot(x2d, wV)

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
