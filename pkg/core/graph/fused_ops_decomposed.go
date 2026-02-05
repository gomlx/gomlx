// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"math"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// SDPADecomposed implements multi-head scaled dot-product attention using primitive ops.
//
// q: [batch, numHeads, seqLen, headDim]
// k: [batch, numKVHeads, kvLen, headDim]
// v: [batch, numKVHeads, kvLen, headDim]
// mask: [seqLen, kvLen] (optional additive mask, nil for none)
//
// Returns: [batch, numHeads, seqLen, headDim]
func SDPADecomposed(q, k, v, mask *Node, numHeads, numKVHeads int, scale float64, causal bool) *Node {
	g := q.Graph()

	// Handle GQA: expand K/V heads to match Q heads.
	headsPerKV := numHeads / numKVHeads
	kExpanded := k
	vExpanded := v
	if headsPerKV > 1 {
		kShape := k.Shape()
		batch := kShape.Dimensions[0]
		kvLen := kShape.Dimensions[2]
		headDim := kShape.Dimensions[3]

		kReshaped := Reshape(k, batch, numKVHeads, 1, kvLen, headDim)
		kBroadcast := BroadcastToDims(kReshaped, batch, numKVHeads, headsPerKV, kvLen, headDim)
		kExpanded = Reshape(kBroadcast, batch, numHeads, kvLen, headDim)

		vReshaped := Reshape(v, batch, numKVHeads, 1, kvLen, headDim)
		vBroadcast := BroadcastToDims(vReshaped, batch, numKVHeads, headsPerKV, kvLen, headDim)
		vExpanded = Reshape(vBroadcast, batch, numHeads, kvLen, headDim)
	}

	// scores = Q @ K^T * scale
	// Q: [batch, numHeads, seqLen, headDim], K: [batch, numHeads, kvLen, headDim]
	// Contract headDim (axis 3 of Q with axis 3 of K), batch over axes {0, 1}
	// -> [batch, numHeads, seqLen, kvLen]
	scores := DotGeneral(q, []int{3}, []int{0, 1}, kExpanded, []int{3}, []int{0, 1})
	scores = MulScalar(scores, scale)

	if mask != nil {
		scores = Add(scores, mask)
	}

	if causal {
		seqLen := q.Shape().Dimensions[2]
		kvLen := k.Shape().Dimensions[2]
		iotaShape := shapes.Make(dtypes.Int32, seqLen, kvLen)
		rowIdx := Iota(g, iotaShape, 0)
		colIdx := Iota(g, iotaShape, 1)
		negInf := Scalar(g, q.DType(), math.Inf(-1))
		causalMask := Where(GreaterThan(colIdx, rowIdx),
			negInf,
			ScalarZero(g, q.DType()))
		// Expand causalMask from [seqLen, kvLen] to [1, 1, seqLen, kvLen] for broadcasting
		// with scores [batch, numHeads, seqLen, kvLen].
		causalMask = ExpandLeftToRank(causalMask, scores.Rank())
		scores = Add(scores, causalMask)
	}

	// S = softmax(scores, axis=-1)
	// Uses FusedSoftmax which itself has FusedOpCaller handling.
	S := Softmax(scores, -1)

	// output = S @ V
	// S: [batch, numHeads, seqLen, kvLen], V: [batch, numHeads, kvLen, headDim]
	// Contract kvLen (axis 3 of S with axis 2 of V), batch over axes {0, 1}
	// -> [batch, numHeads, seqLen, headDim]
	output := DotGeneral(S, []int{3}, []int{0, 1}, vExpanded, []int{2}, []int{0, 1})
	return output
}

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
