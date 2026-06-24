// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// FlashAttention computes causal multi-head attention through the backend's fused
// scaled-dot-product-attention op (the cuDNN flash kernel on CUDA), using the flash backward
// automatically when the backend supplies it. The [B,H,S,S] scores never materialize, in either pass.
//
// query is [B,S,nQH,D] and key, value are [B,S,nKVH,D], with nQH divisible by nKVH. Grouped-query
// attention (nKVH < nQH) is handled by repeating each kv head nQH/nKVH times before the fused op, so
// autodiff reduces the repeated-head gradients back onto the kv heads. Output is [B,S,nQH,D] in
// query's dtype. On backends without the fused op (e.g. SimpleGo / CPU) it falls back to a
// decomposed causal attention, differentiated normally.
func FlashAttention(query, key, value *Node, scale float64) *Node {
	for _, n := range []*Node{query, key, value} {
		n.AssertRank(4)
	}
	if !key.Shape().Equal(value.Shape()) {
		Panicf("FlashAttention requires key and value to share shape; got key=%s value=%s", key.Shape(), value.Shape())
	}
	qDims, kvDims := query.Shape().Dimensions, key.Shape().Dimensions
	numQueryHeads, numKVHeads := qDims[2], kvDims[2]
	if qDims[0] != kvDims[0] || qDims[1] != kvDims[1] || qDims[3] != kvDims[3] {
		Panicf("FlashAttention query %s and key/value %s must share [B,S,*,D]", query.Shape(), key.Shape())
	}
	if numKVHeads == 0 || numQueryHeads%numKVHeads != 0 {
		Panicf("FlashAttention requires query heads (%d) divisible by kv heads (%d)", numQueryHeads, numKVHeads)
	}

	// Expand kv heads for GQA so the fused op sees equal q/kv heads. The repeat is a graph op, so
	// autodiff sums the repeated-head gradients back onto the original kv heads.
	k, v := key, value
	if numKVHeads != numQueryHeads {
		group := numQueryHeads / numKVHeads
		k = repeatKVHeads(key, group)
		v = repeatKVHeads(value, group)
	}

	var output *Node
	err := TryCatch[error](func() {
		// BSHD, causal, equal heads. The backend casts to bf16 and runs the cuDNN flash kernel,
		// and attaches the flash backward as the node's custom gradient.
		output = BackendFusedScaledDotProductAttention(
			query, k, v, nil, numQueryHeads, numQueryHeads,
			compute.AxesLayoutBSHD, scale, true /* causal */, nil)
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			return ConvertDType(naiveCausalAttention(query, key, value, scale), query.DType())
		}
		panic(err)
	}
	return ConvertDType(output, query.DType())
}

// naiveCausalAttention is the decomposed reference and fallback: softmax(scale*QK^T + causal)*V
// in float32. query/key/value are [B,S,H,D] with equal heads.
func naiveCausalAttention(query, key, value *Node, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	dims := q.Shape().Dimensions
	batch, seqLen, heads := dims[0], dims[1], dims[2]

	// scores[b,h,q,k] = scale * sum_d query[b,q,h,d]*key[b,k,h,d]
	scores := MulScalar(Einsum("bqhd,bkhd->bhqk", q, k), scale)
	// Causal mask (true = attend), broadcast to the full score shape.
	causal := BroadcastToDims(Reshape(LowerTriangular(g, seqLen), 1, 1, seqLen, seqLen), batch, heads, seqLen, seqLen)
	attn := MaskedSoftmax(scores, causal, -1)
	// out[b,q,h,d] = sum_k attn[b,h,q,k]*value[b,k,h,d]
	return Einsum("bhqk,bkhd->bqhd", attn, v)
}

// repeatKVHeads expands key/value for grouped-query attention: [B,S,nKV,D] -> [B,S,nKV*group,D],
// repeating each kv head group times contiguously, so output head h uses kv head h/group. This
// matches the model's GQA grouping (reshapeQueryForGQA).
func repeatKVHeads(x *Node, group int) *Node {
	d := x.Shape().Dimensions
	b, s, nKV, dim := d[0], d[1], d[2], d[3]
	x = Reshape(x, b, s, nKV, 1, dim)
	x = BroadcastToDims(x, b, s, nKV, group, dim)
	return Reshape(x, b, s, nKV*group, dim)
}
