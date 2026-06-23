// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"fmt"
	"strconv"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// cuDNN fused-attention (flash) custom_call targets.
const (
	fmhaForwardTarget  = "__cudnn$fmhaSoftmax"
	fmhaBackwardTarget = "__cudnn$fmhaSoftmaxBackward"
)

// Layouts are rank-determined, so fixed: q,k,v BSHD [3,2,1,0], cuDNN output BHSD [3,1,2,0],
// stats [2,1,0], scratch u8 [0].
const (
	layoutBSHD  = "dense<[3, 2, 1, 0]> : tensor<4xindex>"
	layoutBHSD  = "dense<[3, 1, 2, 0]> : tensor<4xindex>"
	layoutStats = "dense<[2, 1, 0]> : tensor<3xindex>"
	layoutU8    = "dense<0> : tensor<1xindex>"

	flashFwdOperandLayouts = "[" + layoutBSHD + ", " + layoutBSHD + ", " + layoutBSHD + "]"
	flashFwdResultLayouts  = "[" + layoutBHSD + ", " + layoutStats + ", " + layoutU8 + "]"
	// Backward operands: q, k, v (BSHD), softmax stats, dOutput (BSHD), output (BSHD).
	flashBwdOperandLayouts = "[" + layoutBSHD + ", " + layoutBSHD + ", " + layoutBSHD + ", " + layoutStats + ", " + layoutBSHD + ", " + layoutBSHD + "]"
	// Backward results: dQ, dK, dV (BHSD), scratch.
	flashBwdResultLayouts = "[" + layoutBHSD + ", " + layoutBHSD + ", " + layoutBHSD + ", " + layoutU8 + "]"
)

// flashFwdBackendConfig builds the forward cudnn_fmha_backend_config for q,k,v [B,S,H,D]
// (score matrix [B,H,S,S]). Captured from JAX dot_product_attention(cudnn), causal; only the
// scale and score-matrix dims vary with shape.
func flashFwdBackendConfig(b, h, s int, scale float64) string {
	return flashBackendConfig(b, h, s, scale,
		`"bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}`)
}

// flashBwdBackendConfig is the backward counterpart: the four backward-gemm dot_dimension_numbers.
func flashBwdBackendConfig(b, h, s int, scale float64) string {
	return flashBackendConfig(b, h, s, scale,
		`"bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}`)
}

// flashBackendConfig builds a cudnn_fmha_backend_config. fmha_scale and the score-matrix
// dimensions [B,H,S,S] vary with shape; dotDimNumbers carries the bmm dot_dimension_numbers
// JSON, the only part that differs between the forward and backward custom-calls.
func flashBackendConfig(b, h, s int, scale float64, dotDimNumbers string) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", %s, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s, dotDimNumbers)
}

// formatScale renders a float as a JSON number (no quotes, shortest round-trip form).
func formatScale(scale float64) string {
	return strconv.FormatFloat(scale, 'g', -1, 64)
}

// FlashAttention computes causal multi-head attention with the cuDNN flash kernel and a flash
// backward supplied as a custom gradient. Scores [B,H,S,S] never materialize, in either pass.
//
// query is [B,S,nQH,D] and key, value are [B,S,nKVH,D], with nQH divisible by nKVH. Grouped-query
// attention (nKVH < nQH) is handled by repeating each kv head nQH/nKVH times. Inputs are cast to
// bfloat16 (the kernel's precision); output is [B,S,nQH,D] bfloat16. On backends without
// custom-call support it falls back to a decomposed attention, differentiated normally.
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
	if numKVHeads != numQueryHeads {
		group := numQueryHeads / numKVHeads
		key = repeatKVHeads(key, group)
		value = repeatKVHeads(value, group)
	}
	dims := query.Shape().Dimensions
	b, s, h, d := dims[0], dims[1], dims[2], dims[3]

	q := ConvertDType(query, dtypes.BFloat16)
	k := ConvertDType(key, dtypes.BFloat16)
	v := ConvertDType(value, dtypes.BFloat16)

	bhsd := shapes.Make(dtypes.BFloat16, b, h, s, d)
	statsShape := shapes.Make(dtypes.Float32, b, h, s)
	scratch := shapes.Make(dtypes.Uint8, 0)

	var output *Node
	err := TryCatch[error](func() {
		// stats and outBSHD are assigned after the forward call below but read only at gradient
		// time, so capturing by reference is safe.
		var stats, outBSHD *Node
		vjpFn := func(node *Node, vjpForOutputs []*Node, _ shapes.Shape) []*Node {
			// vjpForOutputs[0] is the BHSD output adjoint; the backward wants it BSHD bfloat16.
			dOut := ConvertDType(Transpose(vjpForOutputs[0], 1, 2), dtypes.BFloat16)
			spec := compute.CustomCallSpec{
				Target:         fmhaBackwardTarget,
				APIVersion:     2,
				BackendConfig:  flashBwdBackendConfig(b, h, s, scale),
				OperandLayouts: flashBwdOperandLayouts,
				ResultLayouts:  flashBwdResultLayouts,
				OutputShapes:   []shapes.Shape{bhsd, bhsd, bhsd, scratch},
			}
			grads := CustomCall(spec, nil, q, k, v, stats, dOut, outBSHD)
			// dQ/dK/dV come back BHSD; transpose to BSHD to match q,k,v.
			return []*Node{Transpose(grads[0], 1, 2), Transpose(grads[1], 1, 2), Transpose(grads[2], 1, 2)}
		}
		spec := compute.CustomCallSpec{
			Target:         fmhaForwardTarget,
			APIVersion:     2,
			BackendConfig:  flashFwdBackendConfig(b, h, s, scale),
			OperandLayouts: flashFwdOperandLayouts,
			ResultLayouts:  flashFwdResultLayouts,
			OutputShapes:   []shapes.Shape{bhsd, statsShape, scratch},
		}
		fwd := CustomCall(spec, vjpFn, q, k, v) // [output BHSD, stats, scratch]
		stats = fwd[1]
		outBSHD = Transpose(fwd[0], 1, 2) // BHSD -> BSHD
		output = outBSHD
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			return ConvertDType(naiveCausalAttention(query, key, value, scale), query.DType())
		}
		panic(err)
	}
	return output
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
