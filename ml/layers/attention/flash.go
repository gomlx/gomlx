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

// cuDNN flash-attention custom_call targets (XLA lowers these to cuDNN fused MHA).
const (
	fmhaForwardTarget  = "__cudnn$fmhaSoftmax"
	fmhaBackwardTarget = "__cudnn$fmhaSoftmaxBackward"
)

// Layouts are determined by tensor rank, not the concrete dimensions, so they are fixed.
// Operands q,k,v are BSHD [B,S,H,D] with the natural row-major layout [3,2,1,0]; the cuDNN
// output is BHSD [B,H,S,D] (layout [3,1,2,0]); the softmax stats are [B,H,S] ([2,1,0]); the
// scratch workspace is a rank-1 u8 buffer.
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

// flashFwdBackendConfig builds the cudnn_fmha_backend_config for the forward flash attention
// custom_call, for q,k,v shaped [B,S,H,D] (so the materialized score matrix is [B,H,S,S]).
// Captured from JAX 0.10.2 dot_product_attention(cudnn) for a causal mask and verified on a
// cuDNN GPU; only the scale and the score-matrix dimensions vary with shape.
func flashFwdBackendConfig(b, h, s int, scale float64) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s)
}

// flashBwdBackendConfig builds the cudnn_fmha_backend_config for the backward flash attention
// custom_call. Same shape/scale parameterization as the forward; the dot_dimension_numbers
// describe the four backward gemms instead of the two forward ones.
func flashBwdBackendConfig(b, h, s int, scale float64) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", "bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s)
}

// formatScale renders a float as a JSON number (no quotes, shortest round-trip form).
func formatScale(scale float64) string {
	return strconv.FormatFloat(scale, 'g', -1, 64)
}

// FlashAttention computes causal multi-head scaled dot-product attention using the cuDNN
// fused (flash) kernel, with a memory-efficient flash backward supplied as a custom gradient.
//
// query, key, value are [B, S, H, D] (BSHD) and must share the same number of heads H (this is
// the non-grouped path; expand grouped-query KV heads before calling). They are cast to bfloat16
// — the precision the cuDNN flash kernel runs in. The output is [B, S, H, D] bfloat16.
//
// On backends without custom-call support (e.g. SimpleGo), it transparently falls back to a
// decomposed attention that materializes the [B,H,S,S] scores and is differentiated normally.
func FlashAttention(query, key, value *Node, scale float64) *Node {
	for _, n := range []*Node{query, key, value} {
		n.AssertRank(4)
	}
	if !query.Shape().Equal(key.Shape()) || !query.Shape().Equal(value.Shape()) {
		Panicf("FlashAttention requires query/key/value to share shape [B,S,H,D]; got q=%s k=%s v=%s",
			query.Shape(), key.Shape(), value.Shape())
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
		// stats and outBSHD are produced by the forward call below; the VJP closure reads them
		// at gradient-build time (after the forward graph is built), so capturing by reference is
		// safe.
		var stats, outBSHD *Node
		vjpFn := func(node *Node, vjpForOutputs []*Node, _ shapes.Shape) []*Node {
			// vjpForOutputs[0] is the adjoint of the BHSD cuDNN output; cuDNN's backward wants the
			// output gradient in BSHD bfloat16.
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

// naiveCausalAttention is the decomposed reference / fallback: softmax(scale·QKᵀ + causal)·V,
// computed in float32 and returned in float32. query/key/value are [B,S,H,D] with equal heads.
func naiveCausalAttention(query, key, value *Node, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	dims := q.Shape().Dimensions
	batch, seqLen, heads := dims[0], dims[1], dims[2]

	// scores[b,h,i,j] = scale * sum_d q[b,i,h,d]·k[b,j,h,d]
	scores := MulScalar(Einsum("bihd,bjhd->bhij", q, k), scale)
	// Causal mask (true = attend), broadcast to the full [B,H,S,S] score shape.
	causal := BroadcastToDims(Reshape(LowerTriangular(g, seqLen), 1, 1, seqLen, seqLen), batch, heads, seqLen, seqLen)
	attn := MaskedSoftmax(scores, causal, -1)
	// out[b,i,h,d] = sum_j attn[b,h,i,j]·v[b,j,h,d]
	return Einsum("bhij,bjhd->bihd", attn, v)
}
