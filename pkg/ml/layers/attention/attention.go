// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// AxesLayout is a type alias for backends.AxesLayout, re-exported for convenience.
type AxesLayout = backends.AxesLayout

const (
	// LayoutBHSD is the [batch, heads, seq, dim] layout.
	LayoutBHSD = backends.AxesLayoutBHSD

	// LayoutBSHD is the [batch, seq, heads, dim] layout.
	LayoutBSHD = backends.AxesLayoutBSHD
)

// scoreEquation returns the Einsum equation for computing attention scores (Q @ K^T).
func scoreEquation(l AxesLayout) string {
	switch l {
	case LayoutBSHD:
		return "bqhd,bkhd->bqhk"
	default: // LayoutBHSD
		return "bhqd,bhkd->bhqk"
	}
}

// outputEquation returns the Einsum equation for computing the attention output (coefficients @ V).
func outputEquation(l AxesLayout) string {
	switch l {
	case LayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default: // LayoutBHSD
		return "bhqk,bhkd->bhqd"
	}
}

// Core computes the core scaled dot-product attention operation.
// This is the shared implementation used by MultiHeadAttention and ONNX op converters.
//
// The layout parameter determines the axis ordering of the input tensors:
//   - LayoutBHSD: [batch, heads, seq, dim] (PyTorch/ONNX convention)
//   - LayoutBSHD: [batch, seq, heads, dim] (MHA convention)
//
// The scale is typically 1/sqrt(head_dim).
//
// The mask parameter (if non-nil) controls which positions can be attended to:
//   - Boolean mask (DType.IsBoolean()): uses MaskedSoftmax — true means attend, false means ignore.
//     This avoids the -1e9 additive trick which causes gradient and low-precision issues.
//   - Float mask: added to scores before softmax (additive mask).
//
// The mask must be broadcastable to the score tensor layout:
//   - LayoutBHSD: broadcastable to [batch, heads, q_seq, kv_seq]
//   - LayoutBSHD: broadcastable to [batch, q_seq, heads, kv_seq]
//
// The causal and mask parameters are mutually exclusive: providing both will panic.
// If you need both causal masking and an explicit mask, combine them into a single mask
// before calling Core (e.g. LogicalAnd a lower-triangular boolean mask with your mask).
//
// The dropoutRate (if > 0) applies dropout to the attention coefficients during training.
// When dropout is active, the fused path is skipped (fused ops don't support dropout).
// The ctx parameter provides the training/inference context for dropout; it may be nil
// when dropoutRate is 0.
//
// Returns:
//   - output: same shape as value.
//   - coefficients: attention coefficients (from 0 to 1) shaped
//     [batch, heads, q_seq, kv_seq] for LayoutBHSD or
//     [batch, q_seq, heads, kv_seq] for LayoutBSHD.
//     Coefficients are always from the decomposed path (useful for visualization).
func Core(ctx *context.Context, query, key, value *Node, scale float64, mask *Node, dropoutRate float64, layout AxesLayout, causal bool) (output, coefficients *Node) {
	g := query.Graph()

	if causal && mask != nil {
		Panicf("attention.Core: causal and mask are mutually exclusive; combine them into a single mask before calling Core")
	}

	// Build causal mask for the decomposed path.
	decomposedMask := mask
	if causal {
		seqLen := query.Shape().Dimensions[layout.SeqAxis()]
		causalBool := LowerTriangular(g, seqLen)
		// Reshape for correct broadcasting with score layout.
		switch layout {
		case LayoutBHSD:
			causalBool = Reshape(causalBool, 1, 1, seqLen, seqLen)
		default: // LayoutBSHD
			causalBool = Reshape(causalBool, 1, seqLen, 1, seqLen)
		}
		decomposedMask = causalBool
	}

	// Decomposed attention.
	scores := Einsum(scoreEquation(layout), query, key)
	scores = MulScalar(scores, scale)

	if decomposedMask != nil && decomposedMask.DType() != dtypes.Bool {
		// Additive float mask.
		scores = Add(scores, decomposedMask)
		coefficients = Softmax(scores, -1)
	} else {
		// Boolean mask (or nil): MaskedSoftmax handles both.
		if decomposedMask != nil {
			decomposedMask = BroadcastToShape(decomposedMask, scores.Shape())
		}
		coefficients = MaskedSoftmax(scores, decomposedMask, -1)
	}

	// Apply dropout; detect whether it was actually applied by comparing nodes.
	var dropoutActive bool
	if ctx != nil {
		newCoefficients := layers.DropoutStatic(ctx, coefficients, dropoutRate)
		dropoutActive = newCoefficients != coefficients
		coefficients = newCoefficients
	}

	decomposedOutput := Einsum(outputEquation(layout), coefficients, value)

	// Try fused SDPA (not when dropout is active during training).
	if dropoutActive {
		output = decomposedOutput
	} else {
		numHeads := query.Shape().Dimensions[layout.HeadsAxis()]
		numKVHeads := key.Shape().Dimensions[layout.HeadsAxis()]
		output = InternalFusedOpCaller(
			func() *Node {
				return BackendFusedScaledDotProductAttention(
					query, key, value, mask,
					numHeads, numKVHeads, layout, scale, causal)
			},
			func() *Node { return decomposedOutput },
		)
	}
	return output, coefficients
}
