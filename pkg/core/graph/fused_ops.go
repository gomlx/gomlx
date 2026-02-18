// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// BackendFusedSoftmax computes softmax along the specified axis.
// Internal: prefer graph.Softmax which handles fallback and gradients.
func BackendFusedSoftmax(x *Node, axis int) *Node { return backendFusedSoftmax(x, axis) }

// BackendFusedGelu computes GELU activation (exact or approximate).
// Internal: prefer activations.Apply which handles fallback and gradients.
func BackendFusedGelu(x *Node, exact bool) *Node { return backendFusedGelu(x, exact) }

// BackendFusedLayerNorm applies layer normalization. gamma and beta are optional (nil to skip).
// Internal: prefer nn.LayerNorm which handles fallback and gradients.
func BackendFusedLayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	return backendFusedLayerNorm(x, axes, epsilon, gamma, beta)
}

// BackendFusedDense performs fused matmul + optional bias + optional activation.
// Internal: prefer nn.Dense which handles fallback and gradients.
func BackendFusedDense(x, weight, bias *Node, activation backends.ActivationType) *Node {
	return backendFusedDense(x, weight, bias, activation)
}

// BackendFusedScaledDotProductAttention computes multi-head scaled dot-product attention.
// Internal: prefer the InternalFusedOpCaller wrapper in layers which handles fallback and gradients.
func BackendFusedScaledDotProductAttention(query, key, value, mask *Node, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) *Node {
	return backendFusedScaledDotProductAttention(query, key, value, mask, numHeads, numKVHeads, axesLayout, scale, causal)
}

// BackendFusedAttentionQKVProjection performs fused Query-Key-Value projection.
// Internal: prefer attention.QKVProjection which handles fallback and gradients.
func BackendFusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV *Node, queryDim, keyValueDim int) (query, key, value *Node) {
	return backendFusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV, queryDim, keyValueDim)
}

// InternalFusedOpCaller attempts to call fused, and if it panics with
// backends.ErrNotImplemented, falls back to the decomposed version. Any other
// panic is re-thrown — this prevents real bugs in fused op implementations from
// being silently swallowed.
//
// Internal: this is used by higher-level wrappers (graph.Softmax, nn.Dense,
// nn.LayerNorm) that pair a fused backend call with a decomposed fallback.
//
// When the fused call succeeds, the decomposed version is also stored as the
// VJP alternate output, so that reverse-mode autodiff can compute gradients
// through the decomposed graph without hand-written VJPs.
func InternalFusedOpCaller(fused, decomposed func() *Node) *Node {
	// Build decomposed output first so it has a lower nodeIdx than the fused
	// node. The gradient loop iterates from output down to 0, so it will
	// visit the fused node first and transfer its VJP to the decomposed
	// output, which is then processed normally.
	decomposedOutput := decomposed()

	var output *Node
	err := exceptions.TryCatch[error](func() {
		output = fused()
	})
	if err != nil {
		if backends.IsNotImplemented(err) {
			// Expected: fused op doesn't support this config; fall back to decomposed.
			return decomposedOutput
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	// Store decomposed version for VJP computation; dead-code-eliminated if unused.
	output.vjpAlternateOutputs = []*Node{decomposedOutput}
	return output
}

// InternalFusedOpCallerMulti is the multi-output counterpart of InternalFusedOpCaller.
// It handles fused ops that return multiple outputs (e.g. FusedAttentionQKVProjection returning q, k, v).
//
// Like InternalFusedOpCaller, the decomposed version is built first so it has lower
// node indices than the fused nodes. When the fused call succeeds, the decomposed
// outputs are stored as vjpAlternateOutputs on the multi-output parent node for
// reverse-mode autodiff.
func InternalFusedOpCallerMulti(fused, decomposed func() []*Node) []*Node {
	decomposedOutputs := decomposed()

	var outputs []*Node
	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if backends.IsNotImplemented(err) {
			return decomposedOutputs
		}
		panic(err)
	}

	// Find the multi-output parent node via the first split node.
	// The fused function returns split nodes; we need the underlying multi-output node.
	if len(outputs) > 0 {
		firstSplit := outputs[0]
		splitInputs, ok := firstSplit.inputs.(*nodeInputsSplitNode)
		if !ok {
			exceptions.Panicf("InternalFusedOpCallerMulti: fused function returned non-split nodes; cannot set vjpAlternateOutputs")
		}
		parent := splitInputs.multiOutputNode
		parent.vjpAlternateOutputs = decomposedOutputs
	}

	return outputs
}
