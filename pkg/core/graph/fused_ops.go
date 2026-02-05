// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"errors"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

// FusedMultiHeadSDPA computes multi-head scaled dot-product attention.
// Internal: prefer the InternalFusedOpCaller wrapper in layers which handles fallback and gradients.
func FusedMultiHeadSDPA(q, k, v, mask *Node, numHeads, numKVHeads int, scale float64, causal bool) *Node {
	return backendFusedMultiHeadSDPA(q, k, v, mask, numHeads, numKVHeads, scale, causal)
}

// FusedQKVDense performs fused QKV projection.
// Internal: prefer nn.QKVDense which handles fallback and gradients.
func FusedQKVDense(x, wQKV, biasQ, biasK, biasV *Node, qDim, kvDim int) (q, k, v *Node) {
	return backendFusedQKVDense(x, wQKV, biasQ, biasK, biasV, qDim, kvDim)
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
		if errors.Is(err, backends.ErrNotImplemented) {
			// Expected: fused op doesn't support this config; fall back to decomposed.
			return decomposedOutput
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	// Store decomposed version for VJP computation; dead-code-eliminated if unused.
	output.vjpAlternateOutput = decomposedOutput
	return output
}

// FusedOpCallerMulti is the multi-output counterpart of InternalFusedOpCaller.
// It handles fused ops that return multiple outputs (e.g. FusedQKVDense returning q, k, v).
//
// Like InternalFusedOpCaller, the decomposed version is built first so it has lower
// node indices than the fused nodes. When the fused call succeeds, the decomposed
// outputs are stored as vjpAlternateOutputs on the multi-output parent node for
// reverse-mode autodiff.
func FusedOpCallerMulti(fused, decomposed func() []*Node) []*Node {
	decomposedOutputs := decomposed()

	var outputs []*Node
	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if errors.Is(err, backends.ErrNotImplemented) {
			return decomposedOutputs
		}
		panic(err)
	}

	// Find the multi-output parent node via the first split node.
	// The fused function returns split nodes; we need the underlying multi-output node.
	if len(outputs) > 0 {
		firstSplit := outputs[0]
		if splitInputs, ok := firstSplit.inputs.(*nodeInputsSplitNode); ok {
			parent := splitInputs.multiOutputNode
			parent.vjpAlternateOutputs = decomposedOutputs
		}
	}

	return outputs
}

// reverseAutodiffAlternate computes VJPs for the fused node's inputs by running
// reverse-mode autodiff through the decomposed subgraph rooted at alt.
//
// It collects all nodes in the decomposed subgraph (those between alt and the
// fusedInputs), then processes them in reverse topological order using the
// standard VJP dispatch.
func reverseAutodiffAlternate(alt *Node, fusedInputs []*Node, accVJP *Node, outputShape shapes.Shape) []*Node {
	// Build set of fused input node IDs for quick lookup.
	fusedInputSet := make(map[NodeId]int, len(fusedInputs))
	for i, input := range fusedInputs {
		fusedInputSet[input.Id()] = i
	}

	// Collect all nodes in the decomposed subgraph via DFS from alt,
	// stopping at fusedInputs (which are leaves of this subgraph).
	visited := make(map[NodeId]bool)
	var subgraphNodes []*Node
	var collectNodes func(n *Node)
	collectNodes = func(n *Node) {
		if visited[n.Id()] {
			return
		}
		visited[n.Id()] = true
		if _, isFusedInput := fusedInputSet[n.Id()]; isFusedInput {
			return // Stop at fused inputs - they are leaves.
		}
		for _, input := range n.Inputs() {
			collectNodes(input)
		}
		subgraphNodes = append(subgraphNodes, n)
	}
	collectNodes(alt)

	// Initialize VJP accumulation map.
	vjpMap := make(map[NodeId]*Node)
	vjpMap[alt.Id()] = accVJP

	// Process subgraph nodes in reverse order (reverse topological).
	for i := len(subgraphNodes) - 1; i >= 0; i-- {
		node := subgraphNodes[i]
		nodeVJP, ok := vjpMap[node.Id()]
		if !ok || nodeVJP == nil {
			continue // No gradient flows through this node.
		}

		if node.stopGradient {
			continue
		}

		// Find VJP function for this (primitive) node.
		vjpFn := node.customVJP
		if vjpFn == nil {
			var found bool
			vjpFn, found = VJPRegistration[node.Type()]
			if !found {
				exceptions.Panicf("reverseAutodiffAlternate: node %s has type %q with no gradient defined", node, node.Type())
			}
		}

		vjpsForOutputs := []*Node{nodeVJP}
		inputsVJPs := vjpFn(node, vjpsForOutputs, outputShape)

		for ii, input := range node.Inputs() {
			vjp := inputsVJPs[ii]
			if vjp == nil {
				continue
			}
			if existing, ok := vjpMap[input.Id()]; ok && existing != nil {
				vjpMap[input.Id()] = Add(existing, vjp)
			} else {
				vjpMap[input.Id()] = vjp
			}
		}
	}

	// Extract VJPs for the fused node's inputs.
	results := make([]*Node, len(fusedInputs))
	for i, input := range fusedInputs {
		results[i] = vjpMap[input.Id()] // May be nil if no gradient flows to this input.
	}
	return results
}

// reverseAutodiffAlternateMulti computes VJPs for a multi-output fused node's inputs
// by running reverseAutodiffAlternate once per output and accumulating the results.
//
// altOutputs are the decomposed equivalents of each fused output.
// fusedInputs are the original inputs to the fused node.
// vjpsForOutputs holds the VJP for each output (from the split nodes).
func reverseAutodiffAlternateMulti(altOutputs []*Node, fusedInputs []*Node, vjpsForOutputs []*Node, outputShape shapes.Shape) []*Node {
	accumulated := make([]*Node, len(fusedInputs))

	for i, alt := range altOutputs {
		if alt == nil || vjpsForOutputs[i] == nil {
			continue
		}
		partial := reverseAutodiffAlternate(alt, fusedInputs, vjpsForOutputs[i], outputShape)
		for j, vjp := range partial {
			if vjp == nil {
				continue
			}
			if accumulated[j] == nil {
				accumulated[j] = vjp
			} else {
				accumulated[j] = Add(accumulated[j], vjp)
			}
		}
	}

	return accumulated
}
