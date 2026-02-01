// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"errors"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// FusedSoftmax computes softmax along the specified axis.
// Internal: prefer graph.Softmax which handles fallback and gradients.
func FusedSoftmax(x *Node, axis int) *Node { return backendFusedSoftmax(x, axis) }

// FusedGelu computes GELU activation (exact or approximate).
// Internal: prefer activations.Apply which handles fallback and gradients.
func FusedGelu(x *Node, exact bool) *Node { return backendFusedGelu(x, exact) }

// FusedLayerNorm applies layer normalization. gamma and beta are optional (nil to skip).
// Internal: prefer nn.LayerNorm which handles fallback and gradients.
func FusedLayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	return backendFusedLayerNorm(x, axes, epsilon, gamma, beta)
}

// FusedDense performs fused matmul + optional bias + optional activation.
// Internal: prefer nn.Dense which handles fallback and gradients.
func FusedDense(x, weight, bias *Node, activation backends.ActivationType) *Node {
	return backendFusedDense(x, weight, bias, activation)
}

// FusedOpCaller attempts to call fused, and if it panics with
// backends.ErrUnsupportedDType, falls back to decomposed. Any other panic is
// re-thrown — this prevents real bugs in fused op implementations from being
// silently swallowed.
//
// When the fused call succeeds, the decomposed version is also built and stored
// as the VJP alternate output, so that reverse-mode autodiff can compute
// gradients through the decomposed graph without hand-written VJPs.
func FusedOpCaller(fused, decomposed func() *Node) *Node {
	var output *Node
	err := exceptions.TryCatch[error](func() {
		output = fused()
	})
	if err != nil {
		if errors.Is(err, backends.ErrUnsupportedDType) {
			// Expected: fused op doesn't support this dtype; fall back to decomposed.
			return decomposed()
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	// Store decomposed version for VJP computation; dead-code-eliminated if unused.
	output.vjpAlternateOutput = decomposed()
	return output
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
				Panicf("reverseAutodiffAlternate: node %s has type %q with no gradient defined", node, node.Type())
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
