// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package autodiff

import (
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
)

// rematerializationSweep manages the rematerialization from checkpointed nodes during a backward pass
// (Vector-Jacobian Product extraction).
type rematerializationSweep struct {
	// cache deduplicates rematerialized nodes during this specific backward pass.
	// It ensures fan-outs within volatile blocks don't cause duplicate recomputation.
	cache map[*graph.Node]*graph.Node
}

// newRematerializationSweep initializes a new backward pass state.
// The rematCache is scoped entirely to the lifetime of this sweep, preventing
// global state leaks and keeping independent gradient calculations thread-safe.
func newRematerializationSweep() *rematerializationSweep {
	return &rematerializationSweep{
		cache: make(map[*graph.Node]*graph.Node),
	}
}

// forwardInputs rematerializes the inputs of the given node n, where needed due
// to checkpointing.
//
// The trigger is the node that must be ready (calculated) before rematerialization starts,
// to be used in a `SchedulingBarrier`.
// Usually, this is fed with the `adjoint` value: so rematerialization only happens after the
// first gradients have arrived.
func (s *rematerializationSweep) forwardInputs(n, trigger *graph.Node) []*graph.Node {
	if !graphbridge.NodeImpl(n).NeedsRematerialization() {
		return n.Inputs()
	}

	// We are inside a checkpointed block!
	// Lazily reconstruct the inputs on-the-fly, tying their execution to the
	// arrival of this specific adjoint gradient to prevent early memory allocation.
	forwardInputs := make([]*graph.Node, len(n.Inputs()))
	for i, input := range n.Inputs() {
		forwardInputs[i] = s.getRematerialization(input, trigger)
	}
	return forwardInputs
}

// getRematerialization recursively traces backward from node `n` and constructs a parallel,
// cloned recomputation graph.
func (s *rematerializationSweep) getRematerialization(n *graph.Node, trigger *graph.Node) *graph.Node {
	nImpl := graphbridge.NodeImpl(n)
	// Base cases: Stop at the checkpoint(s) or if the input does not need materialization.
	if nImpl.IsCheckpointed() {
		return graph.SchedulingBarrier(n, trigger)
	}
	if !nImpl.NeedsRematerialization() {
		return n
	}

	// Cache Hit: Preserve node sharing for fan-outs.
	if cached, exists := s.cache[n]; exists {
		return cached
	}

	// Recursive Step: Gather the rematerialized clones of all input dependencies.
	inputs := n.Inputs()
	rematInputs := make([]*graph.Node, len(inputs))
	for i, input := range inputs {
		rematInputs[i] = s.getRematerialization(input, trigger)
	}

	// Clone the current operation node using the new, rematerialized inputs.
	rematImpl := nImpl.CloneWithInputs(graphbridge.NodeImpls(rematInputs)...)
	// Pass the flags along to maintain graph invariants on the cloned nodes.
	rematImpl.SetCheckpointed(nImpl.IsCheckpointed(), nImpl.NeedsRematerialization())
	rematNode := graphbridge.Node(rematImpl)

	// Save to the scoped cache.
	s.cache[n] = rematNode
	return rematNode
}
