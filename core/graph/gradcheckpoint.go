package graph

// Additions needed on the existing Node struct (for reference):
// type Node struct { ...
//     isCheckpointed         bool
//     needsRematerialization bool
// }
// - CloneWithInputs(newInputs ...*Node) *Node

// Checkpoint wraps the current node in an optimization barrier, marks it as a
// memory firewall boundary, and kicks off the contagious rematerialization flag.
//
// This should be called at the input boundaries of your checkpointed blocks
// (e.g., at the start of a transformer layer). Downstream nodes will inherit
// the volatile state and be recomputed during the backward pass to save memory.
//
// Example usage for a heavy compute block:
//
//	// 1. Create a checkpoint at every start of a computation:
//	for range numlayers {
//		x = x.Checkpoint()
//		residual := x
//		x = layers.Dense(ctx, x, 1024)
//		x = activations.Gelu(x)
//		// ... more heavy layers ...
//		x = Add(x, residual)
//	}
//
//	// 2. End of the block: stop checkpointing, so the caller of this function
//	//    has a "normal" node (if they don't want to further use checkpointing).
//	return x.StopCheckpoint()
func (n *Node) Checkpoint() *Node {
	// OptimizationBarrier prevents XLA from merging the forward and reverse paths via CSE.
	safeNode := OptimizationBarrier(n)

	// Set the block boundaries
	safeNode.isCheckpointed = true
	safeNode.needsRematerialization = true

	return safeNode
}

// StopCheckpoint wraps the current node in an optimization barrier, marks it as
// a memory firewall boundary, but clears the contagious rematerialization flag.
//
// This isolates downstream layers from unnecessarily duplicating their execution paths.
// Call this exactly once at the final output of your checkpointed block.
//
// See example in [Node.Checkpoint].
func (n *Node) StopCheckpoint() *Node {
	safeNode := OptimizationBarrier(n)

	// Set the block boundaries: save this output in memory, but quarantine the contagion.
	safeNode.isCheckpointed = true
	safeNode.needsRematerialization = false

	return safeNode
}

// rematerializationSweep manages the rematerialization from checkpointed nodes during a backward pass
// (Vector-Jacobian Product extraction).
type rematerializationSweep struct {
	// cache deduplicates rematerialized nodes during this specific backward pass.
	// It ensures fan-outs within volatile blocks don't cause duplicate recomputation.
	cache map[*Node]*Node
}

// newRematerializationSweep initializes a new backward pass state.
// The rematCache is scoped entirely to the lifetime of this sweep, preventing
// global state leaks and keeping independent gradient calculations thread-safe.
func newRematerializationSweep() *rematerializationSweep {
	return &rematerializationSweep{
		cache: make(map[*Node]*Node),
	}
}

// forwardInputs rematerializes the inputs of the given node n, where needed due
// to checkpointing.
//
// The trigger is the node that must be ready (calculated) before rematerialization starts,
// to be used in a `SchedulingBarrier`.
// Usually, this is fed with the `adjoint` value: so rematerialization only happens after the
// first gradients have arrived.
func (s *rematerializationSweep) forwardInputs(n, trigger *Node) []*Node {
	if !n.needsRematerialization {
		return n.Inputs()
	}

	// We are inside a checkpointed block!
	// Lazily reconstruct the inputs on-the-fly, tying their execution to the
	// arrival of this specific adjoint gradient to prevent early memory allocation.
	forwardInputs := make([]*Node, len(n.Inputs()))
	for i, input := range n.Inputs() {
		forwardInputs[i] = s.getRematerialization(input, trigger)
	}
	return forwardInputs
}

// getRematerialization recursively traces backward from node `n` and constructs a parallel,
// cloned recomputation graph.
//
// The `cache` must be provided by the caller (the autodiff sweep) to deduplicate
// nodes during fan-outs and to scope the lifecycle of the cloned nodes to a single backward pass.
func (s *rematerializationSweep) getRematerialization(n *Node, trigger *Node) *Node {
	// Base cases: Stop at the checkpoint(s) or if the input does not needs materialization.
	if n.isCheckpointed {
		return SchedulingBarrier(n, trigger)
	}
	if !n.needsRematerialization {
		return n
	}

	// Cache Hit: Preserve node sharing for fan-outs
	if cached, exists := s.cache[n]; exists {
		return cached
	}

	// Recursive Step: Gather the rematerialized clones of all input dependencies.
	inputs := n.Inputs()
	rematInputs := make([]*Node, len(inputs))
	for i, input := range inputs {
		rematInputs[i] = s.getRematerialization(input, trigger)
	}

	// Clone the current operation node using the new, rematerialized inputs.
	rematNode := n.CloneWithInputs(rematInputs...)

	// Pass the flags along to maintain graph invariants on the cloned nodes.
	rematNode.isCheckpointed = n.isCheckpointed
	rematNode.needsRematerialization = n.needsRematerialization

	// Save to the scoped cache
	s.cache[n] = rematNode
	return rematNode
}
