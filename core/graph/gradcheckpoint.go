// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

// Checkpoint wraps the current node in an optimization barrier, marks it as a
// memory firewall boundary, and kicks off the contagious rematerialization flag.
func (n *Node) Checkpoint() *Node {
	safeNode := OptimizationBarrier(n)
	safeNode.impl.SetCheckpointed(true, true)
	return safeNode
}

// StopCheckpoint wraps the current node in an optimization barrier, marks it as
// a memory firewall boundary, but clears the contagious rematerialization flag.
func (n *Node) StopCheckpoint() *Node {
	safeNode := OptimizationBarrier(n)
	safeNode.impl.SetCheckpointed(true, false)
	return safeNode
}
