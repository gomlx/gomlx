// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Dedup implementation: remove duplicated expressions, also known as "common subexpression elimination".

// nodeDataComparable is implemented by node data types that support de-duplication.
// Implementing this interface allows the Builder to automatically de-duplicate
// nodes with matching inputs and equivalent data.
type nodeDataComparable interface {
	// EqualNodeData returns true if this data is semantically equivalent to other.
	// The other parameter is guaranteed to be the same concrete type.
	EqualNodeData(other nodeDataComparable) bool
}

// nodeDedupKey is used to index into the de-duplication map.
// It provides fast lookup for candidate nodes with the same operation type
// and input structure.
type nodeDedupKey struct {
	opType     backends.OpType
	inputCount int
	firstInput *Node // nil if there are no inputs.
}

// makeNodeDedupKey creates a de-duplication key for a node with the given opType and inputs.
func makeNodeDedupKey(opType backends.OpType, inputs []*Node) nodeDedupKey {
	key := nodeDedupKey{
		opType:     opType,
		inputCount: len(inputs),
	}
	if len(inputs) > 0 {
		key.firstInput = inputs[0]
	}
	return key
}

// innermostFunction finds the "innermost" (deepest) function scope among the inputs.
//
// It returns an error if the scopes are incompatible (i.e., they are not on the same branch of the function tree).
//
// If inputs is empty, it returns nil -- the caller should handle this case (usually by assigning the current function).
func innermostFunction(inputs []*Node) (*Function, error) {
	if len(inputs) == 0 {
		return nil, nil // No inputs, no scope inferred.
	}

	var candidate *Function
	for i, node := range inputs {
		if node == nil {
			return nil, fmt.Errorf("input node #%d is nil", i)
		}
		if node.function == nil {
			return nil, fmt.Errorf("input node #%d has a nil function", i)
		}

		if candidate == nil {
			candidate = node.function
			continue
		}

		other := node.function
		if other == candidate {
			continue
		}

		if candidate.IsAncestorOf(other) {
			// candidate is ancestor of other, so other is deeper.
			// If candidate is ancestor of other, then they are compatible, and other is the new candidate.
			candidate = other
		} else if !other.IsAncestorOf(candidate) {
			// candidate is NOT ancestor of other, AND other is NOT ancestor of candidate.
			// Disjoint branches.
			return nil, fmt.Errorf("incompatible scopes for inputs: scope %q and scope %q are not in the same ancestry line", candidate.name, other.name)
		}
		// else: other is ancestor of candidate, so candidate remains the deeper one.
	}
	return candidate, nil
}

// getOrCreateNode attempts to find a node with the content (opType, shape, inputs, data).
// If found, it returns the node.
// If not, it creates a new node with the filled fields, and returns found=false.
// The function parameter tracks which function this node was created in.
// If f is nil, the function is derived from the inputs (using innermostFunction).
func (b *Builder) getOrCreateNode(f *Function, opType backends.OpType, shape shapes.Shape, inputs []*Node, data any) (n *Node, found bool) {
	// Derive function from inputs if not specified.
	if f == nil {
		var err error
		f, err = innermostFunction(inputs)
		if err != nil {
			// This should never happen if the graph layer validated correctly.
			panic(fmt.Sprintf("getOrCreateNode: %v", err))
		}
		if f == nil {
			// Fallback to mainFn if no inputs (e.g., for constants without inputs).
			f = b.mainFn
		}
	}
	// Try to find existing node.
	key := makeNodeDedupKey(opType, inputs)
	candidates := b.nodeDedup[key]
	for _, candidate := range candidates {
		// Only deduplicate within the same function scope.
		// Deduplicating across functions would cause "different function scope" errors
		// when the node is used in a closure.
		if candidate.function != f {
			continue
		}
		if !slices.Equal(candidate.inputs, inputs) {
			continue
		}
		if !candidate.shape.Equal(shape) {
			continue
		}
		if !dataEqual(candidate.data, data) {
			continue
		}
		return candidate, true
	}

	// Create new node.
	n = b.newNode(f, opType, shape, inputs...)
	n.data = data
	b.nodeDedup[key] = append(b.nodeDedup[key], n)
	return n, false
}

// dataEqual compares node data for equality.
// Handles nil, NodeDataComparable, primitive types (int, []int), and uncomparable data.
func dataEqual(a, b any) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Both must be the same concrete type
	aType := reflect.TypeOf(a)
	bType := reflect.TypeOf(b)
	if aType != bType {
		return false
	}

	// If data implements NodeDataComparable, use that
	if comparable, ok := a.(nodeDataComparable); ok {
		return comparable.EqualNodeData(b.(nodeDataComparable))
	}

	// Handle primitive types
	switch aVal := a.(type) {
	case int:
		return aVal == b.(int)
	case []int:
		return slices.Equal(aVal, b.([]int))
	}

	// For non-comparable data, don't de-duplicate
	return false
}
