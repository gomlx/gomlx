// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
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

// getOrCreateNode attempts to find a node with the content (opType, shape, inputs, data).
// If found, it returns the node.
// If not, it creates a new node with the filled fields, and returns found=false.
//
// It also validates that all input nodes belong to this function or one of its ancestors.
// Using nodes from an ancestor function (closure capture) is not yet supported.
func (f *Function) getOrCreateNode(
	opType backends.OpType, shape shapes.Shape, inputs []*Node, data any) (
	n *Node, found bool) {
	// Check that all input nodes belong to this function or an ancestor.
	for i, node := range inputs {
		if node == nil {
			panic(errors.Errorf("getOrCreateNode(%s): input node #%d is nil", opType, i))
		}
		if node.function == nil {
			panic(errors.Errorf("getOrCreateNode(%s): input node #%d has a nil function", opType, i))
		}
		if node.function == f {
			continue // Same function, OK.
		}
		// Check if the node is from an ancestor function (closure capture).
		if f.IsAncestorOf(node.function) {
			// Node is from a child function - this shouldn't happen in normal usage.
			panic(errors.Errorf(
				"getOrCreateNode(%s): input #%d is from a child function scope %q, not from this function %q",
				opType, i, node.function.name, f.name))
		}
		if node.function.IsAncestorOf(f) {
			// Node is from a parent function (closure capture) - not yet supported.
			panic(errors.Errorf(
				"getOrCreateNode(%s): input #%d uses a node from a parent function scope (closure capturing parent values). "+
					"This is not yet supported in the SimpleGo backend. "+
					"Please pass the value as a closure parameter instead. "+
					"If you need this feature, please open an issue at github.com/gomlx/gomlx",
				opType, i))
		}
		// Completely different function branches - this shouldn't happen.
		panic(errors.Errorf(
			"getOrCreateNode(%s): input #%d is from an incompatible function scope %q, not from this function %q",
			opType, i, node.function.name, f.name))
	}

	// Try to find existing node using function-local dedup.
	key := makeNodeDedupKey(opType, inputs)
	candidates := f.nodeDedup[key]
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
	n = f.newNode(opType, shape, inputs...)
	n.data = data
	f.nodeDedup[key] = append(f.nodeDedup[key], n)
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
