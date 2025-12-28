package simplego

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
)

// Dedup implementation: remove duplicated expressions, also known as "common subexpression elimination".

// nodeDataComparable is implemented by node data types that support de-duplication.
// Implementing this interface allows the Builder to automatically de-duplicate
// nodes with matching inputs and equivalent data.
type nodeDataComparable interface {
	// Equal returns true if this data is semantically equivalent to other.
	// The other parameter is guaranteed to be the same concrete type.
	Equal(other nodeDataComparable) bool
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

// findDuplicateNode searches for an existing node that matches the given parameters.
// Returns nil if no duplicate is found.
//
// The search process:
//  1. Look up candidates by (opType, input count, first input pointer)
//  2. For each candidate, verify all inputs match exactly
//  3. If data implements NodeDataComparable, compare data; otherwise require nil data
func (b *Builder) findDuplicateNode(opType backends.OpType, inputs []*Node, data any) *Node {
	if b.nodeDedup == nil {
		return nil
	}

	key := makeNodeDedupKey(opType, inputs)
	candidates := b.nodeDedup[key]

	for _, candidate := range candidates {
		if !nodesEqual(candidate.inputs, inputs) {
			continue
		}

		if dataEqual(candidate.data, data) {
			return candidate
		}
	}

	return nil
}

// registerForDeduplication adds a node to the de-duplication index.
// Only nodes with data implementing NodeDataComparable (or nil data) should be registered.
func (b *Builder) registerForDeduplication(node *Node) {
	if b.nodeDedup == nil {
		b.nodeDedup = make(map[nodeDedupKey][]*Node)
	}

	key := makeNodeDedupKey(node.opType, node.inputs)
	b.nodeDedup[key] = append(b.nodeDedup[key], node)
}

// nodesEqual checks if two slices of nodes are equal (same pointers).
func nodesEqual(a, b []*Node) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// dataEqual compares node data for equality.
// Handles nil, NodeDataComparable, and uncomparable data.
func dataEqual(a, b any) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Both must be the same concrete type
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false
	}

	// If data implements NodeDataComparable, use that
	if comparable, ok := a.(nodeDataComparable); ok {
		return comparable.Equal(b.(nodeDataComparable))
	}

	// For non-comparable data, don't de-duplicate
	return false
}
