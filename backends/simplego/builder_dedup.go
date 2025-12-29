package simplego

import (
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

// getOrCreateNode attempts to find a node with the content (opType, shape, inputs, data).
// If found, it returns the node.
// If not, it creates a new node with the filled fields, and returns found=false.
func (b *Builder) getOrCreateNode(opType backends.OpType, shape shapes.Shape, inputs []*Node, data any) (n *Node, found bool) {
	// Try to find existing node.
	key := makeNodeDedupKey(opType, inputs)
	candidates := b.nodeDedup[key]
	for _, candidate := range candidates {
		if !slices.Equal(candidate.inputs, inputs) {
			continue
		}
		if !candidate.shape.Equal(shape) {
			continue
		}
		if dataEqual(candidate.data, data) {
			return candidate, true
		}
	}

	// Create new node.
	n = b.newNode(opType, shape, inputs...)
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
