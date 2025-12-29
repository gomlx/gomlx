package simplego

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

// createOrGetNode attempts to find a node with the content (opType, shape, inputs, data).
// If found, it returns the node.
// If not, it creates a new node with the filled fields, and returns found=false.
func (b *Builder) createOrGetNode(opType backends.OpType, shape shapes.Shape, inputs []*Node, data any) (n *Node, found bool) {
	// Try to find existing node.
	key := makeNodeDedupKey(opType, inputs)
	candidates := b.nodeDedup[key]
	for _, candidate := range candidates {
		if !nodesEqual(candidate.inputs, inputs) {
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
