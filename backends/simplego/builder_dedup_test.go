package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// mockComparableData implements NodeDataComparable for testing.
type mockComparableData struct {
	value int
}

func (m *mockComparableData) Equal(other NodeDataComparable) bool {
	return m.value == other.(*mockComparableData).value
}

// mockNonComparableData does NOT implement NodeDataComparable.
type mockNonComparableData struct {
	value int
}

func TestNodesEqual(t *testing.T) {
	b := &Builder{}
	shape := shapes.Make(dtypes.F32, 2, 3)

	node1 := b.newNode(backends.OpTypeAdd, shape)
	node2 := b.newNode(backends.OpTypeMul, shape)
	node3 := b.newNode(backends.OpTypeSub, shape)

	tests := []struct {
		name string
		a, b []*Node
		want bool
	}{
		{"both empty", nil, nil, true},
		{"both empty slices", []*Node{}, []*Node{}, true},
		{"nil vs empty", nil, []*Node{}, true},
		{"same single node", []*Node{node1}, []*Node{node1}, true},
		{"different single node", []*Node{node1}, []*Node{node2}, false},
		{"same multiple nodes", []*Node{node1, node2}, []*Node{node1, node2}, true},
		{"different order", []*Node{node1, node2}, []*Node{node2, node1}, false},
		{"different lengths", []*Node{node1}, []*Node{node1, node2}, false},
		{"three nodes same", []*Node{node1, node2, node3}, []*Node{node1, node2, node3}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nodesEqual(tt.a, tt.b); got != tt.want {
				t.Errorf("nodesEqual() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMakeNodeDedupKey(t *testing.T) {
	b := &Builder{}
	shape := shapes.Make(dtypes.F32, 2, 3)

	node1 := b.newNode(backends.OpTypeAdd, shape)
	node2 := b.newNode(backends.OpTypeMul, shape)

	tests := []struct {
		name       string
		opType     backends.OpType
		inputs     []*Node
		wantCount  int
		wantHasPtr bool // whether firstInput should be non-zero
	}{
		{"no inputs", backends.OpTypeConstant, nil, 0, false},
		{"empty inputs", backends.OpTypeConstant, []*Node{}, 0, false},
		{"one input", backends.OpTypeNeg, []*Node{node1}, 1, true},
		{"two inputs", backends.OpTypeAdd, []*Node{node1, node2}, 2, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key := makeNodeDedupKey(tt.opType, tt.inputs)

			if key.opType != tt.opType {
				t.Errorf("opType = %v, want %v", key.opType, tt.opType)
			}
			if key.inputCount != tt.wantCount {
				t.Errorf("inputCount = %v, want %v", key.inputCount, tt.wantCount)
			}
			if tt.wantHasPtr && key.firstInput == 0 {
				t.Error("firstInput should be non-zero")
			}
			if !tt.wantHasPtr && key.firstInput != 0 {
				t.Error("firstInput should be zero")
			}
		})
	}

	// Verify same inputs produce same key
	key1 := makeNodeDedupKey(backends.OpTypeAdd, []*Node{node1, node2})
	key2 := makeNodeDedupKey(backends.OpTypeAdd, []*Node{node1, node2})
	if key1 != key2 {
		t.Error("same inputs should produce identical keys")
	}

	// Verify different first input produces different key
	key3 := makeNodeDedupKey(backends.OpTypeAdd, []*Node{node2, node1})
	if key1 == key3 {
		t.Error("different first input should produce different key")
	}
}

func TestRegisterAndFindDuplicateNode(t *testing.T) {
	b := &Builder{}
	shape := shapes.Make(dtypes.F32, 2, 3)

	// Create some input nodes
	input1 := b.newNode(backends.OpTypeParameter, shape)
	input2 := b.newNode(backends.OpTypeParameter, shape)

	t.Run("find returns nil on empty builder", func(t *testing.T) {
		result := b.findDuplicateNode(backends.OpTypeAdd, []*Node{input1, input2}, nil)
		if result != nil {
			t.Error("expected nil on empty dedup map")
		}
	})

	// Create and register a node with nil data
	addNode := b.newNode(backends.OpTypeAdd, shape, input1, input2)
	b.registerForDeduplication(addNode)

	t.Run("find exact match with nil data", func(t *testing.T) {
		result := b.findDuplicateNode(backends.OpTypeAdd, []*Node{input1, input2}, nil)
		if result != addNode {
			t.Errorf("expected to find registered node, got %v", result)
		}
	})

	t.Run("no match for different opType", func(t *testing.T) {
		result := b.findDuplicateNode(backends.OpTypeMul, []*Node{input1, input2}, nil)
		if result != nil {
			t.Error("should not find node with different opType")
		}
	})

	t.Run("no match for different inputs", func(t *testing.T) {
		result := b.findDuplicateNode(backends.OpTypeAdd, []*Node{input2, input1}, nil)
		if result != nil {
			t.Error("should not find node with different input order")
		}
	})

	t.Run("no match for different input count", func(t *testing.T) {
		result := b.findDuplicateNode(backends.OpTypeAdd, []*Node{input1}, nil)
		if result != nil {
			t.Error("should not find node with different input count")
		}
	})

	// Test with comparable data
	t.Run("find match with comparable data", func(t *testing.T) {
		nodeWithData := b.newNode(backends.OpTypeReshape, shape, input1)
		nodeWithData.data = &mockComparableData{value: 100}
		b.registerForDeduplication(nodeWithData)

		// Should find with equal data
		result := b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 100})
		if result != nodeWithData {
			t.Error("should find node with equal comparable data")
		}

		// Should not find with different data
		result = b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 999})
		if result != nil {
			t.Error("should not find node with different data value")
		}
	})

	// Test with non-comparable data (should never match)
	t.Run("no match with non-comparable data", func(t *testing.T) {
		nodeWithData := b.newNode(backends.OpTypeSlice, shape, input1)
		nodeWithData.data = &mockNonComparableData{value: 50}
		b.registerForDeduplication(nodeWithData)

		// Should NOT find even with "same" data since it's not comparable
		result := b.findDuplicateNode(backends.OpTypeSlice, []*Node{input1}, &mockNonComparableData{value: 50})
		if result != nil {
			t.Error("should not find node with non-comparable data")
		}
	})
}

func TestMultipleCandidatesWithSameKey(t *testing.T) {
	b := &Builder{}
	shape := shapes.Make(dtypes.F32, 2, 3)

	input1 := b.newNode(backends.OpTypeParameter, shape)

	// Create multiple nodes with same opType and first input but different data
	node1 := b.newNode(backends.OpTypeReshape, shape, input1)
	node1.data = &mockComparableData{value: 1}
	b.registerForDeduplication(node1)

	node2 := b.newNode(backends.OpTypeReshape, shape, input1)
	node2.data = &mockComparableData{value: 2}
	b.registerForDeduplication(node2)

	node3 := b.newNode(backends.OpTypeReshape, shape, input1)
	node3.data = &mockComparableData{value: 3}
	b.registerForDeduplication(node3)

	// Should find the correct node based on data
	result := b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 2})
	if result != node2 {
		t.Errorf("expected node2, got %v", result)
	}

	result = b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 1})
	if result != node1 {
		t.Errorf("expected node1, got %v", result)
	}

	result = b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 3})
	if result != node3 {
		t.Errorf("expected node3, got %v", result)
	}

	// Should not find non-existent data
	result = b.findDuplicateNode(backends.OpTypeReshape, []*Node{input1}, &mockComparableData{value: 999})
	if result != nil {
		t.Error("should not find non-existent data value")
	}
}

func TestFinalizeCleanup(t *testing.T) {
	b := &Builder{}
	shape := shapes.Make(dtypes.F32, 2, 3)

	input := b.newNode(backends.OpTypeParameter, shape)
	addNode := b.newNode(backends.OpTypeAdd, shape, input, input)
	b.registerForDeduplication(addNode)

	// Verify dedup map exists
	if b.nodeDedup == nil {
		t.Fatal("nodeDedup should be initialized")
	}

	b.Finalize()

	// Verify cleanup
	if b.nodeDedup != nil {
		t.Error("nodeDedup should be nil after Finalize")
	}
}
