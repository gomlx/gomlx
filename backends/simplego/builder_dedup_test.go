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

func (m *mockComparableData) EqualNodeData(other nodeDataComparable) bool {
	return m.value == other.(*mockComparableData).value
}

// mockNonComparableData does NOT implement NodeDataComparable.
type mockNonComparableData struct {
	value int
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
			if tt.wantHasPtr && key.firstInput == nil {
				t.Error("firstInput should be non-nil")
			}
			if !tt.wantHasPtr && key.firstInput != nil {
				t.Error("firstInput should be nil")
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

func TestDedup(t *testing.T) {
	t.Run("BinaryOp", func(t *testing.T) {
		// Create a backend and builder
		be, err := New("")
		if err != nil {
			t.Fatalf("Failed to create backend: %v", err)
		}
		defer be.Finalize()
		builder := be.Builder("test").(*Builder)

		// Create two input parameters
		x, err := builder.Parameter("x", shapes.Make(dtypes.F32, 2, 3), nil)
		if err != nil {
			t.Fatalf("Failed to create parameter x: %v", err)
		}
		y, err := builder.Parameter("y", shapes.Make(dtypes.F32, 2, 3), nil)
		if err != nil {
			t.Fatalf("Failed to create parameter y: %v", err)
		}

		// Create the same Add operation twice
		add1, err := builder.Add(x, y)
		if err != nil {
			t.Fatalf("Failed to create first Add: %v", err)
		}
		add2, err := builder.Add(x, y)
		if err != nil {
			t.Fatalf("Failed to create second Add: %v", err)
		}

		// Verify they are the same node (deduplicated)
		if add1 != add2 {
			t.Errorf("Duplicate Add operations should return the same node: add1=%p, add2=%p", add1, add2)
		}

		// Verify the node count hasn't increased unnecessarily
		// We expect: 2 parameters + 1 Add node = 3 nodes
		if len(builder.nodes) != 3 {
			t.Errorf("Expected 3 nodes (2 params + 1 Add), got %d", len(builder.nodes))
		}
	})

	t.Run("UnaryOp", func(t *testing.T) {
		// Create a backend and builder
		be, err := New("")
		if err != nil {
			t.Fatalf("Failed to create backend: %v", err)
		}
		defer be.Finalize()
		builder := be.Builder("test").(*Builder)

		// Create an input parameter
		x, err := builder.Parameter("x", shapes.Make(dtypes.F32, 2, 3), nil)
		if err != nil {
			t.Fatalf("Failed to create parameter x: %v", err)
		}

		// Create the same Neg operation twice
		neg1, err := builder.Neg(x)
		if err != nil {
			t.Fatalf("Failed to create first Neg: %v", err)
		}
		neg2, err := builder.Neg(x)
		if err != nil {
			t.Fatalf("Failed to create second Neg: %v", err)
		}

		// Verify they are the same node (deduplicated)
		if neg1 != neg2 {
			t.Errorf("Duplicate Neg operations should return the same node: neg1=%p, neg2=%p", neg1, neg2)
		}

		// Verify the node count
		// We expect: 1 parameter + 1 Neg node = 2 nodes
		if len(builder.nodes) != 2 {
			t.Errorf("Expected 2 nodes (1 param + 1 Neg), got %d", len(builder.nodes))
		}
	})

	t.Run("SliceOp", func(t *testing.T) {
		// Create a backend and builder
		be, err := New("")
		if err != nil {
			t.Fatalf("Failed to create backend: %v", err)
		}
		defer be.Finalize()
		builder := be.Builder("test").(*Builder)

		// Create an input parameter
		x, err := builder.Parameter("x", shapes.Make(dtypes.F32, 5, 4), nil)
		if err != nil {
			t.Fatalf("Failed to create parameter x: %v", err)
		}

		// Create the same Slice operation twice with identical parameters
		starts := []int{1, 1}
		limits := []int{3, 3}
		strides := []int{1, 1}

		slice1, err := builder.Slice(x, starts, limits, strides)
		if err != nil {
			t.Fatalf("Failed to create first Slice: %v", err)
		}
		slice2, err := builder.Slice(x, starts, limits, strides)
		if err != nil {
			t.Fatalf("Failed to create second Slice: %v", err)
		}

		// Verify they are the same node (deduplicated)
		if slice1 != slice2 {
			t.Errorf("Duplicate Slice operations should return the same node: slice1=%p, slice2=%p", slice1, slice2)
		}

		// Verify the node count
		// We expect: 1 parameter + 1 Slice node = 2 nodes
		if len(builder.nodes) != 2 {
			t.Errorf("Expected 2 nodes (1 param + 1 Slice), got %d", len(builder.nodes))
		}

		// Verify that different slice parameters create different nodes
		starts2 := []int{2, 2}
		slice3, err := builder.Slice(x, starts2, limits, strides)
		if err != nil {
			t.Fatalf("Failed to create third Slice: %v", err)
		}

		if slice1 == slice3 {
			t.Error("Slice operations with different parameters should create different nodes")
		}
	})
}
