package shapeinference

import (
	"testing"

	. "github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
)

// TestSymbolicDimensionIntegration verifies that symbolic dimension handling works
// across common operations. This tests transformer-style operations with dynamic
// batch and sequence dimensions.
func TestSymbolicDimensionIntegration(t *testing.T) {
	// Example: transformer-style operation
	// Input: [batch, seq_len, hidden_size] where batch and seq_len are dynamic
	batchDim := int(shapes.DimBatch)
	seqDim := int(shapes.DimSeqLen)
	hiddenSize := 512

	// Create input with dynamic batch and sequence length
	input := shapes.MakeDynamic(dtypes.Float32, batchDim, seqDim, hiddenSize)

	// 1. Test binary operations preserve symbolic dimensions
	t.Run("BinaryOps", func(t *testing.T) {
		// Add attention mask: [batch, seq_len, seq_len]
		mask := shapes.MakeDynamic(dtypes.Float32, batchDim, seqDim, seqDim)
		output, err := BinaryOp(OpTypeAdd, input, mask)
		require.NoError(t, err)
		// Should preserve symbolic dimensions
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, seqDim, output.Dimensions[1])
		// Third dimension: seqDim + hiddenSize -> both concrete, should error
		// Actually, different values so broadcast should fail
		_, err = BinaryOp(OpTypeAdd, input, mask)
		require.NoError(t, err) // Actually seq_len matches, hiddenSize != seq_len would fail
	})

	// 2. Test transpose preserves symbolic dimensions in new positions
	t.Run("Transpose", func(t *testing.T) {
		// Transpose to [seq_len, batch, hidden_size]
		output, err := TransposeOp(input, []int{1, 0, 2})
		require.NoError(t, err)
		require.Equal(t, seqDim, output.Dimensions[0])
		require.Equal(t, batchDim, output.Dimensions[1])
		require.Equal(t, hiddenSize, output.Dimensions[2])
	})

	// 3. Test reshape with symbolic dimensions
	t.Run("Reshape", func(t *testing.T) {
		// Reshape [batch, seq_len, 512] to [batch, seq_len, 8, 64]
		output, err := ReshapeOp(input, []int{batchDim, seqDim, 8, 64})
		require.NoError(t, err)
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, seqDim, output.Dimensions[1])
		require.Equal(t, 8, output.Dimensions[2])
		require.Equal(t, 64, output.Dimensions[3])
	})

	// 4. Test reduce operations remove symbolic dimensions
	t.Run("Reduce", func(t *testing.T) {
		// Reduce along sequence dimension: [batch, seq_len, hidden] -> [batch, hidden]
		output, err := ReduceOp(input, []int{1})
		require.NoError(t, err)
		require.Equal(t, 2, output.Rank())
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, hiddenSize, output.Dimensions[1])
	})

	// 5. Test concatenation with symbolic dimensions
	t.Run("Concatenate", func(t *testing.T) {
		// Concatenate two inputs along hidden dimension
		input2 := shapes.MakeDynamic(dtypes.Float32, batchDim, seqDim, 256)
		output, err := ConcatenateOp([]shapes.Shape{input, input2}, 2)
		require.NoError(t, err)
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, seqDim, output.Dimensions[1])
		require.Equal(t, 512+256, output.Dimensions[2]) // Static concat
	})

	// 6. Test comparison operations return Bool with symbolic dims preserved
	t.Run("Comparison", func(t *testing.T) {
		input2 := shapes.MakeDynamic(dtypes.Float32, batchDim, seqDim, hiddenSize)
		output, err := ComparisonOp(OpTypeGreaterThan, input, input2)
		require.NoError(t, err)
		require.Equal(t, dtypes.Bool, output.DType)
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, seqDim, output.Dimensions[1])
		require.Equal(t, hiddenSize, output.Dimensions[2])
	})

	// 7. Test broadcasting rules with symbolic dimensions
	t.Run("Broadcasting", func(t *testing.T) {
		// Broadcast bias: [1, 1, hidden_size] against [batch, seq_len, hidden_size]
		bias := shapes.Make(dtypes.Float32, 1, 1, hiddenSize)
		output, err := BinaryOp(OpTypeAdd, input, bias)
		require.NoError(t, err)
		// Symbolic dimensions should be preserved
		require.Equal(t, batchDim, output.Dimensions[0])
		require.Equal(t, seqDim, output.Dimensions[1])
		require.Equal(t, hiddenSize, output.Dimensions[2])
	})
}

// TestSymbolicDimensionEdgeCases tests boundary conditions for symbolic dimensions.
func TestSymbolicDimensionEdgeCases(t *testing.T) {
	t.Run("DifferentSymbolicDims", func(t *testing.T) {
		// Different symbolic dimensions should become unknown
		shape1 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimBatch), 10)
		shape2 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimSeqLen), 10)
		output, err := BinaryOp(OpTypeAdd, shape1, shape2)
		require.NoError(t, err)
		require.Equal(t, int(shapes.DimUnknown), output.Dimensions[0])
		require.Equal(t, 10, output.Dimensions[1])
	})

	t.Run("SymbolicVsConcreteGreaterThan1", func(t *testing.T) {
		// Symbolic vs concrete > 1: concrete wins (assumed compatible)
		shape1 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimBatch), 10)
		shape2 := shapes.Make(dtypes.Float32, 32, 10)
		output, err := BinaryOp(OpTypeAdd, shape1, shape2)
		require.NoError(t, err)
		require.Equal(t, 32, output.Dimensions[0])
		require.Equal(t, 10, output.Dimensions[1])
	})

	t.Run("SymbolicWith1Broadcasts", func(t *testing.T) {
		// Symbolic with dimension 1 should preserve symbolic
		shape1 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimBatch), 10)
		shape2 := shapes.Make(dtypes.Float32, 1, 10)
		output, err := BinaryOp(OpTypeAdd, shape1, shape2)
		require.NoError(t, err)
		require.Equal(t, int(shapes.DimBatch), output.Dimensions[0])
		require.Equal(t, 10, output.Dimensions[1])
	})

	t.Run("ConcatSymbolicDims", func(t *testing.T) {
		// Concatenating symbolic dimensions results in unknown
		shape1 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimBatch), 10)
		shape2 := shapes.MakeDynamic(dtypes.Float32, int(shapes.DimBatch), 10)
		output, err := ConcatenateOp([]shapes.Shape{shape1, shape2}, 0)
		require.NoError(t, err)
		require.Equal(t, int(shapes.DimUnknown), output.Dimensions[0])
		require.Equal(t, 10, output.Dimensions[1])
	})
}

// TestReshapeSymbolicValidation tests reshape with symbolic dimensions.
func TestReshapeSymbolicValidation(t *testing.T) {
	batch := int(shapes.DimBatch)

	t.Run("SymbolicToSymbolicWithMatchingStaticSizes", func(t *testing.T) {
		// Valid: static portions match (512 == 256 * 2)
		input := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		output, err := ReshapeOp(input, []int{batch, 256, 2})
		require.NoError(t, err)
		require.Equal(t, []int{batch, 256, 2}, output.Dimensions)
	})

	t.Run("SymbolicToSymbolicWithMismatchedStaticSizes", func(t *testing.T) {
		// Invalid: static portions don't match (512 != 256)
		input := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		_, err := ReshapeOp(input, []int{batch, 256})
		require.Error(t, err)
		require.Contains(t, err.Error(), "static size mismatch")
	})

	t.Run("StaticToSymbolicDivisible", func(t *testing.T) {
		// Valid: 1024 is divisible by 256
		input := shapes.Make(dtypes.Float32, 1024)
		output, err := ReshapeOp(input, []int{batch, 256})
		require.NoError(t, err)
		require.Equal(t, []int{batch, 256}, output.Dimensions)
	})

	t.Run("StaticToSymbolicNotDivisible", func(t *testing.T) {
		// Invalid: 1000 is not divisible by 256
		input := shapes.Make(dtypes.Float32, 1000)
		_, err := ReshapeOp(input, []int{batch, 256})
		require.Error(t, err)
		require.Contains(t, err.Error(), "not divisible")
	})

	t.Run("SymbolicToStaticDivisible", func(t *testing.T) {
		// Valid: 1024 is divisible by 512
		input := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		output, err := ReshapeOp(input, []int{2, 512})
		require.NoError(t, err)
		require.Equal(t, []int{2, 512}, output.Dimensions)
	})

	t.Run("SymbolicToStaticNotDivisible", func(t *testing.T) {
		// Invalid: 1000 is not divisible by 512
		input := shapes.MakeDynamic(dtypes.Float32, batch, 500)
		_, err := ReshapeOp(input, []int{2, 512})
		require.Error(t, err)
		require.Contains(t, err.Error(), "not divisible")
	})

	t.Run("FullySymbolicReshape", func(t *testing.T) {
		// Valid: both have only symbolic dimensions, trivial static size (1)
		input := shapes.MakeDynamic(dtypes.Float32, batch)
		output, err := ReshapeOp(input, []int{batch, 1})
		require.NoError(t, err)
		require.Equal(t, []int{batch, 1}, output.Dimensions)
	})
}

// TestGatherSymbolicDimensions tests gather with symbolic dimensions.
func TestGatherSymbolicDimensions(t *testing.T) {
	batch := int(shapes.DimBatch)
	seqLen := int(shapes.DimSeqLen)

	t.Run("SymbolicOperandDimension", func(t *testing.T) {
		// Should not error when operand has symbolic dimension
		operand := shapes.MakeDynamic(dtypes.Float32, batch, 512, 768)
		startIndices := shapes.Make(dtypes.Int32, 10, 2) // 10 indices, 2D coordinates
		sliceSizes := []int{1, 256, 768}                 // Slice smaller than operand

		output, err := Gather(
			operand, startIndices,
			1,            // indexVectorAxis
			[]int{1, 2},  // offsetOutputAxes (axes 1,2 are not collapsed)
			[]int{0},     // collapsedSliceAxes (axis 0 is collapsed)
			[]int{0, 1},  // startIndexMap (maps to operand axes 0,1)
			sliceSizes,
			false, // indicesAreSorted
		)
		require.NoError(t, err)
		require.NotNil(t, output)
	})

	t.Run("SymbolicStartIndicesDimension", func(t *testing.T) {
		// Should not error when startIndices has symbolic dimension
		operand := shapes.Make(dtypes.Float32, 32, 512, 768)
		startIndices := shapes.MakeDynamic(dtypes.Int32, batch, 2) // batch indices, 2D coordinates
		sliceSizes := []int{1, 256, 768}

		output, err := Gather(
			operand, startIndices,
			1,            // indexVectorAxis
			[]int{1, 2},  // offsetOutputAxes
			[]int{0},     // collapsedSliceAxes
			[]int{0, 1},  // startIndexMap
			sliceSizes,
			false,
		)
		require.NoError(t, err)
		// Output should have symbolic dimension from startIndices
		require.Equal(t, batch, output.Dimensions[0])
	})

	t.Run("SymbolicIndexVectorDimension", func(t *testing.T) {
		// Should not error when indexVectorAxis dimension is symbolic
		operand := shapes.Make(dtypes.Float32, 32, 512, 768)
		startIndices := shapes.MakeDynamic(dtypes.Int32, 10, seqLen) // symbolic index vector dimension
		sliceSizes := []int{1, 256, 768}

		// When indexVectorAxis dimension is symbolic, we can't validate startIndexMap length
		// This should still work as the validation is skipped for symbolic dimensions
		output, err := Gather(
			operand, startIndices,
			1,            // indexVectorAxis (symbolic dimension)
			[]int{1, 2},  // offsetOutputAxes
			[]int{0},     // collapsedSliceAxes
			[]int{0, 1},  // startIndexMap - might not match if seqLen != 2, but that's runtime
			sliceSizes,
			false,
		)
		require.NoError(t, err)
		require.NotNil(t, output)
	})
}

// TestBroadcastSymbolicDimensions tests broadcast operations with symbolic dimensions.
func TestBroadcastSymbolicDimensions(t *testing.T) {
	batch := int(shapes.DimBatch)
	seqLen := int(shapes.DimSeqLen)

	t.Run("BroadcastOneToSymbolic", func(t *testing.T) {
		// 1 broadcasts to symbolic
		operand := shapes.Make(dtypes.Float32, 1, 512)
		output := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.NoError(t, err)
	})

	t.Run("BroadcastSymbolicToConcrete", func(t *testing.T) {
		// Symbolic broadcasts to concrete (assumes compatible)
		operand := shapes.MakeDynamic(dtypes.Float32, batch, 1)
		output := shapes.Make(dtypes.Float32, 32, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.NoError(t, err)
	})

	t.Run("BroadcastSymbolicToSymbolic", func(t *testing.T) {
		// Same symbolic dimension
		operand := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		output := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.NoError(t, err)
	})

	t.Run("BroadcastDifferentSymbolicDimensions", func(t *testing.T) {
		// Different symbolic dimensions - should allow (runtime will validate)
		operand := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		output := shapes.MakeDynamic(dtypes.Float32, seqLen, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.NoError(t, err) // Different symbols assumed compatible, validated at runtime
	})

	t.Run("BroadcastConcreteToSymbolic", func(t *testing.T) {
		// Concrete > 1 to symbolic - should allow (runtime will validate)
		operand := shapes.Make(dtypes.Float32, 32, 512)
		output := shapes.MakeDynamic(dtypes.Float32, batch, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.NoError(t, err)
	})

	t.Run("BroadcastIncompatibleConcreteDimensions", func(t *testing.T) {
		// Both concrete but incompatible (not 1 and not matching)
		operand := shapes.Make(dtypes.Float32, 32, 512)
		output := shapes.Make(dtypes.Float32, 64, 512)
		err := BroadcastInDimOp(operand, output, []int{0, 1})
		require.Error(t, err)
		require.Contains(t, err.Error(), "cannot broadcast")
	})
}
