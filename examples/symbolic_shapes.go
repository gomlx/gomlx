package examples

// This file demonstrates symbolic shape inference capabilities.
// These examples show how shape inference handles dynamic dimensions
// using the named axis API.

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// ExampleTransformerShapes demonstrates shape inference for a transformer-style model
// with dynamic batch size and sequence length.
func ExampleTransformerShapes() {
	// Define dimensions using named dynamic axes
	hiddenSize := 768 // Static hidden size (e.g., BERT base)
	numHeads := 12    // Static number of attention heads
	headDim := hiddenSize / numHeads // 64

	// Input embeddings: [batch, seq_len, hidden_size]
	// Use WithDynamicAxis to create named dynamic dimensions
	embeddings := shapes.Make(dtypes.Float32, 0, 0, hiddenSize).
		WithDynamicAxis(0, "batch").
		WithDynamicAxis(1, "seq")

	// Self-attention computation

	// 1. Project to Q, K, V: [batch, seq_len, hidden_size]
	qkvShape := embeddings // Same shape for Q, K, V

	// 2. Reshape to multi-head: [batch, seq_len, num_heads, head_dim]
	batchDim := shapes.DynamicDim
	seqDim := shapes.DynamicDim
	multiHeadShape, _ := shapeinference.ReshapeOp(qkvShape, []int{batchDim, seqDim, numHeads, headDim})

	// 3. Transpose for attention: [batch, num_heads, seq_len, head_dim]
	transposed, _ := shapeinference.TransposeOp(multiHeadShape, []int{0, 2, 1, 3})

	// 4. Attention scores: Q @ K^T
	// Q: [batch, num_heads, seq_len, head_dim]
	// K^T: [batch, num_heads, head_dim, seq_len]
	kTransposed, _ := shapeinference.TransposeOp(transposed, []int{0, 1, 3, 2})

	// Note: Matrix multiply would need specialized handling
	// For now we demonstrate the shape transformations

	// 5. Add positional bias: [1, 1, seq_len, seq_len]
	posBias := shapes.Make(dtypes.Float32, 1, 1, 0, 0).
		WithDynamicAxis(2, "seq").
		WithDynamicAxis(3, "seq")
	// Broadcasting would work: [batch, num_heads, seq_len, seq_len] + [1, 1, seq_len, seq_len]

	// 6. Reduce (e.g., mean pooling): [batch, hidden_size]
	pooled, _ := shapeinference.ReduceOp(embeddings, []int{1}) // Reduce along seq_len

	// Results:
	_ = transposed  // [batch, num_heads, seq_len, head_dim] - all preserved
	_ = kTransposed // [batch, num_heads, head_dim, seq_len] - symbolic dims moved
	_ = posBias     // Can broadcast to attention scores
	_ = pooled      // [batch, hidden_size] - sequence dimension reduced
}

// ExampleBatchProcessing shows how batch dimension flows through operations
func ExampleBatchProcessing() {
	// Input: [batch, 224, 224, 3] (image batch)
	input := shapes.Make(dtypes.Float32, 0, 224, 224, 3).
		WithDynamicAxis(0, "batch")

	// Conv output: [batch, 112, 112, 64]
	convOut := shapes.Make(dtypes.Float32, 0, 112, 112, 64).
		WithDynamicAxis(0, "batch")

	// Batch norm scale: [1, 1, 1, 64]
	bnScale := shapes.Make(dtypes.Float32, 1, 1, 1, 64)

	// Apply batch norm (multiply)
	normalized, _ := shapeinference.BinaryOp(backends.OpTypeMul, convOut, bnScale)
	// Result: [batch, 112, 112, 64] - batch dimension preserved

	// Global average pooling: [batch, 64]
	pooled, _ := shapeinference.ReduceOp(normalized, []int{1, 2}) // Reduce spatial dims

	// Final classification: [batch, num_classes]
	numClasses := 1000
	logits := shapes.Make(dtypes.Float32, 0, numClasses).
		WithDynamicAxis(0, "batch")

	_, _, _, _ = input, normalized, pooled, logits
}

// ExampleVariableSequenceLength demonstrates handling sequences of varying length
func ExampleVariableSequenceLength() {
	embedDim := 512

	// Token IDs: [batch, seq_len]
	tokenIDs := shapes.Make(dtypes.Int32, 0, 0).
		WithDynamicAxis(0, "batch").
		WithDynamicAxis(1, "seq")

	// Embeddings: [batch, seq_len, embed_dim]
	embeddings := shapes.Make(dtypes.Float32, 0, 0, embedDim).
		WithDynamicAxis(0, "batch").
		WithDynamicAxis(1, "seq")

	// Attention mask: [batch, seq_len] (1 for real tokens, 0 for padding)
	mask := shapes.Make(dtypes.Bool, 0, 0).
		WithDynamicAxis(0, "batch").
		WithDynamicAxis(1, "seq")

	// Expand mask for broadcasting: [batch, seq_len, 1]
	expandedMask, _ := shapeinference.ReshapeOp(mask, []int{shapes.DynamicDim, shapes.DynamicDim, 1})

	// Masked embeddings (would use Where operation)
	// Result maintains symbolic dimensions: [batch, seq_len, embed_dim]

	_, _, _ = tokenIDs, embeddings, expandedMask
}

// ExampleMixedStaticDynamic shows mixing static and dynamic dimensions
func ExampleMixedStaticDynamic() {
	// Dynamic batch, static spatial: [batch, 32, 32, 128]
	features := shapes.Make(dtypes.Float32, 0, 32, 32, 128).
		WithDynamicAxis(0, "batch")

	// Static kernel: [3, 3, 128, 256]
	kernel := shapes.Make(dtypes.Float32, 3, 3, 128, 256)

	// Conv would produce: [batch, 30, 30, 256] (with valid padding)
	// The batch dimension flows through, spatial dims are computed statically

	// Per-sample operations work naturally
	// E.g., layer norm over [32, 32, 128] maintains batch dimension

	_, _ = features, kernel
}

// ExampleConcatenation demonstrates concatenation with symbolic dimensions
func ExampleConcatenation() {
	// Two feature tensors from different branches
	branch1 := shapes.Make(dtypes.Float32, 0, 64).
		WithDynamicAxis(0, "batch")
	branch2 := shapes.Make(dtypes.Float32, 0, 128).
		WithDynamicAxis(0, "batch")

	// Concatenate along feature dimension
	concatenated, _ := shapeinference.ConcatenateOp([]shapes.Shape{branch1, branch2}, 1)
	// Result: [batch, 192] - batch preserved, features summed

	// Concatenate along batch dimension
	stacked, _ := shapeinference.ConcatenateOp([]shapes.Shape{branch1, branch1}, 0)
	// Result: [?, 64] - two symbolic batches -> unknown total

	_, _ = concatenated, stacked
}

// ExampleDifferentSymbolicDimensions shows what happens when mixing different symbols
func ExampleDifferentSymbolicDimensions() {
	// Tensor A: [batch, 128]
	tensorA := shapes.Make(dtypes.Float32, 0, 128).
		WithDynamicAxis(0, "batch")

	// Tensor B: [seq, 128] (Different symbolic dimension name!)
	tensorB := shapes.Make(dtypes.Float32, 0, 128).
		WithDynamicAxis(0, "seq")

	// Adding these together
	result, _ := shapeinference.BinaryOp(backends.OpTypeAdd, tensorA, tensorB)
	// Result: [?, 128] - Different axis names, axis name becomes empty
	_ = result
}

// ExampleBroadcastingRules demonstrates the broadcasting hierarchy
func ExampleBroadcastingRules() {
	// Rule 1: Same named dynamic + same named dynamic = same named dynamic
	shape1 := shapes.Make(dtypes.Float32, 0, 10).WithDynamicAxis(0, "batch")
	shape2 := shapes.Make(dtypes.Float32, 0, 10).WithDynamicAxis(0, "batch")
	result1, _ := shapeinference.BinaryOp(backends.OpTypeAdd, shape1, shape2)
	// Result: [?batch, 10] - axis name preserved

	// Rule 2: Dynamic + 1 = dynamic (broadcast)
	shape3 := shapes.Make(dtypes.Float32, 0, 10).WithDynamicAxis(0, "batch")
	shape4 := shapes.Make(dtypes.Float32, 1, 10)
	result2, _ := shapeinference.BinaryOp(backends.OpTypeAdd, shape3, shape4)
	// Result: [?batch, 10]

	// Rule 3: Dynamic + concrete > 1 = concrete
	shape5 := shapes.Make(dtypes.Float32, 0, 10).WithDynamicAxis(0, "batch")
	shape6 := shapes.Make(dtypes.Float32, 32, 10)
	result3, _ := shapeinference.BinaryOp(backends.OpTypeAdd, shape5, shape6)
	// Result: [32, 10] - concrete dimension assumed compatible

	_, _, _ = result1, result2, result3
}
