// Example demonstrating pattern caching with bucketing strategies
//
// This example shows how pattern caching can reduce the number of compiled graphs
// when working with variable batch sizes.
//
// Run with: go run main.go
package main

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/tensors/bucketing"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

func main() {
	fmt.Println("=== Pattern Caching Example ===")

	// Initialize backend (would need proper setup in real usage)
	// For this example, we'll show the API usage
	showUsageExamples()
}

func showUsageExamples() {
	fmt.Println("1. Without Pattern Caching (Default)")
	fmt.Println("   Each unique batch size creates a new graph:")
	fmt.Println("   - Batch 1 → Graph 1")
	fmt.Println("   - Batch 2 → Graph 2")
	fmt.Println("   - Batch 3 → Graph 3")
	fmt.Println("   - Batch 4 → Graph 4")
	fmt.Println("   - Batch 5 → Graph 5")
	fmt.Println("   Total: 5 graphs")

	fmt.Println("2. With Pow2 Bucketing")
	fmt.Println("   Batch sizes bucket to powers of 2:")
	fmt.Println("   - Batch 1 → Graph[1]   (new)")
	fmt.Println("   - Batch 2 → Graph[2]   (new)")
	fmt.Println("   - Batch 3 → Graph[4]   (new, bucketed)")
	fmt.Println("   - Batch 4 → Graph[4]   (reused)")
	fmt.Println("   - Batch 5 → Graph[8]   (new, bucketed)")
	fmt.Println("   Total: 4 graphs (20% reduction)")

	fmt.Println("3. With Linear Bucketing (step=8)")
	fmt.Println("   All batch sizes 1-8 use the same graph:")
	fmt.Println("   - Batch 1-8 → Graph[8]   (reused)")
	fmt.Println("   - Batch 9-16 → Graph[16] (new)")
	fmt.Println("   Total: Only 2 graphs for 16 batch sizes (87.5% reduction)")

	showAPIExamples()
}

func showAPIExamples() {
	fmt.Println("=== API Examples ===")

	// Example 1: Default behavior
	fmt.Println("// Example 1: Default behavior (no pattern caching)")
	fmt.Println("exec := MustNewExec(backend, graphFn)")
	fmt.Println()

	// Example 2: Pow2 bucketing
	fmt.Println("// Example 2: Enable Pow2 bucketing on batch axis")
	fmt.Println("exec := MustNewExec(backend, graphFn).WithPow2Bucketing()")
	fmt.Println()

	// Example 3: Linear bucketing
	fmt.Println("// Example 3: Enable linear bucketing with step=8")
	fmt.Println("exec := MustNewExec(backend, graphFn).WithLinearBucketing(8)")
	fmt.Println()

	// Example 4: Custom axes
	fmt.Println("// Example 4: Make batch and sequence length dynamic")
	fmt.Println("exec := MustNewExec(backend, encoderFn).")
	fmt.Println("    SetPatternCaching(bucketing.Pow2()).")
	fmt.Println("    SetDynamicAxes([]int{0, 1})")
	fmt.Println()

	// Example 5: Check cache size
	fmt.Println("// Example 5: Monitor cache size")
	fmt.Println("exec := MustNewExec(backend, graphFn).WithPow2Bucketing()")
	fmt.Println("exec.MustExec(batch1)")
	fmt.Println("exec.MustExec(batch2)")
	fmt.Println("fmt.Println(\"Cached graphs:\", exec.CacheSize())")
	fmt.Println()

	showBucketingExamples()
}

func showBucketingExamples() {
	fmt.Println("=== Bucketing Strategy Examples ===")

	fmt.Println("Pow2 Bucketing:")
	pow2 := bucketing.Pow2()
	for _, size := range []int{1, 2, 3, 5, 9, 17, 33} {
		bucketed := pow2.Bucket(size)
		fmt.Printf("  %2d → %2d\n", size, bucketed)
	}
	fmt.Println()

	fmt.Println("Linear Bucketing (step=8):")
	linear := bucketing.Linear(8)
	for _, size := range []int{1, 5, 8, 9, 15, 16, 17} {
		bucketed := linear.Bucket(size)
		fmt.Printf("  %2d → %2d\n", size, bucketed)
	}
	fmt.Println()

	fmt.Println("Exponential Bucketing (base=1.4):")
	exponential := bucketing.Exponential(1.4)
	for _, size := range []int{1, 2, 3, 5, 9, 17, 33} {
		bucketed := exponential.Bucket(size)
		fmt.Printf("  %2d → %2d\n", size, bucketed)
	}
	fmt.Println()

	fmt.Println("None (no bucketing):")
	none := bucketing.None()
	for _, size := range []int{1, 3, 5, 9, 17} {
		bucketed := none.Bucket(size)
		fmt.Printf("  %2d → %2d\n", size, bucketed)
	}
}

// Example graph functions that could be used with pattern caching

// SimpleSumFn sums all elements in the input tensor
func SimpleSumFn(x *Node) *Node {
	return ReduceAllSum(x)
}

// BatchNormalizeFn example of a function that benefits from variable batch sizes
func BatchNormalizeFn(x *Node) *Node {
	// Simplified batch normalization
	mean := ReduceMean(x, 0)
	variance := ReduceMean(Square(Sub(x, mean)), 0)
	normalized := Div(Sub(x, mean), Sqrt(AddScalar(variance, 1e-5)))
	return normalized
}

// TransformerEncoderFn example with variable batch and sequence length
func TransformerEncoderFn(x *Node) *Node {
	// Simplified transformer encoder block
	// Input: [batch, seqlen, features]
	// This would benefit from bucketing on both batch and seqlen axes

	// Self-attention (simplified)
	q := x // Query
	k := x // Key
	v := x // Value

	// Attention scores
	scores := Dot(q, TransposeAllDims(k, -1, -2))
	attention := Softmax(scores, -1)
	attended := Dot(attention, v)

	// Add & Norm (simplified)
	return Add(x, attended)
}

// ConcreteExample shows a real usage scenario
func ConcreteExample() {
	// This would require proper backend setup
	// Shown for documentation purposes

	fmt.Println("\n=== Concrete Usage Scenario ===")
	fmt.Println("// Training loop with variable batch sizes")
	fmt.Println("backend := backends.MustNew()")
	fmt.Println("exec := MustNewExec(backend, modelFn).WithPow2Bucketing()")
	fmt.Println()
	fmt.Println("for epoch := 0; epoch < 100; epoch++ {")
	fmt.Println("    for batch := range dataloader {")
	fmt.Println("        // Batch size might vary (especially last batch)")
	fmt.Println("        predictions := exec.MustExec(batch)")
	fmt.Println("        // ... compute loss and update weights")
	fmt.Println("    }")
	fmt.Println("}")
	fmt.Println()
	fmt.Println("// With bucketing: Compiles ~7 graphs for batch sizes 1-128")
	fmt.Println("// Without bucketing: Could compile 100+ graphs for varying sizes")
}

// createBatch is a helper to create a batch of given size
func createBatch(size int) []float32 {
	return xslices.SliceWithValue(size, float32(1.0))
}

// demonstrateWithBackend shows actual usage (requires backend)
func demonstrateWithBackend(backend backends.Backend) {
	fmt.Println("\n=== Live Demonstration ===")

	// Create executor with Pow2 bucketing
	exec := MustNewExec(backend, SimpleSumFn).WithPow2Bucketing()

	// Test with various batch sizes
	for _, batchSize := range []int{1, 2, 3, 4, 5, 8, 9, 16} {
		batch := createBatch(batchSize)
		result := exec.MustExec(batch)[0]
		sum := tensors.ToScalar[float32](result)

		fmt.Printf("Batch size %2d → Sum: %.0f (Graphs cached: %d)\n",
			batchSize, sum, exec.CacheSize())
	}

	fmt.Println("\nResult: Used only 5 graphs for 8 different batch sizes")
	fmt.Println("Without bucketing: Would need 8 separate graphs")
}
