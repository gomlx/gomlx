package attention

import (
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	graphtest "github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper function to create 3D test data
func makeTestData3D(batch, seq, dim int) [][][]float32 {
	data := make([][][]float32, batch)
	for i := range data {
		data[i] = make([][]float32, seq)
		for j := range data[i] {
			data[i][j] = make([]float32, dim)
			for k := range data[i][j] {
				data[i][j][k] = float32(i*seq*dim + j*dim + k)
			}
		}
	}
	return data
}

// TestNewIncrementalAttention tests the NewIncrementalAttention constructor and Done() method
func TestNewIncrementalAttention(t *testing.T) {
	t.Run("WithoutCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 4
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			output := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).Done()
			return output
		})

		input := makeTestData3D(1, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, 1, result.Shape().Dimensions[0])
		assert.Equal(t, seqLen, result.Shape().Dimensions[1])
		assert.Equal(t, embedDim, result.Shape().Dimensions[2])
	})

	t.Run("WithCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 1
		embedDim := 32
		numHeads := 2
		headDim := 16
		maxSeqLen := 10

		cache := NewKVCache(ctx, "with_cache_test", 1, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			output := NewIncrementalAttention(ctx, query, numHeads, headDim, cache).Done()
			return output
		})

		input := makeTestData3D(1, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, 1, result.Shape().Dimensions[0])
		assert.Equal(t, seqLen, result.Shape().Dimensions[1])
		assert.Equal(t, embedDim, result.Shape().Dimensions[2])
	})

	t.Run("BatchProcessing", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 4
		seqLen := 2
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, nil).Done()
		})

		input := makeTestData3D(batchSize, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, batchSize, result.Shape().Dimensions[0])
		assert.Equal(t, seqLen, result.Shape().Dimensions[1])
	})

	t.Run("LongerSequence", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		seqLen := 8
		embedDim := 64
		numHeads := 4
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, nil).Done()
		})

		input := makeTestData3D(batchSize, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, batchSize, result.Shape().Dimensions[0])
		assert.Equal(t, seqLen, result.Shape().Dimensions[1])
		assert.Equal(t, embedDim, result.Shape().Dimensions[2])
	})
}

// TestWithRoPE tests the WithRoPE configuration method
func TestWithRoPE(t *testing.T) {
	t.Run("BasicRoPE", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 3
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			output := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				WithRoPE(10000.0).
				WithPosition(0).
				Done()
			return output
		})

		input := makeTestData3D(1, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, seqLen, result.Shape().Dimensions[1])
	})

	t.Run("RoPEWithCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		embedDim := 32
		numHeads := 2
		headDim := 16
		maxSeqLen := 10

		cache := NewKVCache(ctx, "cache_rope_test", 1, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, cache).
				WithRoPE(10000.0).
				WithPosition(3).
				Done()
		})

		input := makeTestData3D(1, 1, embedDim)
		result := exec.MustExec(input)[0]

		// Check output shape
		assert.Equal(t, 3, result.Rank())
		assert.Equal(t, embedDim, result.Shape().Dimensions[2])
	})
}

// TestWithPosition tests the WithPosition configuration method
func TestWithPosition(t *testing.T) {
	t.Run("MultipleSteps", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		embedDim := 32
		numHeads := 2
		headDim := 16
		maxSeqLen := 10

		cache := NewKVCache(ctx, "multi_steps_cache", 1, numHeads, maxSeqLen, headDim, dtypes.Float32)

		// First step
		exec1 := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, cache).
				WithPosition(0).Done()
		})
		result1 := exec1.MustExec(makeTestData3D(1, 1, embedDim))[0]
		assert.Equal(t, 1, result1.Shape().Dimensions[1])

		// Second step - reuse context
		ctx2 := ctx.Reuse()
		exec2 := context.MustNewExec(backend, ctx2, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, cache).
				WithPosition(1).Done()
		})
		result2 := exec2.MustExec(makeTestData3D(1, 1, embedDim))[0]
		assert.Equal(t, 1, result2.Shape().Dimensions[1])
	})
}

// TestSetKeyQueryDim tests the SetKeyQueryDim configuration method
func TestSetKeyQueryDim(t *testing.T) {
	t.Run("CustomDimensions", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 2
		embedDim := 64
		numHeads := 4
		customOutputDim := 128

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, 16, nil).
				SetKeyQueryDim(20).
				SetValueDim(24).
				SetOutputDim(customOutputDim).
				Done()
		})

		input := makeTestData3D(1, seqLen, embedDim)
		result := exec.MustExec(input)[0]

		// Check output dimension matches custom output
		assert.Equal(t, customOutputDim, result.Shape().Dimensions[2])
	})
}

// TestDoneWithAttentionWeights tests the DoneWithAttentionWeights() method
func TestDoneWithAttentionWeights(t *testing.T) {
	t.Run("ReturnsWeights", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 4
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) []*Node {
			output, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				DoneWithAttentionWeights()
			return []*Node{output, weights}
		})

		input := makeTestData3D(1, seqLen, embedDim)
		results := exec.MustExec(input)

		output := results[0]
		weights := results[1]

		// Verify output is returned
		require.NotNil(t, output)
		assert.Equal(t, 3, output.Rank())

		// Verify weights are non-nil
		require.NotNil(t, weights, "Attention weights should not be nil")
		assert.Equal(t, 4, weights.Rank(), "Attention weights should have rank 4")
	})

	t.Run("CorrectShape", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		seqLen := 6
		embedDim := 48
		numHeads := 3
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			_, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				DoneWithAttentionWeights()
			return weights
		})

		input := makeTestData3D(batchSize, seqLen, embedDim)
		weights := exec.MustExec(input)[0]

		// Verify shape: [batch_size, num_heads, seq_len, seq_len]
		require.NotNil(t, weights)
		assert.Equal(t, 4, weights.Rank())
		assert.Equal(t, batchSize, weights.Shape().Dimensions[0], "Batch dimension mismatch")
		assert.Equal(t, numHeads, weights.Shape().Dimensions[1], "Num heads dimension mismatch")
		assert.Equal(t, seqLen, weights.Shape().Dimensions[2], "Query sequence dimension mismatch")
		assert.Equal(t, seqLen, weights.Shape().Dimensions[3], "Key sequence dimension mismatch")
	})

	t.Run("SoftmaxProperty", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		batchSize := 2
		seqLen := 4
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			_, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				DoneWithAttentionWeights()
			return weights
		})

		input := makeTestData3D(batchSize, seqLen, embedDim)
		weights := exec.MustExec(input)[0]

		// Get the actual weights data: [batch, heads, query_seq, key_seq]
		weightsData := weights.Value().([][][][]float32)

		// Verify softmax property: sum over key dimension (last axis) should be ~1.0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for q := 0; q < seqLen; q++ {
					sum := float32(0.0)
					for k := 0; k < seqLen; k++ {
						sum += weightsData[b][h][q][k]
					}
					assert.InDelta(t, 1.0, sum, 0.0001,
						"Attention weights should sum to 1.0 for batch=%d, head=%d, query=%d", b, h, q)
				}
			}
		}
	})

	t.Run("MatchesDone", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()

		seqLen := 5
		embedDim := 40
		numHeads := 2
		headDim := 20

		// Test Done() - separate context
		ctx1 := context.New()
		execDone := context.MustNewExec(backend, ctx1, func(ctx *context.Context, query *Node) *Node {
			return NewIncrementalAttention(ctx, query, numHeads, headDim, nil).Done()
		})

		// Test DoneWithAttentionWeights() - separate context
		ctx2 := context.New()
		execWithWeights := context.MustNewExec(backend, ctx2, func(ctx *context.Context, query *Node) []*Node {
			output, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				DoneWithAttentionWeights()
			return []*Node{output, weights}
		})

		input := makeTestData3D(1, seqLen, embedDim)

		outputDone := execDone.MustExec(input)[0]
		results := execWithWeights.MustExec(input)
		outputWithWeights := results[0]
		weights := results[1]

		// Verify both outputs have the same shape (they use the same underlying done() implementation)
		assert.Equal(t, outputDone.Shape().Dimensions, outputWithWeights.Shape().Dimensions)

		// Verify weights are non-nil
		require.NotNil(t, weights)

		// Note: We don't compare output values since the two contexts create different
		// random weight initializations. The implementation is tested by the shared done() method.
	})

	t.Run("WithCache", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx1 := context.New()

		batchSize := 1
		embedDim := 32
		numHeads := 2
		headDim := 16
		maxSeqLen := 10

		// Create KV cache for first execution
		cache1 := NewKVCache(ctx1, "test_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		// First call with initial sequence
		exec1 := context.MustNewExec(backend, ctx1, func(ctx *context.Context, query *Node) []*Node {
			output, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, cache1).
				WithPosition(0).
				DoneWithAttentionWeights()
			return []*Node{output, weights}
		})

		input1 := makeTestData3D(batchSize, 3, embedDim)
		results1 := exec1.MustExec(input1)
		weights1 := results1[1]

		// Verify first call weights shape: [batch, heads, 3, 3]
		require.NotNil(t, weights1)
		assert.Equal(t, 3, weights1.Shape().Dimensions[2], "First call query seq len")
		assert.Equal(t, 3, weights1.Shape().Dimensions[3], "First call key seq len")

		// Second call with one new token (incremental) - needs separate context
		ctx2 := context.New()
		cache2 := NewKVCache(ctx2, "test_cache", batchSize, numHeads, maxSeqLen, headDim, dtypes.Float32)

		exec2 := context.MustNewExec(backend, ctx2, func(ctx *context.Context, query *Node) []*Node {
			output, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, cache2).
				WithPosition(3).
				DoneWithAttentionWeights()
			return []*Node{output, weights}
		})

		input2 := makeTestData3D(batchSize, 1, embedDim)
		results2 := exec2.MustExec(input2)
		weights2 := results2[1]

		// Verify second call weights shape: [batch, heads, 1, 4]
		// (1 new query token attending to 4 total cached keys)
		require.NotNil(t, weights2)
		assert.Equal(t, 1, weights2.Shape().Dimensions[2], "Second call query seq len")
		assert.Equal(t, 4, weights2.Shape().Dimensions[3], "Second call key seq len (cached + new)")

		// Verify softmax property still holds
		weightsData := weights2.Value().([][][][]float32)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for q := 0; q < 1; q++ {
					sum := float32(0.0)
					for k := 0; k < 4; k++ {
						sum += weightsData[b][h][q][k]
					}
					assert.InDelta(t, 1.0, sum, 0.0001,
						"Cached attention weights should sum to 1.0 for batch=%d, head=%d", b, h)
				}
			}
		}
	})

	t.Run("NonNegative", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		seqLen := 4
		embedDim := 32
		numHeads := 2
		headDim := 16

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, query *Node) *Node {
			_, weights := NewIncrementalAttention(ctx, query, numHeads, headDim, nil).
				DoneWithAttentionWeights()
			return weights
		})

		input := makeTestData3D(1, seqLen, embedDim)
		weights := exec.MustExec(input)[0]

		// Get the actual weights data: [batch, heads, query_seq, key_seq]
		weightsData := weights.Value().([][][][]float32)

		// Verify all weights are non-negative (property of softmax)
		for b := range weightsData {
			for h := range weightsData[b] {
				for q := range weightsData[b][h] {
					for k := range weightsData[b][h][q] {
						weight := weightsData[b][h][q][k]
						assert.True(t, weight >= 0.0 && !math.IsNaN(float64(weight)),
							"Attention weight should be non-negative and not NaN, got %f at [%d][%d][%d][%d]",
							weight, b, h, q, k)
					}
				}
			}
		}
	})
}
