package sample

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

// TestGreedy groups greedy sampling tests.
func TestGreedy(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node { return Greedy(logits) })
		logits := [][]float32{
			{0, 0, 0, 0, 0, 10, 0, 0, 0, 0},
			{0, 0, 0, 10, 0, 0, 0, 0, 0, 0},
		}
		tokens := exec.MustExec(logits)[0]
		assert.Equal(t, dtypes.Int32, tokens.DType())
		assert.Equal(t, []int{2}, tokens.Shape().Dimensions)
		data := tokens.Value().([]int32)
		assert.Equal(t, int32(5), data[0])
		assert.Equal(t, int32(3), data[1])
	})
}

// TestTemperature groups temperature sampling tests.
func TestTemperature(t *testing.T) {
	t.Run("Range", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node { return Temperature(ctx, logits, 1.0) })
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		assert.Equal(t, dtypes.Int32, tokens.DType())
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
	})
}

// TestTopP groups top-p (nucleus) sampling tests.
func TestTopP(t *testing.T) {
	t.Run("BatchShape", func(t *testing.T) {
		// Test that TopPSample correctly handles batch dimensions
		// This tests the BroadcastToShape fix for condition broadcasting
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return TopP(ctx, logits, 0.9, 1.0)
		})

		// Batch of 2 with vocab size 10
		logits := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		}
		tokens := exec.MustExec(logits)[0]

		assert.Equal(t, dtypes.Int32, tokens.DType())
		assert.Equal(t, []int{2}, tokens.Shape().Dimensions)
		data := tokens.Value().([]int32)

		// Tokens should be within valid range
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
		assert.GreaterOrEqual(t, data[1], int32(0))
		assert.Less(t, data[1], int32(10))
	})

	t.Run("SingleBatch", func(t *testing.T) {
		// Test with batch size 1 (common case in generation)
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return TopP(ctx, logits, 0.95, 0.8)
		})

		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]

		assert.Equal(t, dtypes.Int32, tokens.DType())
		assert.Equal(t, []int{1}, tokens.Shape().Dimensions)
		data := tokens.Value().([]int32)

		// Should sample from top tokens (higher probability for larger logits)
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
	})
}

// TestTopKSample groups top-k sampling tests.
func TestTopKSample(t *testing.T) {
	t.Run("BatchShape", func(t *testing.T) {
		// Test that TopKSample correctly handles batch dimensions
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return TopKWithTemperature(ctx, logits, 5, 1.0)
		})

		// Batch of 2 with vocab size 10
		logits := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		}
		tokens := exec.MustExec(logits)[0]

		assert.Equal(t, dtypes.Int32, tokens.DType())
		assert.Equal(t, []int{2}, tokens.Shape().Dimensions)
		data := tokens.Value().([]int32)

		// Tokens should be from top-5 (indices 5-9 for first, 0-4 for second)
		assert.GreaterOrEqual(t, data[0], int32(5))
		assert.Less(t, data[0], int32(10))
		assert.GreaterOrEqual(t, data[1], int32(0))
		assert.Less(t, data[1], int32(5))
	})
}

// TestSampleWithStrategy groups SampleWithStrategy tests.
func TestSampleWithStrategy(t *testing.T) {
	t.Run("Greedy", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return SampleWithStrategy(ctx, logits, StrategyGreedy, 0, 0, 0)
		})
		logits := [][]float32{{0, 0, 0, 0, 0, 0, 0, 10, 0, 0}}
		tokens := exec.MustExec(logits)[0]
		data := tokens.Value().([]int32)
		assert.Equal(t, int32(7), data[0])
	})

	t.Run("Temperature", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return SampleWithStrategy(ctx, logits, StrategyTemperature, 0.8, 0, 0)
		})
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
	})

	t.Run("TopK", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return SampleWithStrategy(ctx, logits, StrategyTopK, 1.0, 3, 0)
		})
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(7))
		assert.Less(t, data[0], int32(10))
	})

	t.Run("TopP", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return SampleWithStrategy(ctx, logits, StrategyTopP, 1.0, 0, 0.8)
		})
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(5))
		assert.Less(t, data[0], int32(10))
	})
}
