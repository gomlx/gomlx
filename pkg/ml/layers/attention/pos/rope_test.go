package pos

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRoPE groups tests for the RoPE function.
func TestRoPE(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			return rope.Apply(x, startPos)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
			{0, 0, 1, 0, 0, 0, 0, 0},
			{0, 0, 0, 1, 0, 0, 0, 0},
		}

		startPos := []int32{0}
		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{4, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentStartPos", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			return rope.Apply(x, startPos)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
		startPos := []int32{5}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentBaseFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(5000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			return rope.Apply(x, startPos)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{1, 2, 3, 4, 5, 6, 7, 8},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("PreservesNorms", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPE(10000.0)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			rotated := rope.Apply(x, startPos)
			return ReduceAllSum(Abs(rotated))
		})

		input := [][]float32{
			{1, 1, 1, 1, 1, 1, 1, 1},
			{2, 2, 2, 2, 2, 2, 2, 2},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		sum := output.Value().(float32)
		assert.Greater(t, sum, float32(0.0))
	})
}

// TestRoPEWithCustomDim groups tests for the RoPEWithCustomDim function.
func TestRoPEWithCustomDim(t *testing.T) {
	t.Run("PartialRange", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Input shape: [batch=2, embed_dim=8]
		rope := NewRoPEWithDimRange(10000.0, 2, 6)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE only to middle dimensions [2:6]
			return rope.Apply(x, startPos)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{2, 3, 4, 5, 6, 7, 8, 9},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		// Shape should be preserved
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("FullEmbedding", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPEWithDimRange(10000.0, 0, 8)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE to the full embedding range [0:8]
			return rope.Apply(x, startPos)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
		}
		startPos := []int32{0}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("StartPosAndFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		rope := NewRoPEWithDimRange(5000.0, 4, 8)
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
			// Apply RoPE to tail half with different startPos and baseFreq
			return rope.Apply(x, startPos)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
		startPos := []int32{5}

		output := exec.MustExec(input, startPos)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("OddRangePanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Range [1:4] has length 3 (odd) and should panic
		require.Panics(t, func() {
			rope := NewRoPEWithDimRange(10000.0, 1, 4)
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
				return rope.Apply(x, startPos)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			startPos := []int32{0}
			_ = exec.MustExec(input, startPos)[0]
		})
	})

	t.Run("OutOfBoundsPanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// dimEnd beyond embedding size should panic
		require.Panics(t, func() {
			rope := NewRoPEWithDimRange(10000.0, 0, 10)
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x, startPos *Node) *Node {
				return rope.Apply(x, startPos)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			startPos := []int32{0}
			_ = exec.MustExec(input, startPos)[0]
		})
	})
}
