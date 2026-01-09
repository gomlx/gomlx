package attention

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRoPE groups tests for the RoPE function.
func TestRoPE(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return RoPE(x, 0, 10000.0)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
			{0, 0, 1, 0, 0, 0, 0, 0},
			{0, 0, 0, 1, 0, 0, 0, 0},
		}

		output := exec.MustExec(input)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{4, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentStartPos", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return RoPE(x, 5, 10000.0)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}

		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("DifferentBaseFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			return RoPE(x, 0, 5000.0)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{1, 2, 3, 4, 5, 6, 7, 8},
		}

		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("PreservesNorms", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			rotated := RoPE(x, 0, 10000.0)
			return ReduceAllSum(Abs(rotated))
		})

		input := [][]float32{
			{1, 1, 1, 1, 1, 1, 1, 1},
			{2, 2, 2, 2, 2, 2, 2, 2},
		}

		output := exec.MustExec(input)[0]
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
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			// Apply RoPE only to middle dimensions [2:6]
			return RoPEWithCustomDim(x, 0, 10000.0, 2, 6)
		})

		input := [][]float32{
			{1, 2, 3, 4, 5, 6, 7, 8},
			{2, 3, 4, 5, 6, 7, 8, 9},
		}

		output := exec.MustExec(input)[0]
		// Shape should be preserved
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("FullEmbedding", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			// Apply RoPE to the full embedding range [0:8]
			return RoPEWithCustomDim(x, 0, 10000.0, 0, 8)
		})

		input := [][]float32{
			{1, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0, 0, 0},
		}

		output := exec.MustExec(input)[0]
		assert.Equal(t, dtypes.Float32, output.DType())
		assert.Equal(t, []int{2, 8}, output.Shape().Dimensions)
	})

	t.Run("StartPosAndFreq", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
			// Apply RoPE to tail half with different startPos and baseFreq
			return RoPEWithCustomDim(x, 5, 5000.0, 4, 8)
		})

		input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}

		output := exec.MustExec(input)[0]
		assert.Equal(t, []int{1, 8}, output.Shape().Dimensions)
	})

	t.Run("OddRangePanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// Range [1:4] has length 3 (odd) and should panic
		require.Panics(t, func() {
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
				return RoPEWithCustomDim(x, 0, 10000.0, 1, 4)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			_ = exec.MustExec(input)[0]
		})
	})

	t.Run("OutOfBoundsPanics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()

		// dimEnd beyond embedding size should panic
		require.Panics(t, func() {
			exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
				return RoPEWithCustomDim(x, 0, 10000.0, 0, 10)
			})

			input := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
			_ = exec.MustExec(input)[0]
		})
	})
}
