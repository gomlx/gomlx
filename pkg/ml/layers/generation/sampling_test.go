package generation

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
)

// TestGreedySample groups greedy sampling tests.
func TestGreedySample(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node { return GreedySample(ctx, logits) })
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

// TestTemperatureSample groups temperature sampling tests.
func TestTemperatureSample(t *testing.T) {
	t.Run("Range", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node { return TemperatureSample(ctx, logits, 1.0) })
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		assert.Equal(t, dtypes.Int32, tokens.DType())
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
	})
}

// TestSampleWithStrategy groups SampleWithStrategy tests.
func TestSampleWithStrategy(t *testing.T) {
	t.Run("Greedy", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, logits *Node) *Node {
			return SampleWithStrategy(ctx, logits, "greedy", 0, 0, 0)
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
			return SampleWithStrategy(ctx, logits, "temperature", 0.8, 0, 0)
		})
		logits := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
		tokens := exec.MustExec(logits)[0]
		data := tokens.Value().([]int32)
		assert.GreaterOrEqual(t, data[0], int32(0))
		assert.Less(t, data[0], int32(10))
	})
}
