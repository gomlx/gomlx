package decode

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	graphtest "github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode/sample"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDecoder groups config default and builder tests.
func TestDecoder(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node { return tokens }
		cfg := New(modelFn)
		assert.Equal(t, 100, cfg.MaxLength)
		assert.Equal(t, sample.StrategyGreedy, cfg.Strategy)
		assert.Equal(t, float32(1.0), cfg.Temperature)
		assert.Equal(t, 50, cfg.TopK)
		assert.Equal(t, float32(0.9), cfg.TopP)
		assert.Equal(t, 4, cfg.BeamSize)
		assert.Equal(t, -1, cfg.EosTokenId)
		assert.False(t, cfg.StopOnEOS)
	})

	t.Run("Builders", func(t *testing.T) {
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node { return tokens }
		cfg := New(modelFn).
			WithMaxLength(50).
			WithStrategy(sample.StrategyTemperature).
			WithTemperature(0.7).
			WithTopK(40).
			WithTopP(0.95).
			WithBeamSize(8).
			WithEOS(2)
		assert.Equal(t, 50, cfg.MaxLength)
		assert.Equal(t, sample.StrategyTemperature, cfg.Strategy)
		assert.Equal(t, float32(0.7), cfg.Temperature)
		assert.Equal(t, 40, cfg.TopK)
		assert.Equal(t, float32(0.95), cfg.TopP)
		assert.Equal(t, 8, cfg.BeamSize)
		assert.Equal(t, 2, cfg.EosTokenId)
		assert.True(t, cfg.StopOnEOS)
	})
}

// TestDecoderSampling groups non-beam sampling tests.
func TestDecoderSampling(t *testing.T) {
	t.Run("Greedy", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node {
			batchSize := tokens.Shape().Dimensions[0]
			seqLen := tokens.Shape().Dimensions[1]
			vocabSize := 10
			g := tokens.Graph()
			logits := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))
			indices := Iota(g, logits.Shape(), 2)
			logits = Where(Equal(indices, ConstAs(indices, 5)), ConstAs(logits, 100.0), logits)
			return logits
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10)
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, ctx, prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		// Expect full sequence [batch=1, length=10]
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
		seq := result.Value().([][]int32)
		// Last generated token should be 5 (greedy)
		assert.Equal(t, int32(5), seq[0][len(seq[0])-1])
	})

	t.Run("Temperature", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node {
			batchSize := tokens.Shape().Dimensions[0]
			seqLen := tokens.Shape().Dimensions[1]
			vocabSize := 10
			g := tokens.Graph()
			return IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyTemperature).WithTemperature(1.5).WithMaxLength(10)
		prompt := [][]int32{{1, 2, 3}}
		result, err := cfg.Decode(backend, ctx, prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
		seq := result.Value().([][]int32)
		last := seq[0][len(seq[0])-1]
		assert.GreaterOrEqual(t, last, int32(0))
		assert.Less(t, last, int32(10))
	})

	t.Run("OneDPrompt", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node {
			batchSize := tokens.Shape().Dimensions[0]
			seqLen := tokens.Shape().Dimensions[1]
			vocabSize := 10
			g := tokens.Graph()
			return IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, vocabSize))
		}
		cfg := New(modelFn).WithStrategy(sample.StrategyGreedy).WithMaxLength(10)
		prompt := []int32{1, 2, 3}
		result, err := cfg.Decode(backend, ctx, prompt)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, 2, result.Rank())
		assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
	})

	t.Run("PromptTooLong", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node { return tokens }
		cfg := New(modelFn).WithMaxLength(5)
		prompt := [][]int32{{1, 2, 3, 4, 5, 6, 7, 8}}
		_, err := cfg.Decode(backend, ctx, prompt)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "prompt length")
	})
}

// TestGenerateBeamSearchNotImplemented remains as a placeholder expectation.
func TestBeamSearch(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node {
		batchSize := tokens.Shape().Dimensions[0]
		seqLen := tokens.Shape().Dimensions[1]
		g := tokens.Graph()
		return IotaFull(g, shapes.Make(dtypes.Float32, batchSize, seqLen, 10))
	}
	cfg := New(modelFn).WithStrategy(sample.StrategyBeamSearch).WithBeamSize(4).WithMaxLength(10)
	prompt := [][]int32{{1, 2, 3}}
	result, err := cfg.Decode(backend, ctx, prompt)
	require.NoError(t, err)
	require.NotNil(t, result)
	// Expect best sequences: shape [batch, seq_len]
	assert.Equal(t, 2, result.Rank())
	assert.Equal(t, []int{1, 10}, result.Shape().Dimensions)
}

// TestStreamingNotImplemented groups streaming placeholder test.
func TestStreamingNotImplemented(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	var modelFn ModelFn = func(ctx *context.Context, tokens *Node) *Node { return tokens }
	cfg := New(modelFn)
	prompt := []int32{1, 2, 3}
	err := cfg.GenerateStreaming(backend, ctx, prompt, func(token int) bool { return true })
	require.Error(t, err)
	assert.Contains(t, err.Error(), "streaming generation not yet implemented")
}
