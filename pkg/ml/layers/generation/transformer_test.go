package generation

import (
	"fmt"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	graphtest "github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTransformerConfig groups transformer config tests.
func TestTransformerConfig(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		cfg := NewTransformerConfig(1000, 128, 4, 8, 16)
		assert.Equal(t, 1000, cfg.VocabSize)
		assert.Equal(t, 128, cfg.EmbedDim)
		assert.Equal(t, 4, cfg.NumLayers)
		assert.Equal(t, 8, cfg.NumHeads)
		assert.Equal(t, 16, cfg.HeadDim)
		assert.Equal(t, 512, cfg.FFNDim)
		assert.Equal(t, 512, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float32, cfg.DType)
		assert.Equal(t, 0.0, cfg.Dropout)
		assert.False(t, cfg.UseRoPE)
		assert.Equal(t, 10000.0, cfg.RoPEBaseFreq)
		assert.True(t, cfg.UseLayerNorm)
		assert.True(t, cfg.UseBias)
	})

	t.Run("Builders", func(t *testing.T) {
		cfg := NewTransformerConfig(1000, 128, 4, 8, 16).
			WithFFNDim(256).
			WithMaxPosEmbed(1024).
			WithDType(dtypes.Float16).
			WithDropout(0.1).
			WithRoPE(5000.0).
			WithLayerNorm(false).
			WithBias(false)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.True(t, cfg.UseRoPE)
		assert.Equal(t, 5000.0, cfg.RoPEBaseFreq)
		assert.False(t, cfg.UseLayerNorm)
		assert.False(t, cfg.UseBias)
	})
}

// TestTransformerBuilder groups transformer builder/exposed functions tests.
func TestTransformerBuilder(t *testing.T) {
	t.Run("BuildCachedTransformer", func(t *testing.T) {
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16)
		transformer := BuildCachedTransformer(ctx, cfg)
		require.NotNil(t, transformer)
		require.NotNil(t, transformer.Config)
		assert.Equal(t, cfg, transformer.Config)
		assert.False(t, transformer.KVCacheShape.Ok()) // Empty shape initially
	})

	t.Run("ForTrainingForward", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		transformer := BuildCachedTransformer(ctx, cfg)
		modelFn := transformer.ForTraining()
		g := NewGraph(backend, "ForTrainingForward")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, tokens)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
		assert.Equal(t, dtypes.Float32, logits.DType())
	})

	t.Run("ForGenerationPrompt", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		transformer := BuildCachedTransformer(ctx, cfg)
		// KV cache shape is now created automatically within forwardFull based on batch size
		modelFn := transformer.ForGeneration()
		g := NewGraph(backend, "ForGenerationPrompt")
		prompt := IotaFull(g, shapes.Make(dtypes.Int32, 2, 5))
		prompt = Mod(prompt, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, prompt, 0)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 5, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerVariants groups variant configurations.
func TestTransformerVariants(t *testing.T) {
	t.Run("WithRoPE", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithRoPE(10000.0)
		transformer := BuildCachedTransformer(ctx, cfg)
		// KV cache shape is now created automatically within forwardFull based on batch size
		modelFn := transformer.ForGeneration()
		g := NewGraph(backend, "WithRoPE")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 1, 4))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, tokens, 0)
		require.NotNil(t, logits)
		assert.Equal(t, []int{1, 4, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutLayerNorm", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithLayerNorm(false)
		transformer := BuildCachedTransformer(ctx, cfg)
		modelFn := transformer.ForTraining()
		g := NewGraph(backend, "WithoutLayerNorm")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, tokens)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutBias", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithBias(false)
		transformer := BuildCachedTransformer(ctx, cfg)
		modelFn := transformer.ForTraining()
		g := NewGraph(backend, "WithoutBias")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, tokens)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithDropout", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := NewTransformerConfig(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithDropout(0.1)
		transformer := BuildCachedTransformer(ctx, cfg)
		modelFn := transformer.ForTraining()
		g := NewGraph(backend, "WithDropout")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := modelFn(ctx, tokens)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerBatchSizes groups batch size variations.
func TestTransformerBatchSizes(t *testing.T) {
	batchSizes := []int{1, 2, 4}
	backend := graphtest.BuildTestBackend()

	for _, batchSize := range batchSizes {
		t.Run(fmt.Sprintf("BatchSize%d", batchSize), func(t *testing.T) {
			ctx := context.New()

			cfg := NewTransformerConfig(100, 64, 2, 4, 16).
				WithFFNDim(128).
				WithMaxPosEmbed(128)

			transformer := BuildCachedTransformer(ctx, cfg)
			modelFn := transformer.ForTraining()

			seqLen := 8

			manager := backend
			g := NewGraph(manager, "TestTransformerBatchSize")

			tokens := IotaFull(g, shapes.Make(dtypes.Int32, batchSize, seqLen))
			tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))

			logits := modelFn(ctx, tokens)
			require.NotNil(t, logits)
			assert.Equal(t, []int{batchSize, seqLen, 100}, logits.Shape().Dimensions)
		})
	}
}
