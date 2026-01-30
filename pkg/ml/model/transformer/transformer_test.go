// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

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

// TestModel groups transformer model tests.
func TestModel(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		cfg := New(1000, 128, 4, 8, 16)
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
		cfg := New(1000, 128, 4, 8, 16).
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

	t.Run("NewFromContext", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParams(map[string]any{
			ParamVocabSize:    1000,
			ParamEmbedDim:     128,
			ParamNumLayers:    4,
			ParamNumHeads:     8,
			ParamHeadDim:      16,
			ParamFFNDim:       256,
			ParamMaxPosEmbed:  1024,
			ParamDType:        "float16",
			ParamDropout:      0.1,
			ParamUseRoPE:      true,
			ParamRoPEBaseFreq: 5000.0,
			ParamUseLayerNorm: false,
			ParamUseBias:      false,
		})

		cfg := NewFromContext(ctx)
		assert.Equal(t, 1000, cfg.VocabSize)
		assert.Equal(t, 128, cfg.EmbedDim)
		assert.Equal(t, 4, cfg.NumLayers)
		assert.Equal(t, 8, cfg.NumHeads)
		assert.Equal(t, 16, cfg.HeadDim)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.True(t, cfg.UseRoPE)
		assert.Equal(t, 5000.0, cfg.RoPEBaseFreq)
		assert.False(t, cfg.UseLayerNorm)
		assert.False(t, cfg.UseBias)
	})

	t.Run("FromContext", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParams(map[string]any{
			ParamFFNDim:       256,
			ParamMaxPosEmbed:  1024,
			ParamDType:        "float16",
			ParamDropout:      0.1,
			ParamUseRoPE:      true,
			ParamRoPEBaseFreq: 5000.0,
			ParamUseLayerNorm: false,
			ParamUseBias:      false,
		})

		cfg := New(1000, 128, 4, 8, 16).FromContext(ctx)
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
	t.Run("ForTrainingForward", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		ctx := context.New()
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		modelFn := cfg.ForTraining()
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
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		modelFn := cfg.ForGeneration()
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
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithRoPE(10000.0)
		modelFn := cfg.ForGeneration()
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
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithLayerNorm(false)
		modelFn := cfg.ForTraining()
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
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithBias(false)
		modelFn := cfg.ForTraining()
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
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithDropout(0.1)
		modelFn := cfg.ForTraining()
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

			cfg := New(100, 64, 2, 4, 16).
				WithFFNDim(128).
				WithMaxPosEmbed(128)

			modelFn := cfg.ForTraining()

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
