// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/support/testutil"
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
		assert.Nil(t, cfg.posEncoder)
		assert.Equal(t, cfg.Normalization, layers.NormalizationLayerNorm)
		assert.True(t, cfg.UseBias)
	})

	t.Run("Builders", func(t *testing.T) {
		cfg := New(1000, 128, 4, 8, 16).
			WithFFNDim(256).
			WithMaxPosEmbed(1024).
			WithDType(dtypes.Float16).
			WithDropout(0.1).
			WithRoPE(5000.0).
			WithNormalization("").
			WithBias(false)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, cfg.Normalization, layers.NormalizationNone)
		assert.False(t, cfg.UseBias)
	})

	t.Run("NewFromContext", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParams(map[string]any{
			ParamVocabSize:     1000,
			ParamEmbedDim:      128,
			ParamNumLayers:     4,
			ParamNumHeads:      8,
			ParamHeadDim:       16,
			ParamFFNDim:        256,
			ParamMaxPosEmbed:   1024,
			ParamDType:         "float16",
			ParamDropout:       0.1,
			ParamUseRoPE:       true,
			ParamRoPEBaseFreq:  5000.0,
			ParamUseBias:       false,
			ParamNormalization: "none", // layers.NormalizationNone or "".
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
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, cfg.Normalization, layers.NormalizationNone)
		assert.False(t, cfg.UseBias)
	})

	t.Run("FromContext", func(t *testing.T) {
		ctx := context.New()
		ctx.SetParams(map[string]any{
			ParamFFNDim:        256,
			ParamMaxPosEmbed:   1024,
			ParamDType:         "float16",
			ParamDropout:       0.1,
			ParamUseRoPE:       true,
			ParamRoPEBaseFreq:  5000.0,
			ParamUseBias:       false,
			ParamNormalization: "", // or layers.NormalizationNone.
		})

		cfg := New(1000, 128, 4, 8, 16).FromContext(ctx)
		assert.Equal(t, 256, cfg.FFNDim)
		assert.Equal(t, 1024, cfg.MaxPosEmbed)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		assert.Equal(t, dtypes.Float16, cfg.DType)
		assert.Equal(t, 0.1, cfg.Dropout)
		require.NotNil(t, cfg.posEncoder)
		rope, ok := cfg.posEncoder.(*pos.RoPE)
		require.True(t, ok)
		assert.Equal(t, 5000.0, rope.BaseFreq)
		assert.Equal(t, layers.NormalizationNone, cfg.Normalization)
		assert.False(t, cfg.UseBias)
	})
}

// TestTransformerBuilder groups transformer builder/exposed functions tests.
func TestTransformerBuilder(t *testing.T) {
	t.Run("BuildGraph", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		cfg := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		g := NewGraph(backend, "BuildGraph")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(cfg.VocabSize)))
		logits := cfg.Logits(ctx, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
		assert.Equal(t, dtypes.Float32, logits.DType())
	})

	t.Run("BuildGraphWithKVCache", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128)
		g := NewGraph(backend, "BuildGraphWithKVCache")
		prompt := IotaFull(g, shapes.Make(dtypes.Int32, 2, 5))
		prompt = Mod(prompt, Const(g, int32(model.VocabSize)))
		logits := model.LogitsWithKVCache(ctx, prompt, 0)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 5, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerVariants groups variant configurations.
func TestTransformerVariants(t *testing.T) {
	t.Run("WithRoPE", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithRoPE(10000.0)
		g := NewGraph(backend, "WithRoPE")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 1, 4))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.LogitsWithKVCache(ctx, tokens, 0)
		require.NotNil(t, logits)
		assert.Equal(t, []int{1, 4, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutLayerNorm", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithNormalization("none")
		g := NewGraph(backend, "WithoutLayerNorm")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(ctx, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithoutBias", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithBias(false)
		g := NewGraph(backend, "WithoutBias")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(ctx, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})

	t.Run("WithDropout", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		ctx := context.New()
		model := New(100, 64, 2, 4, 16).WithFFNDim(128).WithMaxPosEmbed(128).WithDropout(0.1)
		g := NewGraph(backend, "WithDropout")
		tokens := IotaFull(g, shapes.Make(dtypes.Int32, 2, 8))
		tokens = Mod(tokens, Const(g, int32(model.VocabSize)))
		logits := model.Logits(ctx, tokens, nil)
		require.NotNil(t, logits)
		assert.Equal(t, []int{2, 8, 100}, logits.Shape().Dimensions)
	})
}

// TestTransformerBatchSizes groups batch size variations.
func TestTransformerBatchSizes(t *testing.T) {
	batchSizes := []int{1, 2, 4}
	backend := testutil.BuildTestBackend()

	for _, batchSize := range batchSizes {
		t.Run(fmt.Sprintf("BatchSize%d", batchSize), func(t *testing.T) {
			ctx := context.New()

			model := New(100, 64, 2, 4, 16).
				WithFFNDim(128).
				WithMaxPosEmbed(128)

			seqLen := 8

			manager := backend
			g := NewGraph(manager, "TestTransformerBatchSize")

			tokens := IotaFull(g, shapes.Make(dtypes.Int32, batchSize, seqLen))
			tokens = Mod(tokens, Const(g, int32(model.VocabSize)))

			logits := model.Logits(ctx, tokens, nil)
			require.NotNil(t, logits)
			assert.Equal(t, []int{batchSize, seqLen, 100}, logits.Shape().Dimensions)
		})
	}
}
