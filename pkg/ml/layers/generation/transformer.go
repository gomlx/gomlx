/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package generation

import (
	"fmt"
	"strconv"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// TransformerConfig holds configuration for building a cached transformer model.
// TransformerConfig configures a cached transformer model.
type TransformerConfig struct {
	VocabSize    int          // Vocabulary size
	EmbedDim     int          // Embedding dimension
	NumLayers    int          // Transformer layers
	NumHeads     int          // Attention heads per layer
	HeadDim      int          // Head dimension
	FFNDim       int          // Feed-forward hidden dimension
	MaxPosEmbed  int          // Max positional embedding length
	DType        dtypes.DType // Data type
	Dropout      float64      // Dropout rate (0.0 = none)
	UseRoPE      bool         // Use Rotary Position Embeddings
	RoPEBaseFreq float64      // RoPE base frequency (default 10000)
	UseLayerNorm bool         // Use layer normalization
	UseBias      bool         // Use bias in dense layers
}

// NewTransformerConfig creates a default transformer configuration.
// NewTransformerConfig creates a default configuration.
func NewTransformerConfig(vocabSize, embedDim, numLayers, numHeads, headDim int) *TransformerConfig {
	return &TransformerConfig{
		VocabSize:    vocabSize,
		EmbedDim:     embedDim,
		NumLayers:    numLayers,
		NumHeads:     numHeads,
		HeadDim:      headDim,
		FFNDim:       embedDim * 4, // 4x expansion
		MaxPosEmbed:  512,
		DType:        dtypes.Float32,
		Dropout:      0.0,
		UseRoPE:      false,
		RoPEBaseFreq: 10000.0,
		UseLayerNorm: true,
		UseBias:      true,
	}
}

// WithFFNDim sets FFN dimension.
func (cfg *TransformerConfig) WithFFNDim(dim int) *TransformerConfig {
	cfg.FFNDim = dim
	return cfg
}

// WithMaxPosEmbed sets max positional embedding length.
func (cfg *TransformerConfig) WithMaxPosEmbed(maxLen int) *TransformerConfig {
	cfg.MaxPosEmbed = maxLen
	return cfg
}

// WithDType sets data type.
func (cfg *TransformerConfig) WithDType(dtype dtypes.DType) *TransformerConfig {
	cfg.DType = dtype
	return cfg
}

// WithDropout sets dropout rate.
func (cfg *TransformerConfig) WithDropout(rate float64) *TransformerConfig {
	cfg.Dropout = rate
	return cfg
}

// WithRoPE enables RoPE with base frequency.
func (cfg *TransformerConfig) WithRoPE(baseFreq float64) *TransformerConfig {
	cfg.UseRoPE = true
	cfg.RoPEBaseFreq = baseFreq
	return cfg
}

// WithLayerNorm toggles layer normalization.
func (cfg *TransformerConfig) WithLayerNorm(use bool) *TransformerConfig {
	cfg.UseLayerNorm = use
	return cfg
}

// WithBias toggles dense bias.
func (cfg *TransformerConfig) WithBias(use bool) *TransformerConfig {
	cfg.UseBias = use
	return cfg
}

// CachedTransformer wraps a transformer with KV cache support.
type CachedTransformer struct {
	Config *TransformerConfig
	Caches []*attention.KVCache
}

// BuildCachedTransformer creates a transformer with KV cache support for training and generation.
func BuildCachedTransformer(ctx *context.Context, config *TransformerConfig) *CachedTransformer {
	return &CachedTransformer{
		Config: config,
		Caches: nil,
	}
}

// ForTraining: full-sequence forward, no KV cache.
func (ct *CachedTransformer) ForTraining() ModelFn {
	return func(ctx *context.Context, tokens *Node) *Node {
		return ct.forwardFull(ctx, tokens, false, 0)
	}
}

// ForGeneration: incremental forward with KV cache.
func (ct *CachedTransformer) ForGeneration() IncrementalModelFn {
	return func(ctx *context.Context, newTokens *Node, position int) *Node {
		return ct.forwardFull(ctx, newTokens, true, position)
	}
}

// forwardFull: shared path for training and generation.
func (ct *CachedTransformer) forwardFull(
	ctx *context.Context,
	tokens *Node,
	useCache bool,
	position int,
) *Node {
	cfg := ct.Config
	g := tokens.Graph()
	currentSeqLen := tokens.Shape().Dimensions[1]

	embedded := layers.Embedding(ctx.In("token_embed"), tokens, cfg.DType, cfg.VocabSize, cfg.EmbedDim)

	if embedded.Rank() == 2 {
		embedded = ExpandDims(embedded, 1)
	}

	x := embedded
	if !cfg.UseRoPE {
		posEmbedFull := ctx.In("pos_embed").VariableWithShape("embeddings",
			shapes.Make(cfg.DType, cfg.MaxPosEmbed, cfg.EmbedDim)).ValueGraph(g)

		var posEmbed *Node
		if useCache {
			posEmbed = Slice(posEmbedFull, AxisRange(position, position+currentSeqLen))
		} else {
			posEmbed = Slice(posEmbedFull, AxisRange(0, currentSeqLen))
		}

		posEmbed = ExpandDims(posEmbed, 0)
		posEmbed = BroadcastToShape(posEmbed, embedded.Shape())
		x = Add(embedded, posEmbed)
	}

	for layer := 0; layer < cfg.NumLayers; layer++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))

		residual := x
		var attn *Node

		if useCache {
			// Generation mode: use KV cache for efficient incremental decoding
			cache := attention.NewKVCache(
				ctx.In("layer").In(strconv.Itoa(layer)),
				"attn_cache",
				x.Shape().Dimensions[0],
				cfg.NumHeads,
				cfg.MaxPosEmbed,
				cfg.HeadDim,
				cfg.DType,
			)

			// Convert position int to a Node for the new WithKVCache API
			positionNode := Const(x.Graph(), int32(position))
			attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, cfg.NumHeads, cfg.HeadDim).
				WithKVCache(cache, positionNode)

			if cfg.UseRoPE {
				attnBuilder = attnBuilder.WithRoPE(cfg.RoPEBaseFreq)
			}

			if !cfg.UseBias {
				attnBuilder = attnBuilder.UseProjectionBias(false)
			}

			if cfg.Dropout > 0 {
				attnBuilder = attnBuilder.Dropout(cfg.Dropout)
			}

			attn = attnBuilder.Done()
		} else {
			// Training mode: no cache, use causal mask
			attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, cfg.NumHeads, cfg.HeadDim).
				UseCausalMask()

			if cfg.UseRoPE {
				attnBuilder = attnBuilder.WithRoPE(cfg.RoPEBaseFreq)
			}

			if !cfg.UseBias {
				attnBuilder = attnBuilder.UseProjectionBias(false)
			}

			if cfg.Dropout > 0 {
				attnBuilder = attnBuilder.Dropout(cfg.Dropout)
			}

			attn = attnBuilder.Done()
		}

		if cfg.UseLayerNorm {
			x = layers.LayerNormalization(layerCtx.In("norm1"), Add(residual, attn), -1).Done()
		} else {
			x = Add(residual, attn)
		}

		residual = x
		ff := layers.Dense(layerCtx.In("ff1"), x, cfg.UseBias, cfg.FFNDim)
		ff = activations.Gelu(ff)
		if cfg.Dropout > 0 {
			ff = layers.Dropout(layerCtx.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), cfg.Dropout))
		}
		ff = layers.Dense(layerCtx.In("ff2"), ff, cfg.UseBias, cfg.EmbedDim)

		if cfg.UseLayerNorm {
			x = layers.LayerNormalization(layerCtx.In("norm2"), Add(residual, ff), -1).Done()
		} else {
			x = Add(residual, ff)
		}
	}

	logits := layers.Dense(ctx.In("output"), x, false, cfg.VocabSize)

	return logits
}
