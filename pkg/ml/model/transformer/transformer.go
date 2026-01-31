// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/decode"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
)

// Hyperparameter keys for context configuration
const (
	ParamVocabSize    = "transformer_vocab_size"
	ParamEmbedDim     = "transformer_embed_dim"
	ParamNumLayers    = "transformer_num_layers"
	ParamNumHeads     = "transformer_num_heads"
	ParamHeadDim      = "transformer_head_dim"
	ParamFFNDim       = "transformer_ffn_dim"
	ParamMaxPosEmbed  = "transformer_max_pos_embed"
	ParamDType        = "transformer_dtype"
	ParamDropout      = "transformer_dropout"
	ParamUseLayerNorm = "transformer_use_layer_norm"
	ParamUseBias      = "transformer_use_bias"

	// Legacy RoPE parameters, used to configure PosEmbed if set.
	ParamUseRoPE      = "transformer_use_rope"
	ParamRoPEBaseFreq = "transformer_rope_base_freq"
)

// Model holds configuration for building a cached transformer model.
// Model configures a cached transformer model.
type Model struct {
	VocabSize    int          // Vocabulary size
	EmbedDim     int          // Embedding dimension
	NumLayers    int          // Transformer layers
	NumHeads     int          // Attention heads per layer
	HeadDim      int          // Head dimension
	FFNDim       int          // Feed-forward hidden dimension
	MaxPosEmbed  int          // Max positional embedding length
	DType        dtypes.DType // Data type
	Dropout      float64      // Dropout rate (0.0 = none)
	UseLayerNorm bool         // Use layer normalization
	UseBias      bool         // Use bias in dense layers

	PosEmbed pos.Encoder // Positional encoder (e.g. RoPE). If nil, standard absolute positional embeddings are used.
}

// New creates a default transformer configuration.
func New(vocabSize, embedDim, numLayers, numHeads, headDim int) *Model {
	return &Model{
		VocabSize:    vocabSize,
		EmbedDim:     embedDim,
		NumLayers:    numLayers,
		NumHeads:     numHeads,
		HeadDim:      headDim,
		FFNDim:       embedDim * 4, // 4x expansion
		MaxPosEmbed:  512,
		DType:        dtypes.Float32,
		Dropout:      0.0,
		UseLayerNorm: true,
		UseBias:      true,
		PosEmbed:     nil,
	}
}

// NewFromContext creates a transformer model configured from context hyperparameters.
// It reads parameters with the following keys (with defaults):
//   - transformer_vocab_size (required, no default)
//   - transformer_embed_dim (required, no default)
//   - transformer_num_layers (required, no default)
//   - transformer_num_heads (required, no default)
//   - transformer_head_dim (required, no default)
//   - transformer_ffn_dim (default: embed_dim * 4)
//   - transformer_max_pos_embed (default: 512)
//   - transformer_dtype (default: "float32")
//   - transformer_dropout (default: 0.0)
//   - transformer_use_rope (default: false)
//   - transformer_rope_base_freq (default: 10000.0)
//   - transformer_use_layer_norm (default: true)
//   - transformer_use_bias (default: true)
//
// Example usage:
//
//	ctx.SetParams(map[string]any{
//	    "transformer_vocab_size": 50257,
//	    "transformer_embed_dim": 768,
//	    "transformer_num_layers": 12,
//	    "transformer_num_heads": 12,
//	    "transformer_head_dim": 64,
//	})
//	model := transformer.NewFromContext(ctx)
func NewFromContext(ctx *context.Context) *Model {
	// Required parameters
	vocabSize, found := ctx.GetParam(ParamVocabSize)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in context", ParamVocabSize))
	}
	embedDim, found := ctx.GetParam(ParamEmbedDim)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in context", ParamEmbedDim))
	}
	numLayers, found := ctx.GetParam(ParamNumLayers)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in context", ParamNumLayers))
	}
	numHeads, found := ctx.GetParam(ParamNumHeads)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in context", ParamNumHeads))
	}
	headDim, found := ctx.GetParam(ParamHeadDim)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in context", ParamHeadDim))
	}

	// Create model with required parameters
	model := New(
		vocabSize.(int),
		embedDim.(int),
		numLayers.(int),
		numHeads.(int),
		headDim.(int),
	)

	// Apply optional parameters from context
	model.FromContext(ctx)

	return model
}

// FromContext configures the model with optional hyperparameters from the context.
// This allows fine-tuning an existing model configuration.
func (m *Model) FromContext(ctx *context.Context) *Model {
	// Optional parameters with defaults
	m.FFNDim = context.GetParamOr(ctx, ParamFFNDim, m.FFNDim)
	m.MaxPosEmbed = context.GetParamOr(ctx, ParamMaxPosEmbed, m.MaxPosEmbed)
	m.Dropout = context.GetParamOr(ctx, ParamDropout, m.Dropout)
	m.UseLayerNorm = context.GetParamOr(ctx, ParamUseLayerNorm, m.UseLayerNorm)
	m.UseBias = context.GetParamOr(ctx, ParamUseBias, m.UseBias)

	// Legacy RoPE configuration from context
	useRoPE := context.GetParamOr(ctx, ParamUseRoPE, false)
	if useRoPE {
		baseFreq := context.GetParamOr(ctx, ParamRoPEBaseFreq, 10000.0)
		m.PosEmbed = pos.NewRoPE(baseFreq)
	}

	// Handle dtype separately since it's a string
	dtypeStr := context.GetParamOr(ctx, ParamDType, "")
	if dtypeStr != "" {
		dtype, err := dtypes.DTypeString(dtypeStr)
		if err != nil || !dtype.IsFloat() {
			panic(fmt.Sprintf("Invalid hyperparameter value %s=%q", ParamDType, dtypeStr))
		}
		m.DType = dtype
	}

	return m
}

// WithFFNDim sets FFN dimension.
func (m *Model) WithFFNDim(dim int) *Model {
	m.FFNDim = dim
	return m
}

// WithMaxPosEmbed sets max positional embedding length.
func (m *Model) WithMaxPosEmbed(maxLen int) *Model {
	m.MaxPosEmbed = maxLen
	return m
}

// WithDType sets data type.
func (m *Model) WithDType(dtype dtypes.DType) *Model {
	m.DType = dtype
	return m
}

// WithDropout sets dropout rate.
func (m *Model) WithDropout(rate float64) *Model {
	m.Dropout = rate
	return m
}

// WithRoPE enables RoPE with base frequency.
func (m *Model) WithRoPE(baseFreq float64) *Model {
	m.PosEmbed = pos.NewRoPE(baseFreq)
	return m
}

// WithPositionalEmbedding sets the positional encoder.
func (m *Model) WithPositionalEmbedding(encoder pos.Encoder) *Model {
	m.PosEmbed = encoder
	return m
}

// WithLayerNorm toggles layer normalization.
func (m *Model) WithLayerNorm(use bool) *Model {
	m.UseLayerNorm = use
	return m
}

// WithBias toggles dense bias.
func (m *Model) WithBias(use bool) *Model {
	m.UseBias = use
	return m
}

// ForTraining returns a model function for training (full-sequence forward, no KV cache).
func (m *Model) ForTraining() decode.IterativeModelFn {
	return func(ctx *context.Context, tokens *Node) *Node {
		return m.forwardFull(ctx, tokens, false, 0)
	}
}

// ForGeneration returns a model function for generation (incremental forward with KV cache).
func (m *Model) ForGeneration() decode.IncrementalModelFn {
	return func(ctx *context.Context, newTokens *Node, position int) *Node {
		return m.forwardFull(ctx, newTokens, true, position)
	}
}

// forwardFull: shared path for training and generation.
func (m *Model) forwardFull(
	ctx *context.Context,
	tokens *Node,
	useCache bool,
	position int,
) *Node {
	cfg := m
	g := tokens.Graph()
	currentSeqLen := tokens.Shape().Dimensions[1]

	embedded := layers.Embedding(ctx.In("token_embed"), tokens, cfg.DType, cfg.VocabSize, cfg.EmbedDim)

	if embedded.Rank() == 2 {
		embedded = ExpandDims(embedded, 1)
	}

	x := embedded
	if cfg.PosEmbed == nil {
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
			// Convert position int to a Node for the WithKVCache API
			positionNode := Const(x.Graph(), int32(position))
			attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, cfg.NumHeads, cfg.HeadDim).
				WithKVCache(cfg.MaxPosEmbed, positionNode)

			if cfg.PosEmbed != nil {
				attnBuilder = attnBuilder.WithPositionalEncoder(cfg.PosEmbed)
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

			if cfg.PosEmbed != nil {
				attnBuilder = attnBuilder.WithPositionalEncoder(cfg.PosEmbed)
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
