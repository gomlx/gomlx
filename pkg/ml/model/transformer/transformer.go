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
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// Hyperparameter keys for context configuration
const (
	ParamVocabSize   = "transformer_vocab_size"
	ParamEmbedDim    = "transformer_embed_dim"
	ParamNumLayers   = "transformer_num_layers"
	ParamNumHeads    = "transformer_num_heads"
	ParamHeadDim     = "transformer_head_dim"
	ParamFFNDim      = "transformer_ffn_dim"
	ParamMaxPosEmbed = "transformer_max_pos_embed"
	ParamDType       = "transformer_dtype"
	ParamDropout     = "transformer_dropout"

	// ParamUseLayerNorm
	// Deprecated: use ParamNormalization instead.
	ParamUseLayerNorm = "transformer_use_layer_norm"

	ParamUseBias       = "transformer_use_bias"
	ParamArchitecture  = "transformer_architecture"
	ParamNormalization = "transformer_normalization"
	ParamNormEpsilon   = "transformer_norm_epsilon"
	ParamActivation    = "transformer_activation"
	ParamNumKVHeads    = "transformer_num_kv_heads"

	// Legacy RoPE parameters, used to configure PosEmbed if set.
	ParamUseRoPE      = "transformer_use_rope"
	ParamRoPEBaseFreq = "transformer_rope_base_freq"
)

// Model holds configuration for building a cached transformer model.
// Architecture is an enum for the supported transformer architectures.
//
// It is converted to and from string by using the generated ArchitectureString function.
type Architecture int

const (
	ArchitectureStandard Architecture = iota
	ArchitectureGemma
	ArchitectureGemma3
)

//go:generate go tool enumer -type Architecture -trimprefix=Architecture -output=gen_architecture_enumer.go transformer.go

// Model configures a cached transformer model.
type Model struct {
	VocabSize     int              // Vocabulary size
	EmbedDim      int              // Embedding dimension
	NumLayers     int              // Transformer layers
	NumHeads      int              // Attention heads per layer
	HeadDim       int              // Head dimension
	FFNDim        int              // Feed-forward hidden dimension
	MaxPosEmbed   int              // Max positional embedding length
	DType         dtypes.DType     // Data type
	Dropout       float64          // Dropout rate (0.0 = none)
	UseBias       bool             // Use bias in dense layers
	Architecture  Architecture     // e.g. ArchitectureStandard, ArchitectureGemma
	Normalization string           // e.g. "layer", "rms", "batch", "none" or "" (see layers.Normalization*)
	NormEpsilon   float64          // Epsilon for normalization
	Activation    activations.Type // e.g. "gelu", "silu", "gelu_approximate"
	NumKVHeads    int              // For Grouped Query Attention (GQA), 0 means equal to NumHeads

	PosEmbed pos.Encoder // Positional encoder (e.g. RoPE). If nil, standard absolute positional embeddings are used.

	// TransposedProjections indicates whether to assume linear weights are transposed (as [out_features, in_features]),
	// which is standard for PyTorch nn.Linear. This enables dropping in PyTorch models directly.
	TransposedProjections bool
}

// New creates a default transformer configuration.
func New(vocabSize, embedDim, numLayers, numHeads, headDim int) *Model {
	return &Model{
		VocabSize:     vocabSize,
		EmbedDim:      embedDim,
		NumLayers:     numLayers,
		NumHeads:      numHeads,
		HeadDim:       headDim,
		FFNDim:        embedDim * 4, // 4x expansion
		MaxPosEmbed:   512,
		DType:         dtypes.Float32,
		Dropout:       0.0,
		UseBias:       true,
		Architecture:  ArchitectureStandard,
		Normalization: layers.NormalizationLayerNorm,
		NormEpsilon:   1e-5,
		Activation:    activations.TypeGelu,
		NumKVHeads:    0,
		PosEmbed:      nil,
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

	// Deprecated way to specify LayerNorm:
	useLayerNorm := context.GetParamOr(ctx, ParamUseLayerNorm, true)
	if useLayerNorm {
		m.WithLayerNorm(true)
	}
	m.Normalization = context.GetParamOr(ctx, ParamNormalization, m.Normalization)
	m.WithNormalization(m.Normalization)
	m.UseBias = context.GetParamOr(ctx, ParamUseBias, m.UseBias)
	archStr := context.GetParamOr(ctx, ParamArchitecture, m.Architecture.String())
	if arch, err := ArchitectureString(archStr); err == nil {
		m.Architecture = arch
	} else {
		exceptions.Panicf("invalid architecture name %q: options are %v", archStr, ArchitectureValues())
	}
	m.NormEpsilon = context.GetParamOr(ctx, ParamNormEpsilon, m.NormEpsilon)
	m.Activation = activations.FromName(context.GetParamOr(ctx, ParamActivation, m.Activation.String()))
	m.NumKVHeads = context.GetParamOr(ctx, ParamNumKVHeads, m.NumKVHeads)

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
//
// Deprecated: use WithNormalization("layer"), WithNormalization("none") or some other value instead.
func (m *Model) WithLayerNorm(use bool) *Model {
	if use {
		m.Normalization = layers.NormalizationLayerNorm
	} else {
		m.Normalization = layers.NormalizationNone
	}
	return m
}

// WithBias toggles dense bias.
func (m *Model) WithBias(use bool) *Model {
	m.UseBias = use
	return m
}

// WithArchitecture sets the architectural style.
func (m *Model) WithArchitecture(arch Architecture) *Model {
	m.Architecture = arch
	return m
}

// WithNormalization sets the normalization type ("layer", "rms", "batch", "none" or "").
//
// The values "none" or "" mean no normalization.
//
// See constants layers.NormalizationLayerNorm ("layer"), layers.NormalizationRMSNorm ("rms"), layers.NormalizationBatchNorm ("batch")
// and layers.NormalizationNone ("none").
func (m *Model) WithNormalization(norm string) *Model {
	if norm == "" {
		norm = layers.NormalizationNone
	}
	m.Normalization = norm
	return m
}

// WithNormEpsilon sets the epsilon value used for normalization layers.
func (m *Model) WithNormEpsilon(eps float64) *Model {
	m.NormEpsilon = eps
	return m
}

// WithActivation sets the activation function type.
func (m *Model) WithActivation(activation activations.Type) *Model {
	m.Activation = activation
	return m
}

// WithNumKVHeads enables Grouped Query Attention (GQA) if different from NumHeads.
func (m *Model) WithNumKVHeads(num int) *Model {
	m.NumKVHeads = num
	return m
}

// WithTransposedWeights configures the model to assume linear weights are transposed (as [out_features, in_features]),
// which is the format used by PyTorch nn.Linear. This enables loading PyTorch models directly.
func (m *Model) WithTransposedWeights(transposed bool) *Model {
	m.TransposedProjections = transposed
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
		x = m.ForwardLayer(ctx, x, layer, useCache, position)
	}

	logits := layers.Dense(ctx.In("output"), x, false, cfg.VocabSize)

	return logits
}

// ForwardLayer executes a single transformer layer block depending on the configured architecture.
func (m *Model) ForwardLayer(ctx *context.Context, x *Node, layerIdx int, useCache bool, position int) *Node {
	layerCtx := ctx.In(fmt.Sprintf("layer_%d", layerIdx))

	if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 {
		return m.forwardLayerGemma(layerCtx, x, useCache, position)
	}
	return m.forwardLayerStandard(layerCtx, x, useCache, position)
}

func (m *Model) forwardLayerStandard(layerCtx *context.Context, x *Node, useCache bool, position int) *Node {
	cfg := m
	residual := x
	var attn *Node

	if useCache {
		positionNode := Const(x.Graph(), int32(position))
		attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, cfg.NumHeads, cfg.HeadDim).
			WithKVCache(cfg.MaxPosEmbed, positionNode).
			UseTransposedWeights(cfg.TransposedProjections)

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
		attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, cfg.NumHeads, cfg.HeadDim).
			UseCausalMask().
			UseTransposedWeights(cfg.TransposedProjections)

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

	x = Add(residual, attn)
	x = m.normalize(layerCtx.In("norm1"), x)

	residual = x
	ff := m.dense(layerCtx.In("ff1"), x, cfg.UseBias, cfg.FFNDim)
	ff = activations.Apply(cfg.Activation, ff)
	if cfg.Dropout > 0 {
		ff = layers.Dropout(layerCtx.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), cfg.Dropout))
	}
	ff = m.dense(layerCtx.In("ff2"), ff, cfg.UseBias, cfg.EmbedDim)
	x = Add(residual, ff)
	x = m.normalize(layerCtx.In("norm2"), x)
	return x
}

// normalize according to configuration.
// dense behaves like layers.dense but checks whether the model is configured to use transposed projections.
// If it is, it computes the dense layer using DotGeneral (Einsum), expecting weights in the format [outDim, inDim].
// PyTorch nn.Linear stores its matrix in this transposed format.
func (m *Model) dense(ctx *context.Context, op *Node, useBias bool, outputDims ...int) *Node {
	if !m.TransposedProjections {
		return layers.Dense(ctx, op, useBias, outputDims...)
	}
	ctx = ctx.In("dense")
	g := op.Graph()
	inDim := op.Shape().Dim(-1)
	outDim := 1
	for _, d := range outputDims {
		outDim *= d
	}
	wVar := ctx.VariableWithShape("weights", shapes.Make(op.DType(), outDim, inDim))
	w := wVar.ValueGraph(g)
	y := DotGeneral(op, []int{-1}, nil, w, []int{1}, nil)

	if useBias {
		bVar := ctx.VariableWithShape("biases", shapes.Make(op.DType(), outDim))
		y = Add(y, bVar.ValueGraph(g))
	}

	if len(outputDims) > 1 {
		newDims := make([]int, op.Rank()-1+len(outputDims))
		copy(newDims, op.Shape().Dimensions[:op.Rank()-1])
		copy(newDims[op.Rank()-1:], outputDims)
		y = Reshape(y, newDims...)
	}
	return y
}

func (m *Model) normalize(ctx *context.Context, operand *Node) *Node {
	if m.Normalization == layers.NormalizationNone {
		return operand
	}
	switch m.Normalization {
	case layers.NormalizationRMSNorm:
		builder := layers.RMSNorm(ctx, operand).WithEpsilon(m.NormEpsilon)
		if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 {
			builder = builder.WithScaleOffset(1.0)
		}
		return builder.Done()
	case layers.NormalizationLayerNorm:
		return layers.LayerNormalization(ctx, operand, -1).Epsilon(m.NormEpsilon).Done()
	default:
		exceptions.Panicf("unsupported normalization type: %q", m.Normalization)
		return nil
	}
}

func (m *Model) forwardLayerGemma(layerCtx *context.Context, x *Node, useCache bool, position int) *Node {
	cfg := m
	residual := x

	// Pre-attention normalization.
	x = m.normalize(layerCtx.In("input_norm"), x)

	// Attention.
	var attn *Node
	attnBuilder := attention.SelfAttention(layerCtx.In("self_attn"), x, cfg.NumHeads, cfg.HeadDim).
		UseTransposedWeights(m.TransposedProjections)
	if cfg.NumKVHeads > 0 && cfg.NumKVHeads != cfg.NumHeads {
		attnBuilder.SetNumKVHeads(cfg.NumKVHeads)
	}
	if useCache {
		positionNode := Const(x.Graph(), int32(position))
		attnBuilder = attnBuilder.WithKVCache(cfg.MaxPosEmbed, positionNode)
	} else {
		attnBuilder = attnBuilder.UseCausalMask()
	}
	if cfg.PosEmbed != nil {
		attnBuilder = attnBuilder.WithPositionalEncoder(cfg.PosEmbed)
	}
	if !cfg.UseBias {
		attnBuilder = attnBuilder.UseProjectionBias(false)
	}
	if m.Architecture == ArchitectureGemma3 {
		attnBuilder = attnBuilder.WithQKRMSNorm(cfg.NormEpsilon)
	}
	if cfg.Dropout > 0 {
		attnBuilder = attnBuilder.Dropout(cfg.Dropout)
	}
	attn = attnBuilder.Done()

	// Post-attention normalization.
	attn = m.normalize(layerCtx.In("post_attention_norm"), attn)
	x = Add(residual, attn)
	residual = x

	// Pre-feedforward normalization
	x = m.normalize(layerCtx.In("pre_feedforward_norm"), x)

	// Gemma uses SwiGLU: gate_proj, up_proj, down_proj
	ffCtx := layerCtx.In("mlp")
	gate := m.dense(ffCtx.In("gate_proj"), x, cfg.UseBias, cfg.FFNDim)
	up := m.dense(ffCtx.In("up_proj"), x, cfg.UseBias, cfg.FFNDim)
	switchedNode := activations.Apply(cfg.Activation, gate)
	ff := Mul(switchedNode, up)

	if cfg.Dropout > 0 {
		ff = layers.Dropout(ffCtx.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), cfg.Dropout))
	}
	ff = m.dense(ffCtx.In("down_proj"), ff, cfg.UseBias, cfg.EmbedDim)

	// Post-feedforward normalization
	ff = m.normalize(layerCtx.In("post_feedforward_norm"), ff)
	x = Add(residual, ff)
	return x
}
