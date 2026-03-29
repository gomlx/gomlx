// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"
	"math"

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
	ParamUseRoPE       = "transformer_use_rope"
	ParamRoPEBaseFreq  = "transformer_rope_base_freq"
	ParamUseCausalMask = "transformer_use_causal_mask"
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
	VocabSize            int              // Vocabulary size
	EmbedDim             int              // Embedding dimension
	NumLayers            int              // Transformer layers
	NumHeads             int              // Attention heads per layer
	HeadDim              int              // Head dimension
	FFNDim               int              // Feed-forward hidden dimension
	MaxPosEmbed          int              // Max positional embedding length
	DType                dtypes.DType     // Data type
	Dropout              float64          // Dropout rate (0.0 = none)
	UseBias              bool             // Use bias in dense layers
	UseCausalMask        bool             // Use causal masking in attention layers
	ScaleTokenEmbeddings bool             // Whether to scale token embeddings by sqrt(m.EmbedDim).
	FinalNormalization   string           // e.g. "layer", "rms", "batch", "none" or "" (see layers.Normalization*)
	Architecture         Architecture     // e.g. ArchitectureStandard, ArchitectureGemma
	Normalization        string           // e.g. "layer", "rms", "batch", "none" or "" (see layers.Normalization*)
	NormEpsilon          float64          // Epsilon for normalization
	Activation           activations.Type // e.g. "gelu", "silu", "gelu_approximate"
	NumKVHeads           int              // For Grouped Query Attention (GQA), 0 means equal to NumHeads

	posEncoder pos.Encoder // Positional encoder (e.g. RoPE). If nil, standard absolute positional embeddings are used.

	// TransposedProjections indicates whether to assume linear weights are transposed (as [out_features, in_features]),
	// which is standard for PyTorch nn.Linear. This enables dropping in PyTorch models directly.
	TransposedProjections bool
}

// New creates a default transformer configuration.
func New(vocabSize, embedDim, numLayers, numHeads, headDim int) *Model {
	return &Model{
		VocabSize:            vocabSize,
		EmbedDim:             embedDim,
		NumLayers:            numLayers,
		NumHeads:             numHeads,
		HeadDim:              headDim,
		FFNDim:               embedDim * 4, // 4x expansion
		MaxPosEmbed:          512,
		DType:                dtypes.Float32,
		Dropout:              0.0,
		UseBias:              true,
		UseCausalMask:        true,
		ScaleTokenEmbeddings: false,
		FinalNormalization:   layers.NormalizationNone,
		Architecture:         ArchitectureStandard,
		Normalization:        layers.NormalizationLayerNorm,
		NormEpsilon:          1e-5,
		Activation:           activations.TypeGelu,
		NumKVHeads:           0,
		posEncoder:           nil,
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
	m.UseCausalMask = context.GetParamOr(ctx, ParamUseCausalMask, m.UseCausalMask)
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
		m.posEncoder = pos.NewRoPE(baseFreq)
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
	m.posEncoder = pos.NewRoPE(baseFreq)
	return m
}

// WithPositionalEncoder sets the positional encoder.
func (m *Model) WithPositionalEncoder(encoder pos.Encoder) *Model {
	m.posEncoder = encoder
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

// WithCausalMask toggles causal masking.
func (m *Model) WithCausalMask(use bool) *Model {
	m.UseCausalMask = use
	return m
}

// WithFinalNormalization toggles the final normalization after all attention layers.
// The values "none" or "" mean no final normalization.
func (m *Model) WithFinalNormalization(norm string) *Model {
	if norm == "" {
		norm = layers.NormalizationNone
	}
	m.FinalNormalization = norm
	return m
}

// WithScalingOfTokenEmbeddings configuration method configures whether to scale token
// embeddings by the square root of the embedding dimension. If true, it multiplies
// the token embeddings by math.Sqrt(m.EmbedDim). Some models (like Gemma and Llama)
// use this scaling factor.
func (m *Model) WithScalingOfTokenEmbeddings(useScaling bool) *Model {
	m.ScaleTokenEmbeddings = useScaling
	return m
}

// WithArchitecture sets the architectural style.
func (m *Model) WithArchitecture(arch Architecture) *Model {
	m.Architecture = arch
	if arch == ArchitectureGemma || arch == ArchitectureGemma3 {
		m.WithScalingOfTokenEmbeddings(true)
	}
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

// Logits builds the model including the last logits projection to predict the next token,
// for each element of the sequence.
//
// This form can be used to embed full sentences or for training.
//
//   - tokens: shaped [batchSize, seqLen], or simply [seqLen]
//   - mask: optional, if provided the shape must match tokens.Shape(), and it indicates
//     which tokens are valid (1) and which are padding (0). The attention mask (causal or not)
//     is computed taking into consideration the mask.
//
// It returns the logits of the last layer, typically shaped [batchSize, seqLen, vocabSize]
func (m *Model) Logits(ctx *context.Context, tokens, mask *Node) *Node {
	embeddings, _ := m.AllLayers(ctx, tokens, mask, false, 0)
	return m.LogitsFromEmbeddings(ctx, embeddings)
}

// MakeIterativeModelFn returns a "iterative" model function for iteratively (increasing sequence length, no KVCache)
// generation, using the decode package.
func MakeIterativeModelFn(m *Model) decode.IterativeModelFn {
	return func(ctx *context.Context, tokens *Node) *Node {
		return m.Logits(ctx, tokens, nil)
	}
}

// LogitsWithKVCache returns the forward path for the newTokens sequence, using the KV cache.
//
// - newTokens: shaped [batchSize, 1] with the new tokens to be processed.
// - position: the position of the new tokens in the sequence, only used for KV cache.
//
// **Experimental**: likely the KVCache will change in the future.
//
// It returns the logits of the last layer, typically shaped [batchSize, 1, vocabSize],
// and updates the KVCache stored as variables in the cache.
func (m *Model) LogitsWithKVCache(ctx *context.Context, newTokens *Node, position int) *Node {
	embeddings, _ := m.AllLayers(ctx, newTokens, nil, true, position)
	return m.LogitsFromEmbeddings(ctx, embeddings)
}

// MakeIncrementalModelFn returns a model function used by the decoder for incremental generation with KVCache,
// using the decode package.
func MakeIncrementalModelFn(m *Model) decode.IncrementalModelFn {
	return func(ctx *context.Context, newTokens *Node, position int) *Node {
		return m.LogitsWithKVCache(ctx, newTokens, position)
	}
}

// AllLayers takes the input tokens and creates the forward graph for the transformer model,
// returning the last layer and all the intermediate layers.
//
//   - tokens: shaped [batchSize, seqLen], or simply [seqLen]
//   - mask: optional, if provided the shape must match tokens.Shape(), and it indicates
//     which tokens are valid (1) and which are padding (0). The attention mask (causal or not)
//     is computed taking into consideration the mask.
//   - useKVCache: whether to use KV cache for the attention layers.
//   - position: the position of the new tokens in the sequence, only used for KV cache.
//
// See also Logits() or LogitsFromEmbeddings()if you want to get the logits for predicting the next token.
//
// It returns:
//
//   - lastLayer: the final hidden state of the last layer, shaped [batchSize, seqLen,hiddenSize].
//   - allLayers: the input to the first layer and the output of each layer.
//     It follows the HuggingFace convention, where the allLayers[0] is the input to the first attention layer,
//     and the following nodes in allLayers are the outputs of all NumHiddenLayers attention layers.
func (m *Model) AllLayers(ctx *context.Context, tokens, mask *Node, useKVCache bool, position int) (lastLayer *Node, allLayers []*Node) {
	allLayers = make([]*Node, 0, m.NumLayers+2)
	x := m.EmbedTokens(ctx, tokens)
	allLayers = append(allLayers, x)
	x = m.PrePositionalEncoder(ctx, x, position, useKVCache)
	allLayers = append(allLayers, x)
	// Apply all layers.
	for layer := range m.NumLayers {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", layer))
		x = m.ForwardLayer(layerCtx, x, mask, useKVCache, position)
		allLayers = append(allLayers, x)
	}
	if m.FinalNormalization != layers.NormalizationNone {
		x = m.normalize(ctx.In("final_norm"), x, m.FinalNormalization)
		if len(allLayers) > 0 {
			allLayers[len(allLayers)-1] = x
		}
	}
	return x, allLayers
}

// EmbedTokens returns the token embeddings for the given tokens using a lookup table.
// This is the very first step of the transformer model.
//
// If you want the model embeddings after of the full model, take the last layer of AllLayers.
//
// This step is done automatically by AllLayers or Logits, but if needed, it can
// be used separately by calling this method.
func (m *Model) EmbedTokens(ctx *context.Context, tokens *Node) *Node {
	// Tokens embedding table lookup.
	embedded := layers.Embedding(ctx.In("token_embed"), tokens, m.DType, m.VocabSize, m.EmbedDim)
	if embedded.Rank() == 2 {
		embedded = ExpandDims(embedded, 1)
	}
	if m.ScaleTokenEmbeddings {
		scale := Scalar(embedded.Graph(), m.DType, math.Sqrt(float64(m.EmbedDim)))
		embedded = Mul(embedded, scale)
	}
	return embedded
}

// PrePositionalEncoder applies the positional encoder to the given embeddings before
// the first transformer layer.
//
// This only applies if the positional encoder implements
// the pos.PreEncoder interface (some positional encoders run later).
// Otherwise, or if the encoder is nil, it simply returns x.
//
// This step is done automatically by AllLayers or Logits, but if needed, it can
// be used separately by calling this method.
func (m *Model) PrePositionalEncoder(ctx *context.Context, x *Node, position int, useKVCache bool) *Node {
	if m.posEncoder == nil {
		return x
	}
	preEnc, ok := m.posEncoder.(pos.PreEncoder)
	if !ok {
		// Positional encoder is not a pre-enconder.
		return x
	}

	g := x.Graph()
	seqAxis := 1 // Sequence is always the axis 1 for now.
	seqLen := x.Shape().Dimensions[seqAxis]
	posIndices := pos.SequentialPositions(g, Scalar(g, dtypes.Int32, position), seqLen)
	return preEnc.PreEncode(x, posIndices, seqAxis)
}

// LogitsFromEmbeddings takes the embeddings of the last attention layer (returned by AllLayers) and computes
// the logits over the vocabulary size. This is the last step of the model.
//
// This step is done automatically by Logits (which builds the full forward path from the tokens), but if needed, it can
// be used separately by calling this method.
func (m *Model) LogitsFromEmbeddings(ctx *context.Context, embeddings *Node) *Node {
	return layers.Dense(ctx.In("output"), embeddings, false, m.VocabSize)
}

// ForwardLayer executes a single transformer layer block depending on the configured architecture.
//
// - ctx: context must be already scoped for the layer, e.g. ctx.In("layer_0")
// - x: features coming from the previous layer (or token embedding table), shape [batchSize, seqLen, embedDim]
// - mask: (optional, can be nil) shaped [batchSize, seqLen]
// - useCache: if true, use KV cache -- Experimental: KVCache will change.
// - position: position of the first token in the sequence, used only if using KV cache, otherwise we assum 0.
//
// It returns the output of the layer, shape [batchSize, seqLen, embedDim].
//
// This step is done automatically by AllLayers or Logits, but if needed, it can
// be used separately by calling this method.
func (m *Model) ForwardLayer(ctx *context.Context, x, mask *Node, useCache bool, position int) *Node {
	if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 {
		return m.forwardLayerGemma(ctx, x, mask, useCache, position)
	}
	return m.forwardLayerStandard(ctx, x, mask, useCache, position)
}

func (m *Model) forwardLayerStandard(layerCtx *context.Context, x, mask *Node, useCache bool, position int) *Node {
	residual := x
	var attn *Node

	if useCache {
		positionNode := Const(x.Graph(), int32(position))
		attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, m.NumHeads, m.HeadDim).
			WithKVCache(m.MaxPosEmbed, positionNode).
			UseTransposedWeights(m.TransposedProjections)
		if mask != nil {
			attnBuilder = attnBuilder.WithMask(mask)
		}

		if m.posEncoder != nil {
			attnBuilder = attnBuilder.WithPositionalEncoder(m.posEncoder)
		}
		if !m.UseBias {
			attnBuilder = attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder = attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	} else {
		attnBuilder := attention.SelfAttention(layerCtx.In("attn"), x, m.NumHeads, m.HeadDim).
			UseTransposedWeights(m.TransposedProjections)
		if mask != nil {
			attnBuilder = attnBuilder.WithMask(mask)
		}
		if m.UseCausalMask {
			attnBuilder = attnBuilder.WithCausalMask(true)
		}

		if m.posEncoder != nil {
			attnBuilder = attnBuilder.WithPositionalEncoder(m.posEncoder)
		}
		if !m.UseBias {
			attnBuilder = attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder = attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	}

	x = Add(residual, attn)
	x = m.normalize(layerCtx.In("norm1"), x, m.Normalization)

	residual = x
	ff := m.dense(layerCtx.In("ff1"), x, m.UseBias, m.FFNDim)
	ff = activations.Apply(m.Activation, ff)
	if m.Dropout > 0 {
		ff = layers.Dropout(layerCtx.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), m.Dropout))
	}
	ff = m.dense(layerCtx.In("ff2"), ff, m.UseBias, m.EmbedDim)
	x = Add(residual, ff)
	x = m.normalize(layerCtx.In("norm2"), x, m.Normalization)
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

func (m *Model) normalize(ctx *context.Context, operand *Node, normType string) *Node {
	if normType == layers.NormalizationNone {
		return operand
	}
	switch normType {
	case layers.NormalizationRMSNorm:
		builder := layers.RMSNorm(ctx, operand).WithEpsilon(m.NormEpsilon)
		if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 {
			builder = builder.WithScaleOffset(1.0)
		}
		return builder.Done()
	case layers.NormalizationLayerNorm:
		return layers.LayerNormalization(ctx, operand, -1).Epsilon(m.NormEpsilon).Done()
	default:
		exceptions.Panicf("unsupported normalization type: %q", normType)
		return nil
	}
}

func (m *Model) forwardLayerGemma(layerCtx *context.Context, x, mask *Node, useCache bool, position int) *Node {
	residual := x

	// Pre-attention normalization.
	x = m.normalize(layerCtx.In("input_norm"), x, m.Normalization)

	// Attention.
	var attn *Node
	attnBuilder := attention.SelfAttention(layerCtx.In("self_attn"), x, m.NumHeads, m.HeadDim).
		UseTransposedWeights(m.TransposedProjections)
	if mask != nil {
		attnBuilder.WithMask(mask)
	}
	if m.NumKVHeads > 0 && m.NumKVHeads != m.NumHeads {
		attnBuilder.WithNumKVHeads(m.NumKVHeads)
	}
	if useCache {
		positionNode := Const(x.Graph(), int32(position))
		attnBuilder.WithKVCache(m.MaxPosEmbed, positionNode)
	} else if m.UseCausalMask {
		attnBuilder.WithCausalMask(true)
	}
	if m.posEncoder != nil {
		attnBuilder.WithPositionalEncoder(m.posEncoder)
	}
	if !m.UseBias {
		attnBuilder.UseProjectionBias(false)
	}
	if m.Architecture == ArchitectureGemma3 {
		attnBuilder.WithQKRMSNorm(m.NormEpsilon)
	}
	if m.Dropout > 0 {
		dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
		attnBuilder.WithDropout(dropoutRate)
	}
	attn = attnBuilder.Done()

	// Post-attention normalization.
	attn = m.normalize(layerCtx.In("post_attention_norm"), attn, m.Normalization)
	x = Add(residual, attn)
	residual = x

	// Pre-feedforward normalization
	x = m.normalize(layerCtx.In("pre_feedforward_norm"), x, m.Normalization)

	// Gemma uses SwiGLU: gate_proj, up_proj, down_proj
	ffCtx := layerCtx.In("mlp")
	gate := m.dense(ffCtx.In("gate_proj"), x, m.UseBias, m.FFNDim)
	up := m.dense(ffCtx.In("up_proj"), x, m.UseBias, m.FFNDim)
	switchedNode := activations.Apply(m.Activation, gate)
	ff := Mul(switchedNode, up)

	if m.Dropout > 0 {
		ff = layers.Dropout(ffCtx.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), m.Dropout))
	}
	ff = m.dense(ffCtx.In("down_proj"), ff, m.UseBias, m.EmbedDim)

	// Post-feedforward normalization
	ff = m.normalize(layerCtx.In("post_feedforward_norm"), ff, m.Normalization)
	x = Add(residual, ff)
	return x
}
