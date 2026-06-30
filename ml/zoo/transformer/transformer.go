// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"fmt"
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
	"github.com/gomlx/gomlx/support/exceptions"
)

// Hyperparameter keys for scope configuration
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

	ParamUseBias               = "transformer_use_bias"
	ParamArchitecture          = "transformer_architecture"
	ParamNormalization         = "transformer_normalization"
	ParamNormEpsilon           = "transformer_norm_epsilon"
	ParamActivation            = "transformer_activation"
	ParamNumKVHeads            = "transformer_num_kv_heads"
	ParamGlobalHeadDim         = "transformer_global_head_dim"
	ParamNumKVSharedLayers     = "transformer_num_kv_shared_layers"
	ParamAttentionScoreSoftCap = "transformer_attention_score_soft_cap"
	ParamFinalLogitSoftCap     = "transformer_final_logit_soft_cap"

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
	ArchitectureGemma4
)

//go:generate go tool enumer -type Architecture -trimprefix=Architecture -output=gen_architecture_enumer.go transformer.go

type LayerType = attention.LayerType

const (
	GlobalLayer = attention.GlobalLayer
	LocalLayer  = attention.LocalLayer
)

type KVCacheNodes = kvcache.KVCacheNodes

type KVCache = kvcache.KVCache

// NewKVCache creates a new kvcache.KVCache.
func NewKVCache() *kvcache.KVCache {
	return kvcache.NewKVCache()
}

// Model configures a cached transformer model.
type Model struct {
	VocabSize            int             // Vocabulary size
	EmbedDim             int             // Embedding dimension
	NumLayers            int             // Transformer layers
	NumHeads             int             // Attention heads per layer
	HeadDim              int             // Head dimension
	FFNDim               int             // Feed-forward hidden dimension
	MaxPosEmbed          int             // Max positional embedding length
	DType                dtypes.DType    // Data type
	Dropout              float64         // Dropout rate (0.0 = none)
	UseBias              bool            // Use bias in dense layers
	UseCausalMask        bool            // Use causal masking in attention layers
	ScaleTokenEmbeddings bool            // Whether to scale token embeddings by sqrt(m.EmbedDim).
	FinalNormalization   string          // e.g. "layer", "rms", "batch", "none" or "" (see layers.Normalization*)
	Architecture         Architecture    // e.g. ArchitectureStandard, ArchitectureGemma
	EmbedNormalization   string          // e.g. "layer", "rms", "none" or ""
	TokenTypeEmbedSize   int             // e.g. Number of token types, 2 for BERT. See WithTokenTypeEmbedding.
	Normalization        string          // e.g. "layer", "rms", "batch", "none" or "" (see layers.Normalization*)
	NormEpsilon          float64         // Epsilon for normalization
	Activation           activation.Type // e.g. "gelu", "silu", "gelu_approximate"
	NumKVHeads           int             // For Grouped Query Attention (GQA), 0 means equal to NumHeads

	posEncoder       pos.Encoder         // Positional encoder (e.g. RoPE). If nil, standard absolute positional embeddings are used.
	layerPosEncoders map[int]pos.Encoder // Custom positional encoders per layer

	LayerTypes    []LayerType // Defines the type per layer (e.g. GlobalLayer, LocalLayer)
	SlidingWindow int         // Size of the sliding window for LocalLayer layers

	// TransposedProjections indicates whether to assume linear weights are transposed (as [out_features, in_features]),
	// which is standard for PyTorch nn.Linear. This enables dropping in PyTorch models directly.
	TransposedProjections bool

	KVCache *kvcache.KVCache

	NumKVSharedLayers int // Number of KV shared layers at the end of the model

	AttentionScoreSoftCap float64 // Softcap for attention scores
	FinalLogitSoftCap     float64 // Logit softcap for vocabulary prediction

	// Per-Layer Embeddings (PLE) configuration for Gemma 4
	VocabSizePerLayerInput       int
	HiddenSizePerLayerInput      int
	PerLayerInputScale           float64
	PerLayerModelProjectionScale float64
	RMSNormOffset                float64 // Offset added to RMSNorm weights, default is 1.0 (Gemma 1/2/3). Set to 0.0 for Gemma 4.
	GlobalHeadDim                int
	QueryKeyScale                float64
}

// New creates a default transformer configuration.
func New(vocabSize, embedDim, numLayers, numHeads, headDim int) *Model {
	return &Model{
		VocabSize:                    vocabSize,
		EmbedDim:                     embedDim,
		NumLayers:                    numLayers,
		NumHeads:                     numHeads,
		HeadDim:                      headDim,
		FFNDim:                       embedDim * 4, // 4x expansion
		MaxPosEmbed:                  512,
		DType:                        dtypes.Float32,
		Dropout:                      0.0,
		UseBias:                      true,
		UseCausalMask:                true,
		ScaleTokenEmbeddings:         false,
		EmbedNormalization:           layers.NormalizationNone,
		FinalNormalization:           layers.NormalizationNone,
		Architecture:                 ArchitectureStandard,
		Normalization:                layers.NormalizationLayerNorm,
		NormEpsilon:                  1e-5,
		Activation:                   activation.TypeGelu,
		NumKVHeads:                   0,
		posEncoder:                   nil,
		KVCache:                      NewKVCache(),
		PerLayerInputScale:           1,
		PerLayerModelProjectionScale: 1,
		RMSNormOffset:                1,
	}
}

func (m *Model) populateOrderedScopes(scope *model.Scope) {
	if len(m.KVCache.OrderedScopes) > 0 {
		return
	}
	scopes := make([]string, m.NumLayers)
	for i := range m.NumLayers {
		var path string
		if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 || m.Architecture == ArchitectureGemma4 {
			path = scope.At("layer_%d", i).At("self_attn").Scope()
		} else {
			path = scope.At("layer_%d", i).At("attn").Scope()
		}
		scopes[i] = path
	}
	m.KVCache.OrderedScopes = scopes
}

func (m *Model) populateLayerTypes(scope *model.Scope) {
	if len(m.KVCache.LayerTypes) > 0 {
		return
	}
	for i := range m.NumLayers {
		var path string
		if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 || m.Architecture == ArchitectureGemma4 {
			path = scope.At("layer_%d", i).At("self_attn").Scope()
		} else {
			path = scope.At("layer_%d", i).At("attn").Scope()
		}

		lt := GlobalLayer
		if i < len(m.LayerTypes) {
			lt = m.LayerTypes[i]
		}
		m.KVCache.LayerTypes[path] = lt
	}
}

// PopulateKVCacheConfigs initializes the internal KVCache configuration's OrderedScopes
// and LayerTypes based on the model's architecture and the execution scope.
func (m *Model) PopulateKVCacheConfigs(scope *model.Scope) {
	m.populateOrderedScopes(scope)
	m.populateLayerTypes(scope)
}

// NewFromScope creates a transformer model configured from the hyperparameters in Scope:
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
//	scope.SetParams(map[string]any{
//	    "transformer_vocab_size": 50257,
//	    "transformer_embed_dim": 768,
//	    "transformer_num_layers": 12,
//	    "transformer_num_heads": 12,
//	    "transformer_head_dim": 64,
//	})
//	model := transformer.NewFromScope(scope)
func NewFromScope(scope *model.Scope) *Model {
	// Required parameters
	vocabSize, found := scope.GetParam(ParamVocabSize)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in scope", ParamVocabSize))
	}
	embedDim, found := scope.GetParam(ParamEmbedDim)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in scope", ParamEmbedDim))
	}
	numLayers, found := scope.GetParam(ParamNumLayers)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in scope", ParamNumLayers))
	}
	numHeads, found := scope.GetParam(ParamNumHeads)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in scope", ParamNumHeads))
	}
	headDim, found := scope.GetParam(ParamHeadDim)
	if !found {
		panic(fmt.Sprintf("Required hyperparameter %q not found in scope", ParamHeadDim))
	}

	// Create model with required parameters
	m := New(
		vocabSize.(int),
		embedDim.(int),
		numLayers.(int),
		numHeads.(int),
		headDim.(int),
	)

	// Apply optional parameters from scope
	m.FromScope(scope)
	return m
}

// FromScope configures the model with optional hyperparameters from the model.
// This allows fine-tuning an existing model configuration.
func (m *Model) FromScope(scope *model.Scope) *Model {
	// Optional parameters with defaults
	m.FFNDim = model.GetParamOr(scope, ParamFFNDim, m.FFNDim)
	m.MaxPosEmbed = model.GetParamOr(scope, ParamMaxPosEmbed, m.MaxPosEmbed)
	m.Dropout = model.GetParamOr(scope, ParamDropout, m.Dropout)

	// Deprecated way to specify LayerNorm:
	useLayerNorm := model.GetParamOr(scope, ParamUseLayerNorm, true)
	if useLayerNorm {
		m.WithLayerNorm(true)
	}
	m.Normalization = model.GetParamOr(scope, ParamNormalization, m.Normalization)
	m.WithNormalization(m.Normalization)
	m.UseBias = model.GetParamOr(scope, ParamUseBias, m.UseBias)
	m.UseCausalMask = model.GetParamOr(scope, ParamUseCausalMask, m.UseCausalMask)
	archStr := model.GetParamOr(scope, ParamArchitecture, m.Architecture.String())
	if arch, err := ArchitectureString(archStr); err == nil {
		m.Architecture = arch
	} else {
		exceptions.Panicf("invalid architecture name %q: options are %v", archStr, ArchitectureValues())
	}
	m.NormEpsilon = model.GetParamOr(scope, ParamNormEpsilon, m.NormEpsilon)
	m.Activation = activation.FromName(model.GetParamOr(scope, ParamActivation, m.Activation.String()))
	m.NumKVHeads = model.GetParamOr(scope, ParamNumKVHeads, m.NumKVHeads)
	m.GlobalHeadDim = model.GetParamOr(scope, ParamGlobalHeadDim, m.GlobalHeadDim)
	m.NumKVSharedLayers = model.GetParamOr(scope, ParamNumKVSharedLayers, m.NumKVSharedLayers)
	m.AttentionScoreSoftCap = model.GetParamOr(scope, ParamAttentionScoreSoftCap, m.AttentionScoreSoftCap)
	m.FinalLogitSoftCap = model.GetParamOr(scope, ParamFinalLogitSoftCap, m.FinalLogitSoftCap)

	// Legacy RoPE configuration from scope
	useRoPE := model.GetParamOr(scope, ParamUseRoPE, false)
	if useRoPE {
		baseFreq := model.GetParamOr(scope, ParamRoPEBaseFreq, 10000.0)
		m.posEncoder = pos.NewRoPE(baseFreq)
	}

	// Handle dtype separately since it's a string
	dtypeStr := model.GetParamOr(scope, ParamDType, "")
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

// WithPositionalEncoder sets the default positional encoder.
func (m *Model) WithPositionalEncoder(encoder pos.Encoder) *Model {
	m.posEncoder = encoder
	return m
}

// WithLayerPositionalEncoder sets the positional encoder for a specific layer.
func (m *Model) WithLayerPositionalEncoder(layerNum int, encoder pos.Encoder) *Model {
	if m.layerPosEncoders == nil {
		m.layerPosEncoders = make(map[int]pos.Encoder)
	}
	m.layerPosEncoders[layerNum] = encoder
	return m
}

// WithLayerTypes sets the layer types (GlobalLayer, LocalLayer) for each layer.
func (m *Model) WithLayerTypes(layerTypes []LayerType) *Model {
	m.LayerTypes = layerTypes
	return m
}

// WithSlidingWindow sets the sliding window size for local attention layers.
// This is applied for all layers of type LocalLayer.
func (m *Model) WithSlidingWindow(window int) *Model {
	m.SlidingWindow = window
	return m
}

// WithGlobalHeadDim sets the global head dimension for full attention layers.
func (m *Model) WithGlobalHeadDim(dim int) *Model {
	m.GlobalHeadDim = dim
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
	if arch == ArchitectureGemma || arch == ArchitectureGemma3 || arch == ArchitectureGemma4 {
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
func (m *Model) WithActivation(activation activation.Type) *Model {
	m.Activation = activation
	return m
}

// WithNumKVHeads enables Grouped Query Attention (GQA) if different from NumHeads.
func (m *Model) WithNumKVHeads(num int) *Model {
	m.NumKVHeads = num
	return m
}

// WithQueryKeyScale sets the query-key scaling factor. If 0, uses the default 1/sqrt(headDim).
func (m *Model) WithQueryKeyScale(scale float64) *Model {
	m.QueryKeyScale = scale
	return m
}

// WithVocabSizePerLayerInput sets the vocab size for per-layer inputs.
//
// See detailed explanation of per-layer embedding (PLE) in [Model.WithHiddenSizePerLayerInput].
func (m *Model) WithVocabSizePerLayerInput(v int) *Model {
	m.VocabSizePerLayerInput = v
	return m
}

// WithHiddenSizePerLayerInput sets the hidden size for per-layer inputs.
// Setting this to a value > 0 triggers the usage of Per-Layer Embedding (PLE), commonly used in Gemma 4.
//
// When active, PLE provides a dynamic, layer-specific token representation which is combined with a gate
// and added to the residual stream at each layer. It operates as follows:
//
//  1. A layer-specific embedding token representation is looked up from `token_embed_per_layer`.
//  2. The initial model representation (token embeddings + positional encodings) is projected, scaled by
//     PerLayerModelProjectionScale, and normalized via RMSNorm.
//  3. These two representations are added, and if PerLayerInputScale != 1, scaled by PerLayerInputScale.
//  4. In each layer, the current layer's state is projected to compute a gate (`pleGate`), which is multiplied
//     by the layer's specific input segment. The result is projected to the main model embedding dimension,
//     normalized, and added to the residual stream.
func (m *Model) WithHiddenSizePerLayerInput(v int) *Model {
	m.HiddenSizePerLayerInput = v
	return m
}

// WithPerLayerInputScale sets the scale factor for combining context aware per-layer-embedding and token identity.
//
// See detailed explanation of per-layer embedding (PLE) in [Model.WithHiddenSizePerLayerInput].
func (m *Model) WithPerLayerInputScale(v float64) *Model {
	m.PerLayerInputScale = v
	return m
}

// WithPerLayerModelProjectionScale sets the scale factor for per-layer-embedding projection.
//
// See detailed explanation of per-layer embedding (PLE) in [Model.WithHiddenSizePerLayerInput].
func (m *Model) WithPerLayerModelProjectionScale(v float64) *Model {
	m.PerLayerModelProjectionScale = v
	return m
}

// WithTransposedWeights configures the model to assume linear weights are transposed (as [out_features, in_features]),
// which is the format used by PyTorch nn.Linear. This enables loading PyTorch models directly.
func (m *Model) WithTransposedWeights(transposed bool) *Model {
	m.TransposedProjections = transposed
	return m
}

// WithNumKVSharedLayers sets the number of layers (from the top/last layers) that share KV projection states.
func (m *Model) WithNumKVSharedLayers(num int) *Model {
	m.NumKVSharedLayers = num
	return m
}

// WithScoreSoftCap sets the soft cap for the attention score computation (see [nn.SoftCap]).
func (m *Model) WithScoreSoftCap(cap float64) *Model {
	m.AttentionScoreSoftCap = cap
	return m
}

// WithFinalLogitSoftCap sets the logit softcap for the final output logits.
func (m *Model) WithFinalLogitSoftCap(cap float64) *Model {
	m.FinalLogitSoftCap = cap
	return m
}

// WithRMSNormOffset sets the offset added to the RMSNorm scale weight.
// The default is 1.0 (used in Gemma 1/2/3). Set to 0.0 for Gemma 4.
func (m *Model) WithRMSNormOffset(offset float64) *Model {
	m.RMSNormOffset = offset
	return m
}

func (m *Model) getLayerType(layerNum int) LayerType {
	if layerNum < len(m.LayerTypes) {
		return m.LayerTypes[layerNum]
	}
	return GlobalLayer
}

func (m *Model) sourceLayerForShared(layerNum int) int {
	firstSharedLayer := m.NumLayers - m.NumKVSharedLayers
	targetType := m.getLayerType(layerNum)
	for i := firstSharedLayer - 1; i >= 0; i-- {
		if m.getLayerType(i) == targetType {
			return i
		}
	}
	panic(fmt.Sprintf("No matching non-shared layer of type %v found for shared layer %d", targetType, layerNum))
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
// Logits builds the model including the last logits projection to predict the next token,
// for each element of the sequence.
//
// This form can be used to embed full sentences or for training.
//
//   - tokens: shaped [batchSize, seqLen], or simply [seqLen]
//   - mask: optional, shaped [batchSize, seqLen] of some integer dtype or [dtypes.Bool].
//     If provided the shape must match tokens.Shape(), and it indicates
//     which tokens are valid (1/true) and which are padding (0/false). The attention mask (causal or not)
//     is computed taking into consideration the mask.
//
// It returns the logits of the last layer, typically shaped [batchSize, seqLen, vocabSize]
func (m *Model) Logits(scope *model.Scope, tokens, mask *Node) *Node {
	embeddings, _, _ := m.AllLayers(scope, tokens, nil, mask, nil)
	return m.LogitsFromEmbeddings(scope, embeddings)
}

// MakeNaiveModelFn returns a "naive" model function for iterative (increasing sequence length, no KVCache)
// generation, using the decode package.
//
// It returns a generate.NaiveModel interface.
func MakeNaiveModelFn(m *Model) generate.NaiveModelFn {
	return func(scope *model.Scope, tokens *Node, length int) *Node {
		paddedSeqLen := tokens.Shape().Dimensions[1]
		if length > paddedSeqLen {
			exceptions.Panicf("indicated length %d of sequence is larger than the tokens ([batch, seqLen]=%s) length provided", length, tokens.Shape())
		}
		var attentionMask *Node
		if paddedSeqLen != length {
			g := tokens.Graph()
			attentionMask = Iota(g, tokens.Shape(), 1)
			attentionMask = LessThan(attentionMask, ConstAs(attentionMask, length))
		}
		return m.Logits(scope, tokens, attentionMask)
	}
}

// Forward returns the forward path for the model.
//
// - tokens: shaped [batchSize, seqLen]
// - positionIds: shaped [batchSize, seqLen], or nil (which defaults to sequential positions)
// - attentionMask: shaped [batchSize, seqLen], or nil
// - cache: KVCacheNodes, or nil if no KV cache is used.
//
// It returns the logits of the last layer, typically shaped [batchSize, seqLen, vocabSize],
// and the updated KV cache.
func (m *Model) Forward(scope *model.Scope,
	tokens, positionIds *Node,
	attentionMask *Node,
	cache KVCacheNodes,
) (logits *Node, updatedCache KVCacheNodes) {
	embeddings, _, updatedCache := m.AllLayers(scope, tokens, positionIds, attentionMask, cache)
	return m.LogitsFromEmbeddings(scope, embeddings), updatedCache
}

// LogitsWithKVCache returns the forward path for the newTokens sequence, using the KV cache.
func (m *Model) LogitsWithKVCache(scope *model.Scope, newTokens *Node, positionIds *Node, cache KVCacheNodes) (*Node, KVCacheNodes) {
	return m.Forward(scope, newTokens, positionIds, nil, cache)
}

// MakeKVCacheModelFn returns a model function used by the decoder for incremental generation with KVCache,
// using the decode package.
func MakeKVCacheModelFn(m *Model) generate.KVCacheModelFn {
	return func(scope *model.Scope, newTokens *Node, position *Node, cache KVCacheNodes) (*Node, KVCacheNodes) {
		return m.Forward(scope, newTokens, position, nil, cache)
	}
}

// AllLayers takes the input tokens and creates the forward graph for the transformer model,
// returning the last layer and all the intermediate layers.
//
//   - tokens: shaped [batchSize, seqLen], or simply [seqLen]
//   - positionIds: shaped [batchSize, seqLen], or nil
//   - attentionMask: optional, if provided the shape must match tokens.Shape(), and it indicates
//     which tokens are valid (1) and which are padding (0).
//   - cache: the KVCacheNodes map containing cached key/value nodes.
//
// It returns:
//
//   - lastLayer: the final hidden state of the last layer, shaped [batchSize, seqLen,hiddenSize].
//   - allLayers: the input to the first layer and the output of each layer.
//   - updatedCache: the updated KVCacheNodes map.
func (m *Model) AllLayers(scope *model.Scope, tokens, positionIds *Node, attentionMask *Node, cache KVCacheNodes) (lastLayer *Node, allLayers []*Node, updatedCache KVCacheNodes) {
	if tokens.Rank() == 1 {
		tokens = ExpandAxes(tokens, 0)
	}
	g := tokens.Graph()
	seqLen := tokens.Shape().Dimensions[1]

	if positionIds == nil {
		posIdx := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
		positionIds = ExpandDims(posIdx, 0)
		positionIds = BroadcastToDims(positionIds, tokens.Shape().Dimensions...)
	}

	m.populateOrderedScopes(scope)
	m.populateLayerTypes(scope)
	if cache != nil {
		updatedCache = make(KVCacheNodes)
		for k, v := range cache {
			updatedCache[k] = v
		}
	}

	allLayers = make([]*Node, 0, m.NumLayers+1)
	x := m.EmbedTokens(scope, tokens)

	if m.posEncoder != nil {
		if preEnc, ok := m.posEncoder.(pos.PreEncoder); ok {
			x = preEnc.PreEncode(x, positionIds, 1)
		}
	}

	if m.EmbedNormalization != layers.NormalizationNone {
		x = m.normalize(scope.In("embed_norm"), x, m.EmbedNormalization)
	}
	allLayers = append(allLayers, x)

	var positionNode *Node
	if positionIds != nil {
		if positionIds.Rank() == 2 {
			positionNode = Squeeze(Slice(positionIds, AxisElem(0), AxisElem(0)))
		} else if positionIds.Rank() == 1 {
			positionNode = Squeeze(Slice(positionIds, AxisElem(0)))
		} else if positionIds.Rank() == 0 {
			positionNode = positionIds
		} else {
			panic(fmt.Sprintf("positionIds must be rank 0, 1 or 2, got rank %d", positionIds.Rank()))
		}
	} else {
		positionNode = ScalarZero(g, dtypes.Int32)
	}

	var perLayerInputs *Node
	if m.HiddenSizePerLayerInput > 0 {
		// Per-layer Embedding (PLE): see explanation in [Model.WithHiddenSizePerLayerInput].
		embeddedPLE := layers.Embedding(scope.In("token_embed_per_layer"), tokens, m.DType, m.VocabSizePerLayerInput, m.NumLayers*m.HiddenSizePerLayerInput)
		pleScale := Scalar(embeddedPLE.Graph(), m.DType, math.Sqrt(float64(m.HiddenSizePerLayerInput)))
		embeddedPLE = Mul(embeddedPLE, pleScale)
		batchSize := tokens.Shape().Dimensions[0]
		embeddedPLE = Reshape(embeddedPLE, batchSize, seqLen, m.NumLayers, m.HiddenSizePerLayerInput)

		perLayerProjection := m.dense(scope.In("per_layer_model_projection"), x, false, m.NumLayers*m.HiddenSizePerLayerInput)
		perLayerProjectionScale := Scalar(perLayerProjection.Graph(), m.DType, m.PerLayerModelProjectionScale)
		perLayerProjection = Mul(perLayerProjection, perLayerProjectionScale)
		perLayerProjection = Reshape(perLayerProjection, batchSize, seqLen, m.NumLayers, m.HiddenSizePerLayerInput)
		perLayerProjection = m.normalize(scope.In("per_layer_projection_norm"), perLayerProjection, layers.NormalizationRMSNorm)

		perLayerInputs = Add(perLayerProjection, embeddedPLE)
		if m.PerLayerInputScale != 1 {
			perLayerInputs = MulScalar(perLayerInputs, m.PerLayerInputScale)
		}
	}

	var sharedKVs KVCacheNodes
	if cache == nil && m.NumKVSharedLayers > 0 {
		sharedKVs = make(KVCacheNodes)
	}

	// Apply all layers.
	for layerNum := range m.NumLayers {
		layerScope := scope.In("layer_%d", layerNum)
		var perLayerInput *Node
		if perLayerInputs != nil {
			perLayerInput = Squeeze(Slice(perLayerInputs, AxisRange(), AxisRange(), AxisElem(layerNum), AxisRange()), 2)
		}
		x = m.ForwardLayer(layerScope, layerNum, x, attentionMask, positionNode, updatedCache, perLayerInput, sharedKVs)
		allLayers = append(allLayers, x)
	}
	if m.FinalNormalization != layers.NormalizationNone {
		x = m.normalize(scope.In("final_norm"), x, m.FinalNormalization)
		if len(allLayers) > 0 {
			allLayers[len(allLayers)-1] = x
		}
	}
	return x, allLayers, updatedCache
}

// EmbedTokens returns the token embeddings for the given tokens using a lookup table.
// This is the very first step of the transformer model.
//
// If you want the model embeddings after of the full model, take the last layer of AllLayers.
//
// This step is done automatically by AllLayers or Logits, but if needed, it can
// be used separately by calling this method.
//
// - tokens: shaped [batchSize, seqLen] or [seqLen]
//
// Returns: [batchSize, seqLen, EmbedDim]. Notice if no batchSize was given, a batch axis with size of 1 is created.
// EmbedTokens returns the token embeddings for the given tokens using a lookup table.
// This is the very first step of the transformer model.
//
// It defaults to TokenType 0 for all tokens if TokenTypeEmbedSize > 0.
// Use EmbedTokensWithType to select a different index.
func (m *Model) EmbedTokens(scope *model.Scope, tokens *Node) *Node {
	return m.EmbedTokensWithType(scope, tokens, nil)
}

// EmbedTokensWithType returns the token embeddings for the given tokens and tokenTypes using lookup tables.
//
// It requires WithTokenTypeEmbedding configuration.
//
// The tokenTypes define an extra constant embedding added to all tokens based on the "token type".
// For BERT models, there are two different sentences, which gets a different "token type".
//
// tokenTypes must be a scalar int (limited to the vocabSize set in WithTokenTypeEmbedding).
// Or it can be nil, in which case it defaults to TokenType 0 for all tokens if TokenTypeEmbedSize > 0.
func (m *Model) EmbedTokensWithType(scope *model.Scope, tokens, tokenTypes *Node) *Node {
	g := tokens.Graph()
	// Tokens embedding table lookup.
	embedded := layers.Embedding(scope.In("token_embed"), tokens, m.DType, m.VocabSize, m.EmbedDim)
	if embedded.Rank() == 2 {
		embedded = ExpandDims(embedded, 1)
	}
	if m.ScaleTokenEmbeddings {
		scale := Scalar(embedded.Graph(), m.DType, math.Sqrt(float64(m.EmbedDim)))
		embedded = Mul(embedded, scale)
	}
	if m.TokenTypeEmbedSize > 0 {
		if tokenTypes == nil {
			tokenTypes = ScalarZero(g, dtypes.Int32)
		}
		tokenTypeEmbed := layers.Embedding(scope.In("token_type_embed"), tokenTypes, m.DType, m.TokenTypeEmbedSize, m.EmbedDim)
		embedded = Add(embedded, broadcastPrefixToMatch(tokenTypeEmbed, embedded))
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
func (m *Model) PrePositionalEncoder(scope *model.Scope, x *Node, position int, useKVCache bool) *Node {
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
func (m *Model) LogitsFromEmbeddings(scope *model.Scope, embeddings *Node) *Node {
	logits := layers.Dense(scope.In("output"), embeddings, false, m.VocabSize)
	if m.FinalLogitSoftCap > 0 {
		logits = nn.SoftCap(logits, m.FinalLogitSoftCap)
	}
	return logits
}

// // ForwardLayer executes a single transformer layer block depending on the configured architecture.
//
// - scope: scope must be already scoped for the layer, e.g. scope.In("layer_0")
// - x: features coming from the previous layer (or token embedding table), shape [batchSize, seqLen, embedDim]
// - attentionMask: (optional, can be nil) shaped [batchSize, seqLen]
// - position: position node (scalar int32) indicating the current absolute position
// - cache: updated KVCacheNodes map
// - sharedKVs: optional, map of shared key/values for layers that don't project their own KVs
//
// It returns the output of the layer, shape [batchSize, seqLen, embedDim].
func (m *Model) ForwardLayer(scope *model.Scope, layerNum int, x, attentionMask *Node, position *Node, cache KVCacheNodes, perLayerInput *Node, sharedKVs KVCacheNodes) *Node {
	if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 || m.Architecture == ArchitectureGemma4 {
		return m.forwardLayerGemma(scope, layerNum, x, attentionMask, position, cache, perLayerInput, sharedKVs)
	}
	return m.forwardLayerStandard(scope, layerNum, x, attentionMask, position, cache, perLayerInput, sharedKVs)
}

// seqLensFromMask derives int32 [B] actual-length tensors from a rank-2 `[B, Skv]` boolean
// padding mask where valid positions are 1 and padding positions are 0.
//
// This conversion is only correct when the mask is contiguous and right-padded (valid tokens
// occupy exactly [0, L) for each batch element). The returned querySeqLen and keyValueSeqLen
// are identical because in non-KVCache self-attention query and key share the same padding.
func seqLensFromMask(mask *Node) (querySeqLen, keyValueSeqLen *Node) {
	int32Mask := ConvertDType(mask, dtypes.Int32)
	seqLen := ReduceSum(int32Mask, 1)
	return seqLen, seqLen
}

func (m *Model) forwardLayerStandard(layerScope *model.Scope, layerNum int, x, attentionMask *Node, position *Node, cache KVCacheNodes, perLayerInput *Node, sharedKVs KVCacheNodes) *Node {
	residual := x
	var attn *Node

	// Projections
	kvHeads := m.NumKVHeads
	if kvHeads == 0 {
		kvHeads = m.NumHeads
	}

	layerType := GlobalLayer
	if layerNum < len(m.LayerTypes) {
		layerType = m.LayerTypes[layerNum]
	}
	headDim := m.HeadDim
	if layerType == GlobalLayer && m.GlobalHeadDim > 0 {
		headDim = m.GlobalHeadDim
	}

	attnScope := layerScope.In("attn")
	mhaScope := attnScope.At("MultiHeadAttention")
	queryProjected := m.dense(mhaScope.In("query"), x, m.UseBias, m.NumHeads, headDim)

	firstSharedLayer := m.NumLayers - m.NumKVSharedLayers
	isShared := m.NumKVSharedLayers > 0 && layerNum >= firstSharedLayer

	var keyProjected, valueProjected *Node
	var sourceLayerNum int
	var sourceLayerScopePath string
	if isShared {
		sourceLayerNum = m.sourceLayerForShared(layerNum)
		sourceLayerScopePath = m.KVCache.OrderedScopes[sourceLayerNum]
	}

	if !isShared {
		keyProjected = m.dense(mhaScope.In("key"), x, m.UseBias, kvHeads, headDim)
		valueProjected = m.dense(mhaScope.In("value"), x, m.UseBias, kvHeads, headDim)
	} else if cache == nil {
		var ok bool
		keyProjected, ok = sharedKVs[fmt.Sprintf("%d_k", sourceLayerNum)]
		if !ok {
			panic(fmt.Sprintf("Shared key projection not found for source layer %d in layer %d", sourceLayerNum, layerNum))
		}
		valueProjected, ok = sharedKVs[fmt.Sprintf("%d_v", sourceLayerNum)]
		if !ok {
			panic(fmt.Sprintf("Shared value projection not found for source layer %d in layer %d", sourceLayerNum, layerNum))
		}
	}

	// Apply RoPE positional encoding if configured
	posEncoder := m.posEncoder
	if enc := m.layerPosEncoders[layerNum]; enc != nil {
		posEncoder = enc
	}
	if posEncoder != nil {
		if qkEncoder, ok := posEncoder.(pos.QKEncoder); ok {
			g := x.Graph()
			seqLen := x.Shape().Dimensions[1]
			posIndices := pos.SequentialPositions(g, position, seqLen)
			if isShared {
				queryProjected, _ = qkEncoder.EncodeQK(queryProjected, queryProjected, posIndices, 1)
			} else {
				queryProjected, keyProjected = qkEncoder.EncodeQK(queryProjected, keyProjected, posIndices, 1)
			}
		}
	}

	if cache == nil && !isShared && sharedKVs != nil {
		sharedKVs[fmt.Sprintf("%d_k", layerNum)] = keyProjected
		sharedKVs[fmt.Sprintf("%d_v", layerNum)] = valueProjected
	}

	if cache != nil {
		var fullKey, fullValue *Node
		if isShared {
			fullKey, fullValue = m.KVCache.Get(cache, sourceLayerScopePath)
		} else {
			m.KVCache.Update(attnScope, cache, keyProjected, valueProjected, position)
			fullKey, fullValue = m.KVCache.Get(cache, attnScope.Scope())
		}
		batchSize := x.Shape().Dimensions[0]
		seqLen := x.Shape().Dimensions[1]
		cacheSeqLen := fullKey.Shape().Dimensions[2]

		fullKey = TransposeAllDims(fullKey, 0, 2, 1, 3)
		fullValue = TransposeAllDims(fullValue, 0, 2, 1, 3)

		qReshaped := Reshape(queryProjected, batchSize, seqLen, m.NumHeads*headDim)
		kReshaped := Reshape(fullKey, batchSize, cacheSeqLen, kvHeads*headDim)
		vReshaped := Reshape(fullValue, batchSize, cacheSeqLen, kvHeads*headDim)

		slidingWindow := 0
		if layerType == LocalLayer {
			slidingWindow = m.SlidingWindow
		}
		customMask := m.KVCache.BuildAttentionMask(attnScope, cache, x, position, m.UseCausalMask, slidingWindow)

		if attentionMask != nil {
			expandedMask := ExpandDims(attentionMask, 1)
			expandedMask = BroadcastToDims(expandedMask, customMask.Shape().Dimensions...)
			customMask = LogicalAnd(customMask, expandedMask)
		}

		attnBuilder := attention.MultiHeadAttention(attnScope, qReshaped, kReshaped, vReshaped, m.NumHeads, headDim).
			UseTransposedWeights(m.TransposedProjections).
			WithNumKVHeads(kvHeads).
			WithPreProjected(true).
			WithOutputDim(m.EmbedDim).
			WithQueryKeyMatrixMask(customMask).
			WithScoreSoftCap(m.AttentionScoreSoftCap).
			WithQueryKeyScale(m.QueryKeyScale)

		if !m.UseBias {
			attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	} else {
		batchSize := x.Shape().Dimensions[0]
		seqLen := x.Shape().Dimensions[1]
		qReshaped := Reshape(queryProjected, batchSize, seqLen, m.NumHeads*headDim)
		kReshaped := Reshape(keyProjected, batchSize, seqLen, kvHeads*headDim)
		vReshaped := Reshape(valueProjected, batchSize, seqLen, kvHeads*headDim)

		attnBuilder := attention.MultiHeadAttention(attnScope, qReshaped, kReshaped, vReshaped, m.NumHeads, headDim).
			UseTransposedWeights(m.TransposedProjections).
			WithNumKVHeads(kvHeads).
			WithPreProjected(true).
			WithOutputDim(m.EmbedDim).
			WithScoreSoftCap(m.AttentionScoreSoftCap).
			WithQueryKeyScale(m.QueryKeyScale)
		if attentionMask != nil {
			// Rank-2 [B, Skv] padding masks are expressed more efficiently as sequence lengths
			// (cuDNN PADDING masktype). WithSeqLens is mutually exclusive with WithCausalMask
			// or SlidingWindow, so fall back to WithMask for those paths.
			if attentionMask.Rank() == 2 && !m.UseCausalMask && !(layerType == LocalLayer && m.SlidingWindow > 0) {
				qLen, kvLen := seqLensFromMask(attentionMask)
				attnBuilder.WithSeqLens(qLen, kvLen)
			} else {
				attnBuilder.WithMask(attentionMask)
			}
		}
		if m.UseCausalMask {
			attnBuilder.WithCausalMask(true)
		}

		if layerType == LocalLayer && m.SlidingWindow > 0 {
			attnBuilder.WithSlidingWindow(m.SlidingWindow)
		}
		if !m.UseBias {
			attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	}

	x = Add(residual, attn)
	x = m.normalize(layerScope.In("norm1"), x, m.Normalization)

	residual = x
	ff := m.dense(layerScope.In("ff1"), x, m.UseBias, m.FFNDim*m.Activation.HiddenDimMultiplier())
	ff = activation.Apply(m.Activation, ff)
	if m.Dropout > 0 {
		ff = layers.Dropout(layerScope.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), m.Dropout))
	}
	ff = m.dense(layerScope.In("ff2"), ff, m.UseBias, m.EmbedDim)
	x = Add(residual, ff)
	x = m.normalize(layerScope.In("norm2"), x, m.Normalization)
	return x
}

// normalize according to configuration.
// dense behaves like layers.dense but checks whether the model is configured to use transposed projections.
// If it is, it computes the dense layer using DotGeneral (Einsum), expecting weights in the format [outDim, inDim].
// PyTorch nn.Linear stores its matrix in this transposed format.
func (m *Model) dense(scope *model.Scope, op *Node, useBias bool, outputDims ...int) *Node {
	if !m.TransposedProjections {
		return layers.Dense(scope, op, useBias, outputDims...)
	}
	scope = scope.In("dense")
	g := op.Graph()
	inDim := op.Shape().Dim(-1)
	outDim := 1
	for _, d := range outputDims {
		outDim *= d
	}
	wVar := scope.VariableWithShape("weights", shapes.Make(op.DType(), outDim, inDim))
	w := wVar.NodeValue(g)
	y := DotGeneral(op, []int{-1}, nil, w, []int{1}, nil)

	if useBias {
		bVar := scope.VariableWithShape("biases", shapes.Make(op.DType(), outDim))
		y = Add(y, broadcastPrefixToMatch(bVar.NodeValue(g), y))
	}

	if len(outputDims) > 1 {
		newDims := make([]int, op.Rank()-1+len(outputDims))
		copy(newDims, op.Shape().Dimensions[:op.Rank()-1])
		copy(newDims[op.Rank()-1:], outputDims)
		y = Reshape(y, newDims...)
	}
	return y
}

func (m *Model) normalize(scope *model.Scope, operand *Node, normType string) *Node {
	if normType == layers.NormalizationNone {
		return operand
	}
	switch normType {
	case layers.NormalizationRMSNorm:
		builder := norm.RMSNorm(scope, operand).WithEpsilon(m.NormEpsilon)
		if m.Architecture == ArchitectureGemma || m.Architecture == ArchitectureGemma3 || m.Architecture == ArchitectureGemma4 {
			if m.RMSNormOffset != 0 {
				builder = builder.WithScaleOffset(m.RMSNormOffset)
			}
		}
		return builder.Done()
	case layers.NormalizationLayerNorm:
		return norm.LayerNorm(scope, operand, -1).Epsilon(m.NormEpsilon).Done()
	default:
		exceptions.Panicf("unsupported normalization type: %q", normType)
		return nil
	}
}

func (m *Model) forwardLayerGemma(layerScope *model.Scope, layerNum int, x, attentionMask *Node, position *Node, cache KVCacheNodes, perLayerInput *Node, sharedKVs KVCacheNodes) *Node {
	residual := x

	// Pre-attention normalization.
	x = m.normalize(layerScope.In("input_norm"), x, m.Normalization)

	// Projections
	kvHeads := m.NumKVHeads
	if kvHeads == 0 {
		kvHeads = m.NumHeads
	}

	layerType := GlobalLayer
	if layerNum < len(m.LayerTypes) {
		layerType = m.LayerTypes[layerNum]
	}
	headDim := m.HeadDim
	if layerType == GlobalLayer && m.GlobalHeadDim > 0 {
		headDim = m.GlobalHeadDim
	}

	selfAttnScope := layerScope.In("self_attn")
	mhaScope := selfAttnScope.At("MultiHeadAttention")
	queryProjected := m.dense(mhaScope.In("query"), x, m.UseBias, m.NumHeads, headDim)

	firstSharedLayer := m.NumLayers - m.NumKVSharedLayers
	isShared := m.NumKVSharedLayers > 0 && layerNum >= firstSharedLayer

	var keyProjected, valueProjected *Node
	var sourceLayerNum int
	var sourceLayerScopePath string
	if isShared {
		sourceLayerNum = m.sourceLayerForShared(layerNum)
		sourceLayerScopePath = m.KVCache.OrderedScopes[sourceLayerNum]
	}

	if !isShared {
		keyProjected = m.dense(mhaScope.In("key"), x, m.UseBias, kvHeads, headDim)
		valueProjected = m.dense(mhaScope.In("value"), x, m.UseBias, kvHeads, headDim)
	} else if cache == nil {
		var ok bool
		keyProjected, ok = sharedKVs[fmt.Sprintf("%d_k", sourceLayerNum)]
		if !ok {
			panic(fmt.Sprintf("Shared key projection not found for source layer %d in layer %d", sourceLayerNum, layerNum))
		}
		valueProjected, ok = sharedKVs[fmt.Sprintf("%d_v", sourceLayerNum)]
		if !ok {
			panic(fmt.Sprintf("Shared value projection not found for source layer %d in layer %d", sourceLayerNum, layerNum))
		}
	}

	// Apply QK RMSNorm if Gemma 3 or Gemma 4
	if m.Architecture == ArchitectureGemma3 || m.Architecture == ArchitectureGemma4 {
		queryBuilder := norm.RMSNorm(mhaScope.Shared("query"), queryProjected).WithEpsilon(m.NormEpsilon).WithNormalizationAxes(-1)
		if m.RMSNormOffset != 0 {
			queryBuilder = queryBuilder.WithScaleOffset(m.RMSNormOffset)
		}
		queryProjected = queryBuilder.Done()
		if !isShared {
			keyBuilder := norm.RMSNorm(mhaScope.Shared("key"), keyProjected).WithEpsilon(m.NormEpsilon).WithNormalizationAxes(-1)
			if m.RMSNormOffset != 0 {
				keyBuilder = keyBuilder.WithScaleOffset(m.RMSNormOffset)
			}
			keyProjected = keyBuilder.Done()

			if m.Architecture == ArchitectureGemma4 {
				valueProjected = norm.RMSNorm(mhaScope.Shared("value"), valueProjected).WithEpsilon(m.NormEpsilon).WithNormalizationAxes(-1).WithScale(false).Done()
			}
		}
	}

	// Apply RoPE positional encoding if configured
	posEncoder := m.posEncoder
	if enc := m.layerPosEncoders[layerNum]; enc != nil {
		posEncoder = enc
	}
	if posEncoder != nil {
		if qkEncoder, ok := posEncoder.(pos.QKEncoder); ok {
			g := x.Graph()
			seqLen := x.Shape().Dimensions[1]
			posIndices := pos.SequentialPositions(g, position, seqLen)
			if isShared {
				queryProjected, _ = qkEncoder.EncodeQK(queryProjected, queryProjected, posIndices, 1)
			} else {
				queryProjected, keyProjected = qkEncoder.EncodeQK(queryProjected, keyProjected, posIndices, 1)
			}
		}
	}

	if cache == nil && !isShared && sharedKVs != nil {
		sharedKVs[fmt.Sprintf("%d_k", layerNum)] = keyProjected
		sharedKVs[fmt.Sprintf("%d_v", layerNum)] = valueProjected
	}

	var attn *Node
	if cache != nil {
		var fullKey, fullValue *Node
		if isShared {
			fullKey, fullValue = m.KVCache.Get(cache, sourceLayerScopePath)
		} else {
			m.KVCache.Update(selfAttnScope, cache, keyProjected, valueProjected, position)
			fullKey, fullValue = m.KVCache.Get(cache, selfAttnScope.Scope())
		}

		fullKey = TransposeAllDims(fullKey, 0, 2, 1, 3)
		fullValue = TransposeAllDims(fullValue, 0, 2, 1, 3)

		batchSize := x.Shape().Dimensions[0]
		seqLen := x.Shape().Dimensions[1]
		cacheSeqLen := fullKey.Shape().Dimensions[1]

		qReshaped := Reshape(queryProjected, batchSize, seqLen, m.NumHeads*headDim)
		kReshaped := Reshape(fullKey, batchSize, cacheSeqLen, kvHeads*headDim)
		vReshaped := Reshape(fullValue, batchSize, cacheSeqLen, kvHeads*headDim)

		slidingWindow := 0
		if layerType == LocalLayer {
			slidingWindow = m.SlidingWindow
		}
		customMask := m.KVCache.BuildAttentionMask(selfAttnScope, cache, x, position, m.UseCausalMask, slidingWindow)

		if attentionMask != nil {
			expandedMask := ExpandDims(attentionMask, 1)
			expandedMask = BroadcastToDims(expandedMask, customMask.Shape().Dimensions...)
			customMask = LogicalAnd(expandedMask, customMask)
		}

		attnBuilder := attention.MultiHeadAttention(selfAttnScope, qReshaped, kReshaped, vReshaped, m.NumHeads, headDim).
			UseTransposedWeights(m.TransposedProjections).
			WithNumKVHeads(kvHeads).
			WithPreProjected(true).
			WithOutputDim(m.EmbedDim).
			WithQueryKeyMatrixMask(customMask).
			WithScoreSoftCap(m.AttentionScoreSoftCap).
			WithQueryKeyScale(m.QueryKeyScale)

		if !m.UseBias {
			attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	} else {
		batchSize := x.Shape().Dimensions[0]
		seqLen := x.Shape().Dimensions[1]
		qReshaped := Reshape(queryProjected, batchSize, seqLen, m.NumHeads*headDim)
		kReshaped := Reshape(keyProjected, batchSize, seqLen, kvHeads*headDim)
		vReshaped := Reshape(valueProjected, batchSize, seqLen, kvHeads*headDim)

		attnBuilder := attention.MultiHeadAttention(selfAttnScope, qReshaped, kReshaped, vReshaped, m.NumHeads, headDim).
			UseTransposedWeights(m.TransposedProjections).
			WithNumKVHeads(kvHeads).
			WithPreProjected(true).
			WithOutputDim(m.EmbedDim).
			WithScoreSoftCap(m.AttentionScoreSoftCap).
			WithQueryKeyScale(m.QueryKeyScale)
		if attentionMask != nil {
			// Rank-2 [B, Skv] padding masks are expressed more efficiently as sequence lengths
			// (cuDNN PADDING masktype). WithSeqLens is mutually exclusive with WithCausalMask
			// or SlidingWindow, so fall back to WithMask for those paths.
			if attentionMask.Rank() == 2 && !m.UseCausalMask && !(layerType == LocalLayer && m.SlidingWindow > 0) {
				qLen, kvLen := seqLensFromMask(attentionMask)
				attnBuilder.WithSeqLens(qLen, kvLen)
			} else {
				attnBuilder.WithMask(attentionMask)
			}
		}
		if m.UseCausalMask {
			attnBuilder.WithCausalMask(true)
		}

		if layerType == LocalLayer && m.SlidingWindow > 0 {
			attnBuilder.WithSlidingWindow(m.SlidingWindow)
		}
		if !m.UseBias {
			attnBuilder.UseProjectionBias(false)
		}
		if m.Dropout > 0 {
			dropoutRate := Scalar(x.Graph(), x.DType(), m.Dropout)
			attnBuilder.WithDropout(dropoutRate)
		}
		attn = attnBuilder.Done()
	}

	// Post-attention normalization.
	attn = m.normalize(layerScope.In("post_attention_norm"), attn, m.Normalization)
	x = Add(residual, attn)
	residual = x

	// Pre-feedforward normalization
	x = m.normalize(layerScope.In("pre_feedforward_norm"), x, m.Normalization)

	// Gemma uses SwiGLU: gate_proj, up_proj, down_proj
	// Notice: it is not using activation.SwiGLU and instead doing the projections separately.
	ffScope := layerScope.In("mlp")
	gate := m.dense(ffScope.In("gate_proj"), x, m.UseBias, m.FFNDim)
	up := m.dense(ffScope.In("up_proj"), x, m.UseBias, m.FFNDim)
	switchedNode := activation.Apply(m.Activation, gate)
	ff := Mul(switchedNode, up)

	if m.Dropout > 0 {
		ff = layers.Dropout(ffScope.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), m.Dropout))
	}
	ff = m.dense(ffScope.In("down_proj"), ff, m.UseBias, m.EmbedDim)

	// Post-feedforward normalization
	ff = m.normalize(layerScope.In("post_feedforward_norm"), ff, m.Normalization)
	x = Add(residual, ff)

	if perLayerInput != nil {
		residualLayer := x
		pleGate := m.dense(layerScope.In("per_layer_input_gate"), x, false, m.HiddenSizePerLayerInput)
		pleGate = activation.Apply(m.Activation, pleGate) // act_fn (GELU)
		pleGated := Mul(pleGate, perLayerInput)
		pleProj := m.dense(layerScope.In("per_layer_projection"), pleGated, false, m.EmbedDim)
		pleBuilder := norm.RMSNorm(layerScope.In("post_per_layer_input_norm"), pleProj).WithEpsilon(m.NormEpsilon)
		if m.RMSNormOffset != 0 {
			pleBuilder = pleBuilder.WithScaleOffset(m.RMSNormOffset)
		}
		pleProj = pleBuilder.Done()
		x = Add(residualLayer, pleProj)
	}

	if m.HiddenSizePerLayerInput > 0 {
		layerScalarVar := layerScope.VariableWithShape("layer_scalar", shapes.Make(m.DType, 1))
		x = Mul(x, Squeeze(layerScalarVar.NodeValue(x.Graph())))
	}

	return x
}

// WithEmbedNormalization sets the normalization type ("layer", "rms", "none" or "")
// to be applied only once after the token embeddings and before the first transformer layer.
func (m *Model) WithEmbedNormalization(norm string) *Model {
	if norm == "" {
		norm = layers.NormalizationNone
	}
	m.EmbedNormalization = norm
	return m
}

func broadcastPrefixToMatch(x, target *Node) *Node {
	for x.Rank() < target.Rank() {
		x = ExpandAxes(x, 0)
	}
	return BroadcastToShape(x, target.Shape())
}

// WithTokenTypeEmbedding an extra constant embedding based on the "token type".
//
// In BERT models it is the sentence id, when classifying two sentences, and the vocabSize defaults to 2.
//
// The EmbedTokens will default to use the TokenType 0 (the default), but you can use EmbedTokensWithType to
// select a different index.
func (m *Model) WithTokenTypeEmbedding(vocabSize int) *Model {
	m.TokenTypeEmbedSize = vocabSize
	return m
}
