// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/linear"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
)

// forwardLayerQwen35 executes a single Qwen3.5 hybrid decoder layer.
// Each layer uses pre-norm residual connections with zero-centered RMSNorm (1.0 + weight):
//   - input_norm -> token mixer (either Gated DeltaNet linear_attn or gated GQA self_attn) -> residual add
//   - post_attention_norm -> SwiGLU MLP -> residual add
func (m *Model) forwardLayerQwen35(
	layerScope *model.Scope,
	layerNum int,
	x *Node,
	position *Node,
	cache KVCacheNodes,
	options CallOptions,
) *Node {
	residual := x
	x = m.normalize(layerScope.In("input_norm"), x, m.Normalization)

	layerType := m.getLayerType(layerNum)
	var mixed *Node
	if layerType == LinearLayer {
		mixed = m.forwardGatedDeltaNet(layerScope.In("linear_attn"), x, cache, options)
	} else {
		mixed = m.forwardQwen35SelfAttention(layerScope.In("self_attn"), layerNum, x, position, cache, layerType, options)
	}

	x = Add(residual, mixed)
	residual = x

	// Pre-MLP normalization (named post_attention_layernorm in HuggingFace Qwen3.5).
	x = m.normalize(layerScope.In("post_attention_norm"), x, m.Normalization)

	// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
	ffScope := layerScope.In("mlp")
	gate := m.dense(ffScope.In("gate_proj"), x, m.UseBias, m.FFNDim)
	up := m.dense(ffScope.In("up_proj"), x, m.UseBias, m.FFNDim)
	ff := Mul(activation.Apply(m.Activation, gate), up)
	if m.Dropout > 0 {
		ff = layers.Dropout(ffScope.In("ff_dropout"), ff, Scalar(ff.Graph(), ff.DType(), m.Dropout))
	}
	ff = m.dense(ffScope.In("down_proj"), ff, m.UseBias, m.EmbedDim)

	return Add(residual, ff)
}

func (m *Model) forwardQwen35SelfAttention(
	selfAttnScope *model.Scope,
	layerNum int,
	x *Node,
	position *Node,
	cache KVCacheNodes,
	layerType LayerType,
	options CallOptions,
) *Node {
	kvHeads := m.NumKVHeads
	if kvHeads == 0 {
		kvHeads = m.NumHeads
	}
	headDim := m.HeadDim
	if layerType == GlobalLayer && m.GlobalHeadDim > 0 {
		headDim = m.GlobalHeadDim
	}

	batchSpec := DimensionSpecFor(x, 0)
	seqSpec := DimensionSpecFor(x, 1)
	mhaScope := selfAttnScope.At("MultiHeadAttention")

	var queryProjected, outputGate *Node
	if m.AttnOutputGate {
		// Qwen3.5 q_proj outputs [B, S, NumHeads, 2*HeadDim]; split per-head into (query, gate).
		qAndGate := m.dense(mhaScope.In("query"), x, m.UseBias, m.NumHeads, headDim*2)
		queryProjected = Slice(qAndGate, AxisRange().Spacer(), AxisRange(0, headDim))
		gate := Slice(qAndGate, AxisRange().Spacer(), AxisRange(headDim, headDim*2))
		outputGate = DynamicReshape(gate, batchSpec, seqSpec, StaticDim(m.NumHeads*headDim))
	} else {
		queryProjected = m.dense(mhaScope.In("query"), x, m.UseBias, m.NumHeads, headDim)
	}

	keyProjected := m.dense(mhaScope.In("key"), x, m.UseBias, kvHeads, headDim)
	valueProjected := m.dense(mhaScope.In("value"), x, m.UseBias, kvHeads, headDim)

	// QK zero-centered RMSNorm along head_dim.
	qNormBuilder := norm.RMSNorm(mhaScope.Shared("query"), queryProjected).
		WithEpsilon(m.NormEpsilon).
		WithNormalizationAxes(-1).
		WithDType(m.NormDType)
	kNormBuilder := norm.RMSNorm(mhaScope.Shared("key"), keyProjected).
		WithEpsilon(m.NormEpsilon).
		WithNormalizationAxes(-1).
		WithDType(m.NormDType)
	if m.RMSNormOffset != 0 {
		qNormBuilder = qNormBuilder.WithScaleOffset(m.RMSNormOffset)
		kNormBuilder = kNormBuilder.WithScaleOffset(m.RMSNormOffset)
	}
	queryProjected = qNormBuilder.Done()
	keyProjected = kNormBuilder.Done()

	// Partial/full RoPE positional encoding.
	posEncoder := m.posEncoder
	if enc := m.layerPosEncoders[layerNum]; enc != nil {
		posEncoder = enc
	}
	if posEncoder != nil {
		if qkEncoder, ok := posEncoder.(pos.QKEncoder); ok {
			g := x.Graph()
			posIndices := DynamicIota(g, dtypes.Int32, 0, seqSpec)
			if position != nil {
				posIndices = Add(posIndices, ConvertDType(position, dtypes.Int32))
			}
			queryProjected, keyProjected = qkEncoder.EncodeQK(queryProjected, keyProjected, posIndices, 1)
		}
	}

	if cache != nil {
		return m.buildCachedAttention(selfAttnScope, x, queryProjected, keyProjected, valueProjected, outputGate, position, cache, false, "", layerType, kvHeads, headDim, options)
	}

	qReshaped := DynamicReshape(queryProjected, batchSpec, seqSpec, StaticDim(m.NumHeads*headDim))
	kReshaped := DynamicReshape(keyProjected, batchSpec, seqSpec, StaticDim(kvHeads*headDim))
	vReshaped := DynamicReshape(valueProjected, batchSpec, seqSpec, StaticDim(kvHeads*headDim))

	attnBuilder := attention.MultiHeadAttention(selfAttnScope, qReshaped, kReshaped, vReshaped, m.NumHeads, headDim).
		UseTransposedWeights(m.TransposedProjections).
		WithNumKVHeads(kvHeads).
		WithPreProjected(true).
		WithOutputDim(m.EmbedDim).
		WithScoreSoftCap(m.AttentionScoreSoftCap).
		WithQueryKeyScale(m.QueryKeyScale)
	if outputGate != nil {
		attnBuilder.WithOutputGate(outputGate)
	}
	if options.AttentionMask != nil {
		attnBuilder.WithMask(options.AttentionMask)
	}
	if options.SeqLen != nil {
		attnBuilder.WithSeqLens(options.SeqLen, options.SeqLen)
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
		attnBuilder.WithDropout(Scalar(x.Graph(), x.DType(), m.Dropout))
	}
	return attnBuilder.Done()
}

// forwardGatedDeltaNet implements Qwen3.5's Gated DeltaNet linear attention block
// using [linear.GatedDeltaNet].
func (m *Model) forwardGatedDeltaNet(
	attnScope *model.Scope,
	x *Node,
	cache KVCacheNodes,
	options CallOptions,
) *Node {
	kernelSize := m.LinearConvKernelDim
	if kernelSize <= 0 {
		kernelSize = 4
	}
	stateDType := m.LinearStateDType
	if stateDType == dtypes.InvalidDType {
		stateDType = dtypes.Float32
	}

	b := linear.GatedDeltaNet(attnScope, x, m.LinearNumValueHeads, m.LinearValueHeadDim).
		WithNumKeyHeads(m.LinearNumKeyHeads).
		WithKeyHeadDim(m.LinearKeyHeadDim).
		WithOutputDim(m.EmbedDim).
		WithShortConv(kernelSize).
		WithActivation(m.Activation).
		WithNormEpsilon(m.NormEpsilon).
		WithStateDType(stateDType).
		WithBias(false).
		UseTransposedWeights(m.TransposedProjections)

	if options.AttentionMask != nil {
		b.WithAttentionMask(options.AttentionMask)
	}
	if options.SeqLen != nil {
		b.WithSeqLen(options.SeqLen)
	}
	if cache != nil {
		convState, recurrentState := m.KVCache.GetLinearState(cache, attnScope.Scope())
		b.WithState(convState, recurrentState).WithReturnState(true)
	}

	out, newConvState, newRecurrentState := b.DoneWithState()
	if cache != nil {
		m.KVCache.SetLinearState(cache, attnScope.Scope(), newConvState, newRecurrentState)
	}
	return out
}
