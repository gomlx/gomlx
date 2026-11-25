// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package transformer

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/exceptions"
)

const gatedDeltaRuleChunkSize = 64

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

// forwardGatedDeltaNet implements Qwen3.5's Gated DeltaNet linear attention block.
func (m *Model) forwardGatedDeltaNet(
	attnScope *model.Scope,
	x *Node,
	cache KVCacheNodes,
	options CallOptions,
) *Node {
	g := x.Graph()
	numKHeads := m.LinearNumKeyHeads
	numVHeads := m.LinearNumValueHeads
	headKDim := m.LinearKeyHeadDim
	headVDim := m.LinearValueHeadDim
	if numKHeads <= 0 || numVHeads <= 0 || headKDim <= 0 || headVDim <= 0 {
		exceptions.Panicf("GatedDeltaNet requires positive LinearNumKeyHeads (%d), LinearNumValueHeads (%d), LinearKeyHeadDim (%d), LinearValueHeadDim (%d)",
			numKHeads, numVHeads, headKDim, headVDim)
	}
	if numVHeads%numKHeads != 0 {
		exceptions.Panicf("GatedDeltaNet requires LinearNumValueHeads (%d) divisible by LinearNumKeyHeads (%d)", numVHeads, numKHeads)
	}
	kernelSize := m.LinearConvKernelDim
	if kernelSize <= 0 {
		kernelSize = 4
	}
	keyDim := numKHeads * headKDim
	valueDim := numVHeads * headVDim
	convDim := 2*keyDim + valueDim

	// Zero out padding tokens (matches HuggingFace apply_mask_to_padding_states).
	x = applyLinearPaddingMask(x, options)

	batchSpec := DimensionSpecFor(x, 0)
	seqSpec := DimensionSpecFor(x, 1)

	// Input projections.
	mixedQKV := m.dense(attnScope.In("in_proj_qkv"), x, false, convDim)          // [B, S, convDim]
	z := m.dense(attnScope.In("in_proj_z"), x, false, numVHeads, headVDim)       // [B, S, numVHeads, headVDim]
	b := m.dense(attnScope.In("in_proj_b"), x, false, numVHeads)                 // [B, S, numVHeads]
	a := m.dense(attnScope.In("in_proj_a"), x, false, numVHeads)                 // [B, S, numVHeads]

	var convState, recurrentState *Node
	if cache != nil {
		convState, recurrentState = m.KVCache.GetLinearState(cache, attnScope.Scope())
	}

	// Causal depthwise Conv1d + SiLU.
	convWeightVar := attnScope.VariableWithShape("conv1d_weight", shapes.Make(m.DType, convDim, 1, kernelSize))
	convWeightKC := Transpose(Squeeze(convWeightVar.NodeValue(g), 1), 0, 1) // [K, convDim]
	mixedQKV, newConvState := causalDepthwiseConv1D(mixedQKV, convState, convWeightKC, kernelSize, cache != nil)
	mixedQKV = activation.Apply(m.Activation, mixedQKV)

	// Split into query, key, value.
	query := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(0, keyDim)),
		batchSpec, seqSpec, StaticDim(numKHeads), StaticDim(headKDim),
	)
	key := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(keyDim, 2*keyDim)),
		batchSpec, seqSpec, StaticDim(numKHeads), StaticDim(headKDim),
	)
	value := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(2*keyDim, convDim)),
		batchSpec, seqSpec, StaticDim(numVHeads), StaticDim(headVDim),
	)

	// Compute beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias) in Float32.
	aLogVar := attnScope.VariableWithShape("A_log", shapes.Make(dtypes.Float32, numVHeads))
	dtBiasVar := attnScope.WithInitializer(initializer.One).VariableWithShape("dt_bias", shapes.Make(m.DType, numVHeads))

	beta := Sigmoid(ConvertDType(b, dtypes.Float32)) // [B, S, numVHeads]
	aF32 := ConvertDType(a, dtypes.Float32)
	dtBiasF32 := Reshape(ConvertDType(dtBiasVar.NodeValue(g), dtypes.Float32), 1, 1, numVHeads)
	negExpA := Reshape(Neg(Exp(ConvertDType(aLogVar.NodeValue(g), dtypes.Float32))), 1, 1, numVHeads)
	gDecay := Mul(negExpA, Softplus(Add(aF32, dtBiasF32))) // [B, S, numVHeads]

	// Grouped value attention: repeat_interleave Q and K heads to match numVHeads.
	if numKHeads != numVHeads {
		rep := numVHeads / numKHeads
		query = repeatInterleaveHeads(query, rep)
		key = repeatInterleaveHeads(key, rep)
	}

	// Promote Q, K, V to Float32, apply L2 norm on Q/K, and scale Q by 1/sqrt(headKDim).
	origDType := query.DType()
	qF32 := l2NormLastAxis(ConvertDType(query, dtypes.Float32), 1e-6)
	kF32 := l2NormLastAxis(ConvertDType(key, dtypes.Float32), 1e-6)
	qF32 = MulScalar(qF32, 1.0/math.Sqrt(float64(headKDim)))
	vF32 := ConvertDType(value, dtypes.Float32)

	var coreOutF32, newRecurrentState *Node
	if x.Shape().Dimensions[1] == 1 {
		coreOutF32, newRecurrentState = gatedDeltaRuleSingleStep(qF32, kF32, vF32, gDecay, beta, recurrentState)
	} else {
		coreOutF32, newRecurrentState = gatedDeltaRuleChunked(qF32, kF32, vF32, gDecay, beta, recurrentState, gatedDeltaRuleChunkSize)
	}
	coreOut := ConvertDType(coreOutF32, origDType)

	if cache != nil {
		stateDType := m.LinearStateDType
		if stateDType == dtypes.InvalidDType {
			stateDType = dtypes.Float32
		}
		if recurrentState != nil {
			stateDType = recurrentState.DType()
		}
		m.KVCache.SetLinearState(cache, attnScope.Scope(), newConvState, ConvertDType(newRecurrentState, stateDType))
	}

	// Gated RMSNorm (one-centered weight, no +1.0 offset, multiplied by silu(z)).
	normWeightVar := attnScope.In("norm").WithInitializer(initializer.One).VariableWithShape("weight", shapes.Make(dtypes.Float32, headVDim))
	coreOut = rmsNormGated(coreOut, z, normWeightVar.NodeValue(g), m.NormEpsilon)

	// Flatten heads and project back to hidden size.
	coreOut = DynamicReshape(coreOut, batchSpec, seqSpec, StaticDim(valueDim))
	return m.dense(attnScope.In("out_proj"), coreOut, false, m.EmbedDim)
}

func applyLinearPaddingMask(x *Node, options CallOptions) *Node {
	g := x.Graph()
	var mask *Node
	if options.AttentionMask != nil {
		m := options.AttentionMask
		if m.DType() != dtypes.Bool {
			m = NotEqual(m, ScalarZero(g, m.DType()))
		}
		mask = m
	} else if options.SeqLen != nil {
		seqLen := options.SeqLen
		if seqLen.Rank() == 0 {
			seqLen = ExpandAxes(seqLen, 0)
		}
		indices := DynamicIota(g, seqLen.DType(), 1, DimensionSpecFor(x, 0), DimensionSpecFor(x, 1))
		mask = LessThan(indices, ExpandAxes(seqLen, 1))
	}
	if mask == nil {
		return x
	}
	mask3D := BroadcastLike(ExpandDims(mask, -1), x)
	return Where(mask3D, x, ZerosLike(x))
}

// causalDepthwiseConv1D computes causal depthwise 1D convolution on channels-last input x [B, S, C]
// with weightKC [K, C] and optional previous state convState [B, K-1, C].
// Returns (out [B, S, C], newConvState [B, K-1, C] or nil when returnState is false).
func causalDepthwiseConv1D(x, convState, weightKC *Node, kernelSize int, returnState bool) (out, newConvState *Node) {
	g := x.Graph()
	origDType := x.DType()
	batchDim := x.Shape().Dimensions[0]
	seqDim := x.Shape().Dimensions[1]
	convDim := x.Shape().Dimensions[2]
	batchSpec := DimensionSpecFor(x, 0)
	seqSpec := DimensionSpecFor(x, 1)
	padLen := kernelSize - 1

	workDType := origDType
	if workDType.IsHalfPrecision() {
		workDType = dtypes.Float32
	}

	// Construct padded sequence [B, padLen + S, C].
	var padded *Node
	if convState != nil {
		padded = Concatenate([]*Node{convState, x}, 1)
	} else if padLen > 0 {
		zeros := DynamicBroadcastInDim(ScalarZero(g, origDType), nil, batchSpec, StaticDim(padLen), StaticDim(convDim))
		padded = Concatenate([]*Node{zeros, x}, 1)
	} else {
		padded = x
	}

	// Update conv state: last padLen tokens of padded [B, padLen, C].
	if returnState && padLen > 0 {
		if seqDim != shapes.DynamicDim {
			newConvState = Slice(padded, AxisRange(), AxisRange(seqDim, seqDim+padLen), AxisRange())
		} else {
			seqLenI32 := ConvertDType(DimensionSize(x, 1), dtypes.Int32)
			tailIdx := ExpandDims(Add(Iota(g, shapes.Make(dtypes.Int32, padLen), 0), seqLenI32), -1) // [padLen, 1]
			if batchDim == 1 {
				flat := DynamicReshape(padded, DimensionSpecFor(padded, 1), StaticDim(convDim))
				newConvState = ExpandDims(Gather(flat, tailIdx), 0)
			} else {
				pT := Transpose(padded, 0, 1) // [padLen+S, B, C]
				newConvState = Transpose(Gather(pT, tailIdx), 0, 1)
			}
		}
	}

	// Fast path for single-token decode (static S == 1): padded is [B, K, C].
	if seqDim == 1 {
		paddedW := ConvertDType(padded, workDType)
		wRow := ExpandDims(ConvertDType(weightKC, workDType), 0) // [1, K, C]
		acc := ReduceSum(Mul(paddedW, wRow), 1)                  // [B, C]
		return ExpandDims(ConvertDType(acc, origDType), 1), newConvState
	}

	// General path (static S > 1 or dynamic S): sum K shifted taps directly.
	var flatPadded, transposedPadded *Node
	var baseSeqIdx *Node
	if seqDim == shapes.DynamicDim {
		baseSeqIdx = DynamicIota(g, dtypes.Int32, 0, seqSpec)
		if batchDim == 1 {
			flatPadded = DynamicReshape(padded, DimensionSpecFor(padded, 1), StaticDim(convDim))
		} else {
			transposedPadded = Transpose(padded, 0, 1) // [padLen+S, B, C]
		}
	}

	var acc *Node
	for k := range kernelSize {
		var shifted *Node
		if k == kernelSize-1 {
			shifted = x
		} else if seqDim != shapes.DynamicDim {
			shifted = Slice(padded, AxisRange(), AxisRange(k, k+seqDim), AxisRange())
		} else {
			idx := baseSeqIdx
			if k > 0 {
				idx = AddScalar(baseSeqIdx, k)
			}
			idx2D := ExpandDims(idx, -1) // [S, 1]
			if batchDim == 1 {
				shifted = ExpandDims(Gather(flatPadded, idx2D), 0)
			} else {
				shifted = Transpose(Gather(transposedPadded, idx2D), 0, 1)
			}
		}
		wk := Reshape(Slice(weightKC, AxisElem(k), AxisRange()), 1, 1, convDim)
		term := Mul(ConvertDType(shifted, workDType), ConvertDType(wk, workDType))
		if acc == nil {
			acc = term
		} else {
			acc = Add(acc, term)
		}
	}
	return ConvertDType(acc, origDType), newConvState
}

func repeatInterleaveHeads(x *Node, repeats int) *Node {
	// x: [B, S, numKHeads, headDim] -> [B, S, numKHeads*repeats, headDim]
	batchSpec := DimensionSpecFor(x, 0)
	seqSpec := DimensionSpecFor(x, 1)
	numKHeads := x.Shape().Dimensions[2]
	headDim := x.Shape().Dimensions[3]
	expanded := ExpandDims(x, 3) // [B, S, numKHeads, 1, headDim]
	broadcasted := DynamicBroadcastInDim(
		expanded,
		[]int{0, 1, 2, 3, 4},
		batchSpec, seqSpec, StaticDim(numKHeads), StaticDim(repeats), StaticDim(headDim),
	)
	return DynamicReshape(broadcasted, batchSpec, seqSpec, StaticDim(numKHeads*repeats), StaticDim(headDim))
}

func l2NormLastAxis(x *Node, eps float64) *Node {
	sqSum := ReduceAndKeep(Square(x), ReduceSum, -1)
	return Mul(x, Rsqrt(AddScalar(sqSum, eps)))
}

// rmsNormGated implements Qwen3_5RMSNormGated:
//   normed = x_f32 * rsqrt(mean(x_f32^2, -1) + eps)
//   out = (weight_f32 * normed.to(origDType).to(f32) * silu(z_f32)).to(origDType)
// Note: weight is one-centered (NO +1.0 offset).
func rmsNormGated(x, z, weight *Node, eps float64) *Node {
	origDType := x.DType()
	xF32 := ConvertDType(x, dtypes.Float32)
	variance := ReduceAndKeep(Square(xF32), ReduceMean, -1)
	normed := Mul(xF32, Rsqrt(AddScalar(variance, eps)))
	if origDType != dtypes.Float32 {
		normed = ConvertDType(ConvertDType(normed, origDType), dtypes.Float32)
	}
	headVDim := x.Shape().Dim(-1)
	w := Reshape(ConvertDType(weight, dtypes.Float32), 1, 1, 1, headVDim)
	gate := activation.Apply(activation.TypeSilu, ConvertDType(z, dtypes.Float32))
	return ConvertDType(Mul(Mul(normed, w), gate), origDType)
}

// gatedDeltaRuleSingleStep executes one recurrent step of the gated delta rule for static seqLen == 1.
// Inputs (all Float32):
//   - q, k: [B, 1, H, Dk] (already L2-normalized; q already scaled by 1/sqrt(Dk))
//   - v:    [B, 1, H, Dv]
//   - g:    [B, 1, H] (log decay <= 0)
//   - beta: [B, 1, H] (in (0, 1))
//   - initialState: optional [B, H, Dk, Dv] (or nil for zeros)
//
// Returns (out [B, 1, H, Dv], newState [B, H, Dk, Dv]) in Float32.
func gatedDeltaRuleSingleStep(q, k, v, gDecay, beta, initialState *Node) (out, newState *Node) {
	graph := q.Graph()
	batchSpec := DimensionSpecFor(q, 0)
	numHeads := q.Shape().Dimensions[2]
	headKDim := q.Shape().Dimensions[3]
	headVDim := v.Shape().Dimensions[3]

	var state *Node
	if initialState == nil {
		state = DynamicBroadcastInDim(ScalarZero(graph, dtypes.Float32), nil,
			batchSpec, StaticDim(numHeads), StaticDim(headKDim), StaticDim(headVDim))
	} else {
		state = ConvertDType(initialState, dtypes.Float32)
	}

	q0 := Squeeze(q, 1) // [B, H, Dk]
	k0 := Squeeze(k, 1) // [B, H, Dk]
	v0 := Squeeze(v, 1) // [B, H, Dv]

	decay0 := ExpandAxes(Exp(Squeeze(gDecay, 1)), -2, -1) // [B, H, 1, 1]
	beta0 := ExpandDims(Squeeze(beta, 1), -1)             // [B, H, 1]

	// Decay state, retrieve memory along k, apply gated delta update, and read along q.
	state = Mul(state, decay0)
	kvMem := DotGeneral(k0, []int{2}, []int{0, 1}, state, []int{2}, []int{0, 1}) // [B, H, Dv]
	delta := Mul(Sub(v0, kvMem), beta0)                                          // [B, H, Dv]
	state = Add(state, Mul(ExpandDims(k0, -1), ExpandDims(delta, -2)))           // [B, H, Dk, Dv]
	out0 := DotGeneral(q0, []int{2}, []int{0, 1}, state, []int{2}, []int{0, 1})  // [B, H, Dv]

	return ExpandDims(out0, 1), state
}

// gatedDeltaRuleChunked implements the chunkwise gated delta rule (matching HuggingFace's
// torch_chunk_gated_delta_rule) for arbitrary static or dynamic sequence lengths.
// Inputs (all Float32):
//   - query, key: [B, S, H, Dk] (already L2-normalized; query already scaled by 1/sqrt(Dk))
//   - value:      [B, S, H, Dv]
//   - gDecay:     [B, S, H] (log decay <= 0)
//   - beta:       [B, S, H]
//   - initialState: optional [B, H, Dk, Dv] (or nil for zeros)
//
// Returns (out [B, S, H, Dv], finalState [B, H, Dk, Dv]) in Float32.
func gatedDeltaRuleChunked(query, key, value, gDecay, beta, initialState *Node, chunkSize int) (out, finalState *Node) {
	g := query.Graph()
	batchSpec := DimensionSpecFor(query, 0)
	seqDim := query.Shape().Dimensions[1]
	seqAxisName := query.Shape().AxisName(1)
	numHeads := query.Shape().Dimensions[2]
	headKDim := query.Shape().Dimensions[3]
	headVDim := value.Shape().Dimensions[3]

	// Transpose to [B, H, S, D] and [B, H, S].
	q := TransposeAllDims(query, 0, 2, 1, 3)
	k := TransposeAllDims(key, 0, 2, 1, 3)
	v := TransposeAllDims(value, 0, 2, 1, 3)
	bT := TransposeAllDims(beta, 0, 2, 1)
	dT := TransposeAllDims(gDecay, 0, 2, 1)

	// Pad sequence axis (axis 2) to a multiple of chunkSize with zeros.
	// Zero-padded positions have k=0, v=0, beta=0, decay=0 (exp(0)=1), leaving the recurrent state unchanged.
	zeroF32 := ScalarZero(g, dtypes.Float32)
	var padSizeNode *Node
	var staticPadSize int
	if seqDim != shapes.DynamicDim {
		staticPadSize = (chunkSize - seqDim%chunkSize) % chunkSize
		if staticPadSize > 0 {
			q = Pad(q, zeroF32, PadAxis{}, PadAxis{}, PadAxis{End: staticPadSize}, PadAxis{})
			k = Pad(k, zeroF32, PadAxis{}, PadAxis{}, PadAxis{End: staticPadSize}, PadAxis{})
			v = Pad(v, zeroF32, PadAxis{}, PadAxis{}, PadAxis{End: staticPadSize}, PadAxis{})
			bT = Pad(bT, zeroF32, PadAxis{}, PadAxis{}, PadAxis{End: staticPadSize})
			dT = Pad(dT, zeroF32, PadAxis{}, PadAxis{}, PadAxis{End: staticPadSize})
		}
	} else {
		seqLenDyn := DimensionSize(q, 2)
		cNode := Scalar(g, seqLenDyn.DType(), chunkSize)
		padSizeNode = Rem(Sub(cNode, Rem(seqLenDyn, cNode)), cNode)
		paddedAxisName := shapes.UniqueAxisName("gdn_padded_seq")
		pad4D := []DynamicPadAxisSpec{
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			{EndNode: padSizeNode, TargetAxisName: paddedAxisName},
			DynamicPadAxis(0, 0, 0),
		}
		pad3D := []DynamicPadAxisSpec{
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			{EndNode: padSizeNode, TargetAxisName: paddedAxisName},
		}
		q = DynamicPad(q, zeroF32, pad4D...)
		k = DynamicPad(k, zeroF32, pad4D...)
		v = DynamicPad(v, zeroF32, pad4D...)
		bT = DynamicPad(bT, zeroF32, pad3D...)
		dT = DynamicPad(dT, zeroF32, pad3D...)
	}

	// Scale K and V by beta.
	bExpanded := ExpandDims(bT, -1) // [B, H, totalSeq, 1]
	vBeta := Mul(v, bExpanded)
	kBeta := Mul(k, bExpanded)

	// Reshape into chunks: [B, H, numChunks, C, D] and [B, H, numChunks, C].
	// Share a single numChunksSpec across all tensors so dynamic axis names unify identically.
	q = DynamicReshape(q, batchSpec, StaticDim(numHeads), InferredDim(), StaticDim(chunkSize), StaticDim(headKDim))
	numChunksSpec := DimensionSpecFor(q, 2)
	k = DynamicReshape(k, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headKDim))
	kBeta = DynamicReshape(kBeta, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headKDim))
	vBeta = DynamicReshape(vBeta, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headVDim))
	dT = DynamicReshape(dT, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize))

	// Cumulative log-decay within each chunk: [B, H, numChunks, C].
	cumDecay := CumSum(dT, 3)

	// Pairwise decay matrix: pairwise_decay[..., i, j] = exp(cumDecay_i - cumDecay_j) for j <= i, else 0.
	// Mask strictly upper triangle (j > i) to -inf before Exp to avoid overflow.
	strictlyUpperMask := makeUpperTriangularMask(g, chunkSize, 1) // [1, 1, 1, C, C]
	pairwiseDiff := Sub(ExpandDims(cumDecay, 4), ExpandDims(cumDecay, 3))
	negInf := Infinity(g, dtypes.Float32, -1)
	pairwiseDiff = Where(BroadcastLike(strictlyUpperMask, pairwiseDiff), BroadcastLike(negInf, pairwiseDiff), pairwiseDiff)
	pairwiseDecay := Exp(pairwiseDiff) // [B, H, numChunks, C, C]

	// Strictly lower triangle mask (j < i) for UT transform system (I + tril(utSystem, -1)).
	strictlyLowerMask := makeLowerTriangularMask(g, chunkSize, -1) // [1, 1, 1, C, C]
	utSystem := Mul(DotGeneral(kBeta, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2}), pairwiseDecay)
	utLower := Where(BroadcastLike(strictlyLowerMask, utSystem), utSystem, ZerosLike(utSystem))

	// Exact block-recursive inverse of unit lower-triangular matrix (I + utLower).
	utInv := invertUnitLowerTriangular64(utLower)

	// Intra-chunk quantities.
	expCumDecay := Exp(cumDecay) // [B, H, numChunks, C]
	decayedKBeta := Mul(kBeta, ExpandDims(expCumDecay, -1))
	newValues := DotGeneral(utInv, []int{4}, []int{0, 1, 2}, vBeta, []int{3}, []int{0, 1, 2})        // [B, H, numChunks, C, Dv]
	kCumDecay := DotGeneral(utInv, []int{4}, []int{0, 1, 2}, decayedKBeta, []int{3}, []int{0, 1, 2}) // [B, H, numChunks, C, Dk]
	intraChunkAttn := Mul(DotGeneral(q, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2}), pairwiseDecay)

	// Pre-apply chunk-relative decays to q, k, and state decay factor.
	cumDecayLast := Slice(cumDecay, AxisRange().Spacer(), AxisRange(chunkSize-1, chunkSize)) // [B, H, numChunks, 1]
	qDecayed := Mul(q, ExpandDims(expCumDecay, -1))                                          // [B, H, numChunks, C, Dk]
	kDecayed := Mul(k, ExpandDims(Exp(Sub(cumDecayLast, cumDecay)), -1))                     // [B, H, numChunks, C, Dk]
	chunkDecay := ExpandDims(Exp(cumDecayLast), -1)                                          // [B, H, numChunks, 1, 1]

	// Initial recurrent state: [B, H, Dk, Dv].
	var initState *Node
	if initialState == nil {
		initState = DynamicBroadcastInDim(zeroF32, nil,
			batchSpec, StaticDim(numHeads), StaticDim(headKDim), StaticDim(headVDim))
	} else {
		initState = ConvertDType(initialState, dtypes.Float32)
	}

	// Inter-chunk sequential scan.
	var coreOutChunks *Node // [B, H, numChunks, C, Dv]
	numChunksStatic := q.Shape().Dimensions[2]
	if numChunksStatic == 1 {
		// Unrolled single-chunk fast path (no While loop).
		nv := Squeeze(newValues, 2)
		kc := Squeeze(kCumDecay, 2)
		qc := Squeeze(qDecayed, 2)
		kd := Squeeze(kDecayed, 2)
		ia := Squeeze(intraChunkAttn, 2)
		cd := Squeeze(chunkDecay, 2)

		vNew := Sub(nv, DotGeneral(kc, []int{3}, []int{0, 1}, initState, []int{2}, []int{0, 1}))
		interAttn := DotGeneral(qc, []int{3}, []int{0, 1}, initState, []int{2}, []int{0, 1})
		chunkOut := Add(interAttn, DotGeneral(ia, []int{3}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))
		finalState = Add(Mul(initState, cd), DotGeneral(kd, []int{2}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))
		coreOutChunks = ExpandDims(chunkOut, 2)
	} else {
		// Move numChunks to axis 0 so each While iteration performs a contiguous Gather & DynamicUpdateSlice.
		nv0 := TransposeAllDims(newValues, 2, 0, 1, 3, 4)      // [numChunks, B, H, C, Dv]
		kc0 := TransposeAllDims(kCumDecay, 2, 0, 1, 3, 4)      // [numChunks, B, H, C, Dk]
		qc0 := TransposeAllDims(qDecayed, 2, 0, 1, 3, 4)       // [numChunks, B, H, C, Dk]
		kd0 := TransposeAllDims(kDecayed, 2, 0, 1, 3, 4)       // [numChunks, B, H, C, Dk]
		ia0 := TransposeAllDims(intraChunkAttn, 2, 0, 1, 3, 4) // [numChunks, B, H, C, C]
		cd0 := TransposeAllDims(chunkDecay, 2, 0, 1, 3, 4)     // [numChunks, B, H, 1, 1]

		outInit := ZerosLike(nv0) // [numChunks, B, H, C, Dv]
		numChunksNode := ConvertDType(DimensionSize(nv0, 0), dtypes.Int32)

		condFn := NewClosure(g, func(g *Graph) []*Node {
			idx := Parameter(g, "idx", shapes.Scalar[int32]())
			_ = Parameter(g, "state", initState.Shape())
			_ = Parameter(g, "out", outInit.Shape())
			return []*Node{LessThan(idx, numChunksNode)}
		})

		bodyFn := NewClosure(g, func(g *Graph) []*Node {
			idx := Parameter(g, "idx", shapes.Scalar[int32]())
			st := Parameter(g, "state", initState.Shape())
			outBuf := Parameter(g, "out", outInit.Shape())

			nv := Gather(nv0, idx) // [B, H, C, Dv]
			kc := Gather(kc0, idx) // [B, H, C, Dk]
			qc := Gather(qc0, idx) // [B, H, C, Dk]
			kd := Gather(kd0, idx) // [B, H, C, Dk]
			ia := Gather(ia0, idx) // [B, H, C, C]
			cd := Gather(cd0, idx) // [B, H, 1, 1]

			vNew := Sub(nv, DotGeneral(kc, []int{3}, []int{0, 1}, st, []int{2}, []int{0, 1}))
			interAttn := DotGeneral(qc, []int{3}, []int{0, 1}, st, []int{2}, []int{0, 1})
			chunkOut := Add(interAttn, DotGeneral(ia, []int{3}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))
			newSt := Add(Mul(st, cd), DotGeneral(kd, []int{2}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))

			zero := Const(g, int32(0))
			newOut := DynamicUpdateSlice(outBuf, ExpandDims(chunkOut, 0), []*Node{idx, zero, zero, zero, zero})
			return []*Node{Add(idx, Const(g, int32(1))), newSt, newOut}
		})

		results := While(condFn, bodyFn, Const(g, int32(0)), initState, outInit)
		finalState = results[1]
		coreOutChunks = TransposeAllDims(results[2], 1, 2, 0, 3, 4) // [B, H, numChunks, C, Dv]
	}

	// Flatten chunk dimension back to [B, H, totalSeq, Dv], drop padding to [B, H, S, Dv], and transpose to [B, S, H, Dv].
	coreOutFlat := DynamicReshape(coreOutChunks, batchSpec, StaticDim(numHeads), InferredDim(), StaticDim(headVDim))
	if seqDim != shapes.DynamicDim {
		if staticPadSize > 0 {
			coreOutFlat = Slice(coreOutFlat, AxisRange(), AxisRange(), AxisRange(0, seqDim), AxisRange())
		}
	} else {
		coreOutFlat = DynamicPad(coreOutFlat, zeroF32,
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxisSpec{EndNode: Neg(padSizeNode), TargetAxisName: seqAxisName},
			DynamicPadAxis(0, 0, 0),
		)
	}
	return TransposeAllDims(coreOutFlat, 0, 2, 1, 3), finalState
}

// invertUnitLowerTriangular64 computes the exact inverse of (I + L) where L is a strictly
// lower-triangular matrix of shape [..., 64, 64] using a numerically stable block-recursive
// Schur complement decomposition (1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64).
func invertUnitLowerTriangular64(L *Node) *Node {
	g := L.Graph()
	rank := L.Rank()
	c := L.Shape().Dimensions[rank-1]
	if c != 64 {
		exceptions.Panicf("invertUnitLowerTriangular64 expects last two dimensions to be 64x64, got %s", L.Shape())
	}

	// Step 1: Extract the 4 diagonal 16x16 blocks of L and stack into [..., 4, 16, 16].
	diag16 := make([]*Node, 4)
	for b := range 4 {
		s, e := b*16, (b+1)*16
		diag16[b] = Slice(L, AxisRange().Spacer(), AxisRange(s, e), AxisRange(s, e))
	}
	L16 := Stack(diag16, rank-2) // [..., 4, 16, 16]

	// Invert all 4 16x16 unit lower-triangular blocks in parallel via block-doubling (1->2->4->8->16).
	// Partition L16 into non-overlapping dyadic off-diagonal levels L_0 (m=1), L_1 (m=2), L_2 (m=4), L_3 (m=8):
	//   D_1 = I - L_0
	//   D_{k+1} = D_k - D_k @ L_k @ D_k
	eye16 := makeIdentityMatrix(g, 16, rank+1)
	D := Sub(BroadcastLike(eye16, L16), Mul(L16, makeDyadicLowerBlockMask(g, 16, 1, rank+1)))
	batchAxes16 := make([]int, rank-1)
	for i := range batchAxes16 {
		batchAxes16[i] = i
	}
	for _, m := range []int{2, 4, 8} {
		Lm := Mul(L16, makeDyadicLowerBlockMask(g, 16, m, rank+1))
		DL := DotGeneral(D, []int{rank}, batchAxes16, Lm, []int{rank - 1}, batchAxes16)
		DLD := DotGeneral(DL, []int{rank}, batchAxes16, D, []int{rank - 1}, batchAxes16)
		D = Sub(D, DLD)
	}
	// D has shape [..., 4, 16, 16] holding the 4 exact 16x16 diagonal block inverses.

	// Step 2: Merge pairs of 16x16 blocks -> two 32x32 blocks [..., 2, 32, 32].
	inv0 := Slice(D, AxisRange().Spacer(), AxisElem(0), AxisRange(), AxisRange())
	inv1 := Slice(D, AxisRange().Spacer(), AxisElem(1), AxisRange(), AxisRange())
	inv2 := Slice(D, AxisRange().Spacer(), AxisElem(2), AxisRange(), AxisRange())
	inv3 := Slice(D, AxisRange().Spacer(), AxisElem(3), AxisRange(), AxisRange())
	A16 := Concatenate([]*Node{inv0, inv2}, rank-2) // [..., 2, 16, 16]
	B16 := Concatenate([]*Node{inv1, inv3}, rank-2) // [..., 2, 16, 16]

	C01 := Slice(L, AxisRange().Spacer(), AxisRange(16, 32), AxisRange(0, 16))
	C23 := Slice(L, AxisRange().Spacer(), AxisRange(48, 64), AxisRange(32, 48))
	C16 := Stack([]*Node{C01, C23}, rank-2) // [..., 2, 16, 16]

	BC16 := DotGeneral(B16, []int{rank}, batchAxes16, C16, []int{rank - 1}, batchAxes16)
	LL16 := Neg(DotGeneral(BC16, []int{rank}, batchAxes16, A16, []int{rank - 1}, batchAxes16))
	top32 := Concatenate([]*Node{A16, ZerosLike(A16)}, rank)
	bot32 := Concatenate([]*Node{LL16, B16}, rank)
	inv32 := Concatenate([]*Node{top32, bot32}, rank-1) // [..., 2, 32, 32]

	// Step 3: Merge the two 32x32 blocks -> final 64x64 inverse [..., 64, 64].
	A32 := Squeeze(Slice(inv32, AxisRange().Spacer(), AxisElem(0), AxisRange(), AxisRange()), rank-2) // [..., 32, 32]
	B32 := Squeeze(Slice(inv32, AxisRange().Spacer(), AxisElem(1), AxisRange(), AxisRange()), rank-2) // [..., 32, 32]
	C32 := Slice(L, AxisRange().Spacer(), AxisRange(32, 64), AxisRange(0, 32))                        // [..., 32, 32]

	batchAxes32 := make([]int, rank-2)
	for i := range batchAxes32 {
		batchAxes32[i] = i
	}
	BC32 := DotGeneral(B32, []int{rank - 1}, batchAxes32, C32, []int{rank - 2}, batchAxes32)
	LL32 := Neg(DotGeneral(BC32, []int{rank - 1}, batchAxes32, A32, []int{rank - 2}, batchAxes32))
	top64 := Concatenate([]*Node{A32, ZerosLike(A32)}, rank-1)
	bot64 := Concatenate([]*Node{LL32, B32}, rank-1)
	return Concatenate([]*Node{top64, bot64}, rank-2)
}

func makeUpperTriangularMask(g *Graph, n, diagonalOffset int) *Node {
	data := make([]bool, n*n)
	for i := range n {
		for j := range n {
			if j >= i+diagonalOffset {
				data[i*n+j] = true
			}
		}
	}
	return Reshape(Const(g, data), 1, 1, 1, n, n)
}

func makeLowerTriangularMask(g *Graph, n, diagonalOffset int) *Node {
	data := make([]bool, n*n)
	for i := range n {
		for j := range n {
			if j <= i+diagonalOffset {
				data[i*n+j] = true
			}
		}
	}
	return Reshape(Const(g, data), 1, 1, 1, n, n)
}

func makeIdentityMatrix(g *Graph, n, targetRank int) *Node {
	data := make([]float32, n*n)
	for i := range n {
		data[i*n+i] = 1.0
	}
	dims := make([]int, targetRank)
	for i := range targetRank - 2 {
		dims[i] = 1
	}
	dims[targetRank-2] = n
	dims[targetRank-1] = n
	return Reshape(Const(g, data), dims...)
}

// makeDyadicLowerBlockMask creates a [1,...,1, n, n] float32 mask with 1.0 at the lower-left
// m x m sub-blocks of each (2m) x (2m) diagonal block, and 0.0 elsewhere.
func makeDyadicLowerBlockMask(g *Graph, n, m, targetRank int) *Node {
	data := make([]float32, n*n)
	blockSize := 2 * m
	for b := 0; b < n; b += blockSize {
		for r := b + m; r < b+blockSize; r++ {
			for c := b; c < b+m; c++ {
				data[r*n+c] = 1.0
			}
		}
	}
	dims := make([]int, targetRank)
	for i := range targetRank - 2 {
		dims[i] = 1
	}
	dims[targetRank-2] = n
	dims[targetRank-1] = n
	return Reshape(Const(g, data), dims...)
}
