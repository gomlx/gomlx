// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package linear

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/exceptions"
)

// DecayMode specifies the form of multiplicative state decay in a linear attention layer.
type DecayMode int

const (
	// DecayNone disables multiplicative state decay (Simple Linear Attention or un-gated DeltaNet).
	DecayNone DecayMode = iota

	// DecayScalar applies a data-dependent scalar log-decay per head (Mamba-2 / SSD, Gated DeltaNet / Qwen3.5):
	// g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias), shaped [B, S, numVHeads].
	DecayScalar

	// DecayChannelWise applies a data-dependent log-decay per key channel (GLA, Kimi Delta Attention):
	// g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias), shaped [B, S, numVHeads, headKDim].
	DecayChannelWise
)

// Builder configures a stateful linear attention / (Gated) DeltaNet layer.
// Create one via [New], [GatedDeltaNet], [DeltaNet], [GLA], or [Simple], configure with fluent
// With* methods, and finish with [Builder.Done] or [Builder.DoneWithState].
type Builder struct {
	scope *model.Scope
	x     *Node

	numKHeads int
	numVHeads int
	headKDim  int
	headVDim  int
	outputDim int

	decayMode       DecayMode
	useDeltaRule    bool
	useQKL2Norm     bool
	queryScale      float64
	convKernelSize  int
	act             activation.Type
	useOutputGate   bool
	useOutputNorm   bool
	normEpsilon     float64
	useBias         bool
	transposed      bool
	stateDType      dtypes.DType
	chunkSize       int
	unrollChunks    *bool
	convState       *Node
	recurrentState  *Node
	returnState     bool
	attentionMask   *Node
	seqLen          *Node
}

// New creates a Builder configured by default for Gated DeltaNet (scalar decay gate, delta rule,
// causal depthwise Conv1d with kernel size 4, SiLU activation, QK L2 normalization, and gated RMSNorm).
//
// Input x must have shape [batchSize, seqLen, embedDim].
// numHeads sets both the key and value head counts (override key heads via [Builder.WithNumKeyHeads]),
// and headDim sets both key and value head dimensions (override via [Builder.WithKeyHeadDim] / [Builder.WithValueHeadDim]).
func New(scope *model.Scope, x *Node, numHeads, headDim int) *Builder {
	if x.Rank() != 3 {
		exceptions.Panicf("linear.New expects rank-3 input [batch, seq, embedDim], got %s", x.Shape())
	}
	return &Builder{
		scope:          scope,
		x:              x,
		numKHeads:      numHeads,
		numVHeads:      numHeads,
		headKDim:       headDim,
		headVDim:       headDim,
		outputDim:      x.Shape().Dimensions[2],
		decayMode:      DecayScalar,
		useDeltaRule:   true,
		useQKL2Norm:    true,
		convKernelSize: 4,
		act:            activation.TypeSilu,
		useOutputGate:  true,
		useOutputNorm:  true,
		normEpsilon:    1e-6,
		stateDType:     dtypes.Float32,
	}
}

// GatedDeltaNet creates a Builder configured for Gated DeltaNet (Yang et al., ICLR 2025; Qwen3.5).
// Equivalent to [New].
func GatedDeltaNet(scope *model.Scope, x *Node, numHeads, headDim int) *Builder {
	return New(scope, x, numHeads, headDim)
}

// DeltaNet creates a Builder configured for un-gated DeltaNet (DecayNone + DeltaRule enabled).
func DeltaNet(scope *model.Scope, x *Node, numHeads, headDim int) *Builder {
	return New(scope, x, numHeads, headDim).WithDecayMode(DecayNone)
}

// GLA creates a Builder configured for Gated Linear Attention (DecayChannelWise + DeltaRule disabled).
func GLA(scope *model.Scope, x *Node, numHeads, headDim int) *Builder {
	return New(scope, x, numHeads, headDim).
		WithDecayMode(DecayChannelWise).
		WithDeltaRule(false).
		WithQKL2Norm(false)
}

// Simple creates a Builder configured for simple un-gated linear attention
// (DecayNone, DeltaRule disabled, no short conv1d, no output gate, plain RMSNorm).
func Simple(scope *model.Scope, x *Node, numHeads, headDim int) *Builder {
	return New(scope, x, numHeads, headDim).
		WithDecayMode(DecayNone).
		WithDeltaRule(false).
		WithShortConv(0).
		WithOutputGate(false).
		WithQKL2Norm(false)
}

// WithNumKeyHeads configures Grouped Value Attention (GVA), where numVHeads is a multiple of numKHeads.
func (b *Builder) WithNumKeyHeads(numKHeads int) *Builder {
	b.numKHeads = numKHeads
	return b
}

// WithKeyHeadDim sets the key/query head dimension separately from the value head dimension.
func (b *Builder) WithKeyHeadDim(dim int) *Builder {
	b.headKDim = dim
	return b
}

// WithValueHeadDim sets the value head dimension separately from the key/query head dimension.
func (b *Builder) WithValueHeadDim(dim int) *Builder {
	b.headVDim = dim
	return b
}

// WithOutputDim sets the output embedding dimension of out_proj (defaults to input embedDim).
func (b *Builder) WithOutputDim(dim int) *Builder {
	b.outputDim = dim
	return b
}

// WithDecayMode sets the state decay mechanism (DecayNone, DecayScalar, or DecayChannelWise).
func (b *Builder) WithDecayMode(mode DecayMode) *Builder {
	b.decayMode = mode
	return b
}

// WithDeltaRule enables or disables the delta-rule surgical erasure (and the in_proj_b projection).
func (b *Builder) WithDeltaRule(enabled bool) *Builder {
	b.useDeltaRule = enabled
	return b
}

// WithShortConv sets the causal depthwise 1D convolution kernel size over concatenated [Q, K, V].
// Set <= 0 to disable the short convolution.
func (b *Builder) WithShortConv(kernelSize int) *Builder {
	b.convKernelSize = kernelSize
	return b
}

// WithActivation sets the activation applied after the short convolution (default SiLU).
func (b *Builder) WithActivation(act activation.Type) *Builder {
	b.act = act
	return b
}

// WithQKL2Norm toggles L2 normalization on Q and K before the recurrence.
func (b *Builder) WithQKL2Norm(enabled bool) *Builder {
	b.useQKL2Norm = enabled
	return b
}

// WithQueryScale overrides the default 1/sqrt(headKDim) query scaling factor.
// Set to a negative value (e.g. -1) to disable scaling.
func (b *Builder) WithQueryScale(scale float64) *Builder {
	b.queryScale = scale
	return b
}

// WithOutputGate enables or disables the multiplicative output gate projection (in_proj_z + RMSNormGated).
func (b *Builder) WithOutputGate(enabled bool) *Builder {
	b.useOutputGate = enabled
	return b
}

// WithOutputNorm enables or disables output normalization before out_proj when output gating is disabled.
// (When WithOutputGate(true) is active, RMSNormGated is always applied.)
func (b *Builder) WithOutputNorm(enabled bool) *Builder {
	b.useOutputNorm = enabled
	return b
}

// WithNormEpsilon sets the epsilon for output (gated) RMSNorm (default 1e-6).
func (b *Builder) WithNormEpsilon(eps float64) *Builder {
	b.normEpsilon = eps
	return b
}

// WithBias enables or disables biases in the linear projection layers (default false).
func (b *Builder) WithBias(useBias bool) *Builder {
	b.useBias = useBias
	return b
}

// UseTransposedWeights configures linear layers to expect PyTorch-style [outDim, inDim] weight layout.
func (b *Builder) UseTransposedWeights(transposed bool) *Builder {
	b.transposed = transposed
	return b
}

// WithStateDType sets the dtype of the returned recurrent state tensor (default Float32).
func (b *Builder) WithStateDType(dtype dtypes.DType) *Builder {
	b.stateDType = dtype
	return b
}

// WithChunkSize sets the chunk length for chunkwise scan (must be a power of 2 >= 2; 0 uses default).
func (b *Builder) WithChunkSize(chunkSize int) *Builder {
	b.chunkSize = chunkSize
	return b
}

// WithUnrollChunks forces unrolled inter-chunk scanning (true) or graph.While scanning (false) for
// static sequence lengths. By default, chunks are automatically unrolled when scope.IsTraining(g) is true
// so reverse-mode autodiff works out of the box.
func (b *Builder) WithUnrollChunks(unroll bool) *Builder {
	b.unrollChunks = &unroll
	return b
}

// WithState supplies the previous causal convolution state ([B, K-1, convDim]) and recurrent state
// ([B, numVHeads, headKDim, headVDim]) for incremental cached decoding. Passing non-nil states
// also enables returning updated states from [Builder.DoneWithState].
func (b *Builder) WithState(convState, recurrentState *Node) *Builder {
	b.convState = convState
	b.recurrentState = recurrentState
	if convState != nil || recurrentState != nil {
		b.returnState = true
	}
	return b
}

// WithReturnState explicitly requests updated (convState, recurrentState) outputs from [Builder.DoneWithState]
// even when starting from zero initial states (e.g., cached prefill).
func (b *Builder) WithReturnState(ret bool) *Builder {
	b.returnState = ret
	return b
}

// WithAttentionMask zeros out padded token positions in the input before projections
// (2D tensor [B, S] where non-zero/true indicates valid tokens). Mutually exclusive with WithSeqLen.
func (b *Builder) WithAttentionMask(mask *Node) *Builder {
	b.attentionMask = mask
	return b
}

// WithSeqLen zeros out padded token positions past per-batch lengths [B] (or scalar) before projections.
// Mutually exclusive with WithAttentionMask.
func (b *Builder) WithSeqLen(seqLen *Node) *Builder {
	b.seqLen = seqLen
	return b
}

// Done builds the layer graph and returns the output tensor shaped [B, S, outputDim].
func (b *Builder) Done() *Node {
	out, _, _ := b.DoneWithState()
	return out
}

// DoneWithState builds the layer graph and returns (output [B, S, outputDim], newConvState, newRecurrentState).
// newConvState is nil if short convolution is disabled or state output was not requested;
// newRecurrentState is nil if state output was not requested.
func (b *Builder) DoneWithState() (output, newConvState, newRecurrentState *Node) {
	if b.numKHeads <= 0 || b.numVHeads <= 0 || b.headKDim <= 0 || b.headVDim <= 0 {
		exceptions.Panicf("linear.Builder requires positive numKHeads (%d), numVHeads (%d), headKDim (%d), headVDim (%d)",
			b.numKHeads, b.numVHeads, b.headKDim, b.headVDim)
	}
	if b.numVHeads%b.numKHeads != 0 {
		exceptions.Panicf("linear.Builder requires numVHeads (%d) divisible by numKHeads (%d)", b.numVHeads, b.numKHeads)
	}
	if b.attentionMask != nil && b.seqLen != nil {
		exceptions.Panicf("linear.Builder: AttentionMask and SeqLen cannot both be set")
	}

	g := b.x.Graph()
	x := applyPaddingMask(b.x, b.attentionMask, b.seqLen)
	origDType := x.DType()
	batchSpec := DimensionSpecFor(x, 0)
	seqSpec := DimensionSpecFor(x, 1)

	keyDim := b.numKHeads * b.headKDim
	valueDim := b.numVHeads * b.headVDim
	convDim := 2*keyDim + valueDim

	// Input projections.
	mixedQKV := b.dense(b.scope.In("in_proj_qkv"), x, convDim) // [B, S, convDim]
	var z *Node
	if b.useOutputGate {
		z = b.dense(b.scope.In("in_proj_z"), x, b.numVHeads, b.headVDim) // [B, S, numVHeads, headVDim]
	}

	var betaNode *Node
	if b.useDeltaRule {
		bProj := b.dense(b.scope.In("in_proj_b"), x, b.numVHeads) // [B, S, numVHeads]
		betaNode = Sigmoid(ConvertDType(bProj, dtypes.Float32))
	}

	var logDecay *Node
	switch b.decayMode {
	case DecayScalar:
		aProj := b.dense(b.scope.In("in_proj_a"), x, b.numVHeads) // [B, S, numVHeads]
		aLogVar := b.scope.VariableWithShape("A_log", shapes.Make(dtypes.Float32, b.numVHeads))
		dtBiasVar := b.scope.WithInitializer(initializer.One).VariableWithShape("dt_bias", shapes.Make(origDType, b.numVHeads))
		aF32 := ConvertDType(aProj, dtypes.Float32)
		dtBiasF32 := Reshape(ConvertDType(dtBiasVar.NodeValue(g), dtypes.Float32), 1, 1, b.numVHeads)
		negExpA := Reshape(Neg(Exp(ConvertDType(aLogVar.NodeValue(g), dtypes.Float32))), 1, 1, b.numVHeads)
		logDecay = Mul(negExpA, Softplus(Add(aF32, dtBiasF32))) // [B, S, numVHeads]

	case DecayChannelWise:
		aProj := b.dense(b.scope.In("in_proj_a"), x, b.numVHeads, b.headKDim) // [B, S, numVHeads, headKDim]
		aLogVar := b.scope.VariableWithShape("A_log", shapes.Make(dtypes.Float32, b.numVHeads, b.headKDim))
		dtBiasVar := b.scope.WithInitializer(initializer.One).VariableWithShape("dt_bias", shapes.Make(origDType, b.numVHeads, b.headKDim))
		aF32 := ConvertDType(aProj, dtypes.Float32)
		dtBiasF32 := Reshape(ConvertDType(dtBiasVar.NodeValue(g), dtypes.Float32), 1, 1, b.numVHeads, b.headKDim)
		negExpA := Reshape(Neg(Exp(ConvertDType(aLogVar.NodeValue(g), dtypes.Float32))), 1, 1, b.numVHeads, b.headKDim)
		logDecay = Mul(negExpA, Softplus(Add(aF32, dtBiasF32))) // [B, S, numVHeads, headKDim]
	}

	// Optional causal depthwise Conv1d + activation.
	if b.convKernelSize > 0 {
		convWeightVar := b.scope.VariableWithShape("conv1d_weight", shapes.Make(origDType, convDim, 1, b.convKernelSize))
		convWeightKC := Transpose(Squeeze(convWeightVar.NodeValue(g), 1), 0, 1) // [K, convDim]
		mixedQKV, newConvState = CausalDepthwiseConv1D(mixedQKV, b.convState, convWeightKC, b.convKernelSize, b.returnState)
		if b.act != activation.TypeNone {
			mixedQKV = activation.Apply(b.act, mixedQKV)
		}
	}

	// Split Q, K, V.
	query := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(0, keyDim)),
		batchSpec, seqSpec, StaticDim(b.numKHeads), StaticDim(b.headKDim),
	)
	key := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(keyDim, 2*keyDim)),
		batchSpec, seqSpec, StaticDim(b.numKHeads), StaticDim(b.headKDim),
	)
	value := DynamicReshape(
		Slice(mixedQKV, AxisRange().Spacer(), AxisRange(2*keyDim, convDim)),
		batchSpec, seqSpec, StaticDim(b.numVHeads), StaticDim(b.headVDim),
	)

	unroll := b.scope.IsTraining(g)
	if b.unrollChunks != nil {
		unroll = *b.unrollChunks
	}

	coreOut, finalState := Core(query, key, value, CoreOptions{
		LogDecay:     logDecay,
		Beta:         betaNode,
		InitialState: b.recurrentState,
		UseQKL2Norm:  b.useQKL2Norm,
		QueryScale:   b.queryScale,
		ChunkSize:    b.chunkSize,
		UnrollChunks: unroll,
	})

	if b.returnState {
		stateDType := b.stateDType
		if stateDType == dtypes.InvalidDType {
			stateDType = dtypes.Float32
		}
		if b.recurrentState != nil {
			stateDType = b.recurrentState.DType()
		}
		newRecurrentState = ConvertDType(finalState, stateDType)
	}

	// Output normalization & optional gating.
	if b.useOutputGate {
		normWeightVar := b.scope.In("norm").WithInitializer(initializer.One).VariableWithShape("weight", shapes.Make(dtypes.Float32, b.headVDim))
		coreOut = RMSNormGated(coreOut, z, normWeightVar.NodeValue(g), b.normEpsilon)
	} else if b.useOutputNorm {
		coreOut = norm.RMSNorm(b.scope.In("norm"), coreOut).
			WithEpsilon(b.normEpsilon).
			WithNormalizationAxes(-1).
			Done()
	}

	// Flatten heads and project back to outputDim.
	coreOut = DynamicReshape(coreOut, batchSpec, seqSpec, StaticDim(valueDim))
	output = b.dense(b.scope.In("out_proj"), coreOut, b.outputDim)
	return output, newConvState, newRecurrentState
}

func (b *Builder) dense(scope *model.Scope, x *Node, outputDims ...int) *Node {
	layout := compute.DenseLayoutInputOutputs
	if b.transposed {
		layout = compute.DenseLayoutOutputsInput
	}
	return layers.NewDense(scope, x, outputDims...).
		WithBias(b.useBias).
		WithLayout(layout).
		Done()
}

func applyPaddingMask(x, attentionMask, seqLen *Node) *Node {
	g := x.Graph()
	var mask *Node
	if attentionMask != nil {
		m := attentionMask
		if m.DType() != dtypes.Bool {
			m = NotEqual(m, ScalarZero(g, m.DType()))
		}
		mask = m
	} else if seqLen != nil {
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
