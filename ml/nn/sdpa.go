// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"math"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
	. "github.com/gomlx/gomlx/support/exceptions"
	"k8s.io/klog/v2"
)

// FusedSDPAConfig holds optional configuration for [FusedScaledDotProductAttention].
// Zero values provide sensible defaults (Scale = 1/sqrt(headDim), no mask, non-causal).
type FusedSDPAConfig struct {
	// Scale is the scaling factor for attention scores (typically 1/sqrt(head_dim)).
	// If 0, it defaults to 1/sqrt(head_dim), where head_dim is the last dimension of query.
	Scale float64

	// Causal applies a causal (lower-triangular) attention mask.
	// Mutually exclusive with Mask.
	Causal bool

	// Mask is an optional boolean or additive float mask broadcastable to the score shape
	// ([batch, numHeads, q_seq, kv_seq] for LayoutBHSD, or [batch, q_seq, numHeads, kv_seq] for LayoutBSHD).
	// Mutually exclusive with Causal and QuerySeqLen/KVSeqLen.
	Mask *Node

	// Bias is an optional additive attention-score bias [batch, numHeads, q_seq, kv_seq]
	// added to scores before softmax.
	Bias *Node

	// QuerySeqLen and KVSeqLen are optional per-batch actual sequence lengths (int32 [B]).
	// Only supported with compute.AttentionAxesLayoutBSHD; mutually exclusive with Mask.
	QuerySeqLen *Node
	KVSeqLen    *Node

	// QuantizedMatmuls indicates quantized matmul semantics when supported by the backend.
	QuantizedMatmuls bool
}

// FusedScaledDotProductAttention computes scaled dot-product attention using the backend's
// fused op when available, falling back automatically to decomposed graph operations otherwise.
//
// It does not depend on model.Store. For higher-level attention with dropout, score soft-capping,
// or returning attention coefficients, use package ml/layers/attention.
func FusedScaledDotProductAttention(
	query, key, value *Node,
	axesLayout compute.AttentionAxesLayout,
	configs ...*FusedSDPAConfig,
) *Node {
	if len(configs) > 1 {
		Panicf("nn.FusedScaledDotProductAttention: expected at most one FusedSDPAConfig, got %d", len(configs))
	}
	var cfg FusedSDPAConfig
	if len(configs) == 1 && configs[0] != nil {
		cfg = *configs[0]
	}

	if cfg.Causal && cfg.Mask != nil {
		Panicf("nn.FusedScaledDotProductAttention: Causal and Mask are mutually exclusive")
	}
	hasSeqLens := cfg.QuerySeqLen != nil || cfg.KVSeqLen != nil
	if hasSeqLens && cfg.Mask != nil {
		Panicf("nn.FusedScaledDotProductAttention: QuerySeqLen/KVSeqLen are mutually exclusive with a non-nil Mask")
	}
	if hasSeqLens && axesLayout != compute.AttentionAxesLayoutBSHD {
		Panicf("nn.FusedScaledDotProductAttention: QuerySeqLen/KVSeqLen require LayoutBSHD, got %s", axesLayout)
	}

	numQueryHeads := query.Shape().Dimensions[axesLayout.HeadsAxis()]
	numKVHeads := key.Shape().Dimensions[axesLayout.HeadsAxis()]
	if numQueryHeads <= 0 || numKVHeads <= 0 || numQueryHeads%numKVHeads != 0 {
		Panicf("nn.FusedScaledDotProductAttention: numQueryHeads (%d) must be positive and divisible by numKVHeads (%d)",
			numQueryHeads, numKVHeads)
	}

	scale := cfg.Scale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(query.Shape().Dimensions[query.Rank()-1]))
	}

	decomposed := func() *Node {
		return sdpaDecomposed(query, key, value, axesLayout, scale, cfg)
	}

	capabilities := query.Graph().Backend().Capabilities()
	hasFused := capabilities.Operations[compute.OpTypeFusedScaledDotProductAttention]
	if !hasFused {
		return decomposed()
	}

	hasFusedVJP := capabilities.Operations[compute.OpTypeFusedScaledDotProductAttentionVJP]
	output, isFused := graphbridge.FusedOpCaller(
		func() *Node {
			fusedConfig := &compute.ScaledDotProductAttentionConfig{
				Scale:            scale,
				Causal:           cfg.Causal,
				QuantizedMatmuls: cfg.QuantizedMatmuls,
			}
			if cfg.Mask != nil {
				fusedConfig.Mask = graphbridge.BackendOutputs(cfg.Mask)[0]
			}
			if cfg.QuerySeqLen != nil {
				fusedConfig.QuerySeqLen = graphbridge.BackendOutputs(cfg.QuerySeqLen)[0]
			}
			if cfg.KVSeqLen != nil {
				fusedConfig.KeyValueSeqLen = graphbridge.BackendOutputs(cfg.KVSeqLen)[0]
			}
			if cfg.Bias != nil {
				fusedConfig.Bias = graphbridge.BackendOutputs(cfg.Bias)[0]
			}
			klog.V(2).Info("nn.FusedScaledDotProductAttention(): attempting to use FusedScaledDotProductAttention op.")
			out, _ := graphbridge.FusedScaledDotProductAttention(query, key, value, axesLayout, fusedConfig)
			return out
		},
		decomposed,
		!hasFusedVJP,
	)
	if isFused {
		klog.V(2).Info("nn.FusedScaledDotProductAttention(): using FusedScaledDotProductAttention op.")
	} else {
		klog.V(2).Info("nn.FusedScaledDotProductAttention(): using decomposed op.")
	}
	return output
}

func sdpaDecomposed(
	query, key, value *Node,
	layout compute.AttentionAxesLayout,
	scale float64,
	cfg FusedSDPAConfig,
) *Node {
	g := query.Graph()
	numQueryHeads := query.Shape().Dimensions[layout.HeadsAxis()]
	numKVHeads := key.Shape().Dimensions[layout.HeadsAxis()]

	decomposedMask := cfg.Mask
	if cfg.Causal {
		seqAxis := layout.SeqAxis()
		qIdx := ExpandDims(DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(query, seqAxis)), -1)
		kIdx := ExpandDims(DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(key, seqAxis)), 0)
		causalBool := GreaterOrEqual(qIdx, kIdx)
		switch layout {
		case compute.AttentionAxesLayoutBHSD:
			causalBool = ExpandAxes(causalBool, 0, 1)
		default:
			causalBool = ExpandAxes(causalBool, 0, 2)
		}
		decomposedMask = causalBool
	}

	if cfg.QuerySeqLen != nil || cfg.KVSeqLen != nil {
		padMask := sdpaSeqLenPaddingMask(query, key, cfg.QuerySeqLen, cfg.KVSeqLen)
		if decomposedMask != nil {
			decomposedMask = LogicalAnd(decomposedMask, padMask)
		} else {
			decomposedMask = padMask
		}
	}

	isGQA := numQueryHeads != numKVHeads
	decomposedQuery := query
	decomposedBias := cfg.Bias
	if decomposedBias != nil && layout == compute.AttentionAxesLayoutBSHD {
		decomposedBias = TransposeAllAxes(decomposedBias, 0, 2, 1, 3)
	}
	if decomposedMask != nil && decomposedMask.Rank() < 4 {
		decomposedMask = sdpaExpandMaskTo4D(decomposedMask, layout)
	}
	if isGQA {
		decomposedQuery = sdpaReshapeQueryForGQA(query, numQueryHeads, numKVHeads, layout)
		if decomposedMask != nil {
			decomposedMask = sdpaReshapeMaskForGQA(decomposedMask, numQueryHeads, numKVHeads, layout)
		}
		if decomposedBias != nil {
			decomposedBias = sdpaReshapeMaskForGQA(decomposedBias, numQueryHeads, numKVHeads, layout)
		}
	}

	scores := Einsum(sdpaScoreEquation(layout, isGQA), decomposedQuery, key)
	scores = MulScalar(scores, scale)

	if decomposedBias != nil {
		scores = Add(scores, decomposedBias)
	}

	var coefficients *Node
	if decomposedMask != nil && decomposedMask.DType() != dtypes.Bool {
		scores = Add(scores, decomposedMask)
		coefficients = Softmax(scores, -1)
	} else {
		coefficients = MaskedSoftmax(scores, decomposedMask, -1)
	}

	decomposedOutput := Einsum(sdpaOutputEquation(layout, isGQA), coefficients, value)
	if isGQA {
		headsAxis := layout.HeadsAxis()
		specs := DimensionSpecsFor(decomposedOutput)
		specs = slices.Delete(specs, headsAxis+1, headsAxis+2)
		specs[headsAxis] = StaticDim(numQueryHeads)
		decomposedOutput = DynamicReshape(decomposedOutput, specs...)
	}
	return decomposedOutput
}

func sdpaScoreEquation(l compute.AttentionAxesLayout, isGQA bool) string {
	if isGQA {
		switch l {
		case compute.AttentionAxesLayoutBSHD:
			return "bqhgd,bkhd->bqhgk"
		default:
			return "bhgqd,bhkd->bhgqk"
		}
	}
	switch l {
	case compute.AttentionAxesLayoutBSHD:
		return "bqhd,bkhd->bqhk"
	default:
		return "bhqd,bhkd->bhqk"
	}
}

func sdpaOutputEquation(l compute.AttentionAxesLayout, isGQA bool) string {
	if isGQA {
		switch l {
		case compute.AttentionAxesLayoutBSHD:
			return "bqhgk,bkhd->bqhgd"
		default:
			return "bhgqk,bhkd->bhgqd"
		}
	}
	switch l {
	case compute.AttentionAxesLayoutBSHD:
		return "bqhk,bkhd->bqhd"
	default:
		return "bhqk,bhkd->bhqd"
	}
}

func sdpaExpandMaskTo4D(mask *Node, layout compute.AttentionAxesLayout) *Node {
	for mask.Rank() < 4 {
		switch mask.Rank() {
		case 1:
			mask = ExpandAxes(mask, 0)
		case 2:
			mask = ExpandAxes(mask, 1, 2)
		case 3:
			mask = ExpandAxes(mask, layout.HeadsAxis())
		}
	}
	return mask
}

func sdpaReshapeQueryForGQA(query *Node, numQueryHeads, numKVHeads int, layout compute.AttentionAxesLayout) *Node {
	groupSize := numQueryHeads / numKVHeads
	headsAxis := layout.HeadsAxis()
	specs := DimensionSpecsFor(query)
	specs = slices.Insert(specs, headsAxis+1, StaticDim(groupSize))
	specs[headsAxis] = StaticDim(numKVHeads)
	return DynamicReshape(query, specs...)
}

func sdpaReshapeMaskForGQA(mask *Node, numQueryHeads, numKVHeads int, layout compute.AttentionAxesLayout) *Node {
	headsAxis := layout.HeadsAxis()
	maskHeadDim := mask.Shape().Dimensions[headsAxis]
	if maskHeadDim == numQueryHeads {
		groupSize := numQueryHeads / numKVHeads
		specs := DimensionSpecsFor(mask)
		specs = slices.Insert(specs, headsAxis+1, StaticDim(groupSize))
		specs[headsAxis] = StaticDim(numKVHeads)
		return DynamicReshape(mask, specs...)
	}
	return InsertAxes(mask, headsAxis+1)
}

func sdpaSeqLenPaddingMask(query, key, querySeqLen, keyValueSeqLen *Node) *Node {
	g := query.Graph()
	var kvMask, qMask *Node
	if keyValueSeqLen != nil {
		kvIota := DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(key, 1))
		kvIota = ExpandAxes(kvIota, 0, 1, 2)
		kvLenBC := ExpandDims(ConvertDType(keyValueSeqLen, dtypes.Int32), 1, 2, 3)
		kvMask = LessThan(kvIota, kvLenBC)
	}
	if querySeqLen != nil {
		qIota := DynamicIota(g, dtypes.Int32, 0, DimensionSpecFor(query, 1))
		qIota = ExpandAxes(qIota, 0, 2, 3)
		qLenBC := ExpandDims(ConvertDType(querySeqLen, dtypes.Int32), 1, 2, 3)
		qMask = LessThan(qIota, qLenBC)
	}
	switch {
	case kvMask != nil && qMask != nil:
		return LogicalAnd(qMask, kvMask)
	case kvMask != nil:
		return kvMask
	default:
		return qMask
	}
}
