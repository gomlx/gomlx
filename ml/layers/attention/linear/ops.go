// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package linear

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
)

// CausalDepthwiseConv1D computes a causal depthwise 1D convolution on channels-last input x
// shaped [B, S, C] with kernel weightKC shaped [K, C] and an optional previous convolution state
// convState shaped [B, K-1, C].
//
// It supports both static and dynamic batch/sequence dimensions across all backends:
//   - For static S == 1 (autoregressive decode), it computes the K-tap dot product directly without slicing.
//   - For static S > 1 or dynamic S, it sums K shifted taps directly (avoiding generic grouped-conv overhead).
//
// Returns (out [B, S, C], newConvState [B, K-1, C] or nil when returnState is false or K <= 1).
func CausalDepthwiseConv1D(x, convState, weightKC *Node, kernelSize int, returnState bool) (out, newConvState *Node) {
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

	// Construct left-padded sequence [B, padLen + S, C].
	var padded *Node
	if convState != nil {
		padded = Concatenate([]*Node{convState, x}, 1)
	} else if padLen > 0 {
		zeros := DynamicBroadcastInDim(ScalarZero(g, origDType), nil, batchSpec, StaticDim(padLen), StaticDim(convDim))
		padded = Concatenate([]*Node{zeros, x}, 1)
	} else {
		padded = x
	}

	// Extract updated conv state: the last padLen tokens of padded, shaped [B, padLen, C].
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

// RMSNormGated applies one-centered RMS normalization along the last axis of x followed by
// multiplicative SiLU gating with z (matching Qwen3_5RMSNormGated / FLA FusedRMSNormGated):
//
//	normed = x_f32 * rsqrt(mean(x_f32², -1) + eps)
//	out    = (weight_f32 * normed * silu(z_f32)).to(origDType)
//
// Weight has shape [D] (one-centered, i.e. NO +1.0 offset), and x and z have shape [..., D].
func RMSNormGated(x, z, weight *Node, eps float64) *Node {
	origDType := x.DType()
	xF32 := ConvertDType(x, dtypes.Float32)
	variance := ReduceAndKeep(Square(xF32), ReduceMean, -1)
	normed := Mul(xF32, Rsqrt(AddScalar(variance, eps)))
	if origDType != dtypes.Float32 {
		normed = ConvertDType(ConvertDType(normed, origDType), dtypes.Float32)
	}
	dim := x.Shape().Dim(-1)
	wDims := make([]int, x.Rank())
	for i := range x.Rank() - 1 {
		wDims[i] = 1
	}
	wDims[x.Rank()-1] = dim
	w := Reshape(ConvertDType(weight, dtypes.Float32), wDims...)
	gate := activation.Apply(activation.TypeSilu, ConvertDType(z, dtypes.Float32))
	return ConvertDType(Mul(Mul(normed, w), gate), origDType)
}

// L2NormLastAxis normalizes x along its last axis: x * rsqrt(sum(x², -1) + eps).
func L2NormLastAxis(x *Node, eps float64) *Node {
	sqSum := ReduceAndKeep(Square(x), ReduceSum, -1)
	return Mul(x, Rsqrt(AddScalar(sqSum, eps)))
}

// RepeatInterleaveHeads repeats each head in x [B, S, numKHeads, headDim] consecutively
// `repeats` times along axis 2, returning [B, S, numKHeads*repeats, headDim].
func RepeatInterleaveHeads(x *Node, repeats int) *Node {
	if repeats == 1 {
		return x
	}
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
