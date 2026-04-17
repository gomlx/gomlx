// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/ggml"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// QuantizedDense performs a quantized dense (linear) transformation with optional activation:
//
//	y = activation(x @ dequant(weights, quant) + bias)
//
// The weights are dequantized on the fly using the quantization metadata, then multiplied with x.
//
// Parameters:
//   - x: [batch..., K] float32 input activations.
//   - weights: [K, N] with dtype reflecting storage type (e.g. Int4, Int8).
//     For sub-byte types, Bitcast packed uint8 data to the correct dtype first.
//   - quant: quantization metadata (scheme, scale, zeroPoint, blockAxis, blockSize).
//   - bias: [N] float32 (nil means no bias).
//   - activation: optional; if omitted or activations.TypeNone, no activation is applied.
//
// If the backend supports fused QuantizedDense, the optimized native implementation is
// used; otherwise the operation is decomposed into primitives. Fallback is handled
// automatically via InternalFusedOpCaller.
func QuantizedDense(x, weights *Node, quant *Quantization, bias *Node,
	activation ...activations.Type) *Node {

	act := activations.TypeNone
	if len(activation) > 0 {
		if len(activation) > 1 {
			Panicf("nn.QuantizedDense() can only take one optional activation, got %v", activation)
		}
		act = activation[0]
	}

	backendAct := act.ToBackend()

	// GGML weights: types with a decomposed graph-level fallback (Q4_0, Q8_0, IQ4_NL)
	// work on any backend (including XLA). K-quant types (Q4_K, Q6_K, etc.) require
	// a backend with a fused executor (e.g. simplego); other backends will panic here.
	if quant.Scheme == backends.QuantGGML {
		if ggml.CanDecompose(quant.GGMLType) {
			return InternalFusedOpCaller(
				func() *Node { return BackendFusedQuantizedDense(x, weights, bias, quant, backendAct) },
				func() *Node { return ggml.DenseDecomposed(x, weights, quant.GGMLType, bias, act) },
			)
		}
		return BackendFusedQuantizedDense(x, weights, bias, quant, backendAct)
	}

	decomposed := func() *Node {
		return quantizedDenseDecomposed(x, weights, quant, bias, act)
	}

	return InternalFusedOpCaller(
		func() *Node {
			return BackendFusedQuantizedDense(x, weights, bias, quant, backendAct)
		},
		decomposed,
	)
}

// quantizedDenseDecomposed implements QuantizedDense using primitive graph ops.
// Weights have their dtype set to the actual storage type (Int4, Int8, etc.).
func quantizedDenseDecomposed(x, weights *Node, quant *Quantization, bias *Node,
	act activations.Type) *Node {

	// Only blockAxis=1 (output-features axis) is currently supported.
	if quant.BlockAxis != 1 {
		Panicf("nn.QuantizedDense: only BlockAxis=1 is supported, got %d", quant.BlockAxis)
	}

	// NF4 quantization uses a fixed lookup table and does not support zero points.
	if quant.Scheme == backends.QuantNF4 && quant.ZeroPoint != nil {
		Panicf("nn.QuantizedDense: ZeroPoint must be nil for NF4 quantization scheme")
	}

	g := x.Graph()
	xShape := x.Shape()
	K := xShape.Dimensions[xShape.Rank()-1]
	wShape := weights.Shape()
	N := wShape.Dimensions[1]

	// Step A: Dequantize weights to [K, N] float32.
	var dequant *Node
	switch quant.Scheme {
	case backends.QuantNF4:
		dequant = dequantNF4FromTyped(g, weights, K, N)
	case backends.QuantLinear:
		dequant = ConvertDType(weights, dtypes.Float32) // [K, N] float32
	default:
		Panicf("nn.QuantizedDense: unsupported quantization scheme %v", quant.Scheme)
	}

	// Step B: Expand scales from [K, numBlocks] to [K, N].
	blockSize := quant.BlockSize
	groupIdxSlice := make([]int32, N)
	for j := range N {
		groupIdxSlice[j] = int32(j / blockSize)
	}
	groupIdx := Const(g, groupIdxSlice)                  // [N] int32
	scalesT := Transpose(quant.Scale, 0, 1)              // [numBlocks, K]
	groupIdxForGather := Reshape(groupIdx, N, 1)         // [N, 1]
	expandedScales := Gather(scalesT, groupIdxForGather) // [N, K]
	expandedScales = Transpose(expandedScales, 0, 1)     // [K, N]

	// Step C: Apply scales.
	dequant = Mul(dequant, expandedScales) // [K, N]

	// Step C2: Apply zero points if present.
	if quant.ZeroPoint != nil {
		zpT := Transpose(quant.ZeroPoint, 0, 1)      // [numBlocks, K]
		expandedZP := Gather(zpT, groupIdxForGather) // [N, K]
		expandedZP = Transpose(expandedZP, 0, 1)     // [K, N]
		dequant = Add(dequant, expandedZP)
	}

	// Step D: Flatten x to 2D [M, K] if needed, then matmul.
	M := xShape.Size() / K
	x2d := x
	if xShape.Rank() > 2 {
		x2d = Reshape(x, M, K)
	}

	y := Dot(x2d, dequant).Product() // [M, N]

	// Reshape output back if x had more than 2 dimensions.
	if xShape.Rank() > 2 {
		outDims := make([]int, xShape.Rank())
		copy(outDims, xShape.Dimensions[:xShape.Rank()-1])
		outDims[xShape.Rank()-1] = N
		y = Reshape(y, outDims...)
	}

	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	if act != activations.TypeNone {
		y = activations.Apply(act, y)
	}
	return y
}

// dequantNF4FromTyped dequantizes NF4 weights (Int4/Uint4, already unpacked) to [K, N] float32
// using the QLoRA lookup table. The nibble indices are in [0..15].
func dequantNF4FromTyped(g *Graph, weights *Node, K, N int) *Node {
	// Convert to int32 indices for Gather. For Int4 (sign-extended [-8..7]),
	// mask to [0..15] first by converting via Uint8 then Int32.
	wDType := weights.Shape().DType
	var indices *Node
	switch wDType {
	case dtypes.Uint4:
		indices = ConvertDType(weights, dtypes.Int32) // [K, N] int32, values [0..15]
	case dtypes.Int4:
		// Int4 values are sign-extended to int8 [-8..7]. To recover the original nibble [0..15],
		// convert to int32 and add 16 to negative values (equivalent to masking with 0x0F).
		indices = ConvertDType(weights, dtypes.Int32)
		// BitwiseAnd with 0x0F to get unsigned nibble [0..15].
		mask := Scalar(g, dtypes.Int32, int32(0x0F))
		indices = BitwiseAnd(indices, mask)
	default:
		Panicf("dequantNF4FromTyped: expected Int4 or Uint4 weights, got %s", wDType)
	}
	nf4Table := Const(g, backends.NF4LookupTable[:]) // [16] float32
	indicesForGather := Reshape(indices, K, N, 1)    // [K, N, 1]
	return Gather(nf4Table, indicesForGather)        // [K, N] float32
}
