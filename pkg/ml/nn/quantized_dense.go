// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

// nf4LookupValues contains the 16 fixed QLoRA NF4 dequantization values.
var nf4LookupValues = [16]float32{
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
	-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0,
	0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
	0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
}

// QuantizedDense performs a quantized dense (linear) transformation with optional activation:
//
//	y = activation(x @ dequant(packedWeights, scales) + bias)
//
// The packed weights are dequantized on the fly using the specified quantization format
// and per-group scale factors, then multiplied with x.
//
// Parameters:
//   - x: [batch..., K] float32 input activations.
//   - packedWeights: [K, N/2] uint8 for NF4/Int4 (two values per byte, low nibble first),
//     or [K, N] int8 for Int8.
//   - scales: [K, numGroups] float32, where numGroups = ceil(N / groupSize).
//   - bias: [N] float32 (nil means no bias).
//   - quantFormat: NF4, Int4, or Int8.
//   - groupSize: number of output columns sharing a single scale factor.
//   - outFeatures: the N dimension (number of output columns).
//   - activation: optional; if omitted or activations.TypeNone, no activation is applied.
//
// If the backend supports fused QuantizedDense, the optimized native implementation is
// used; otherwise the operation is decomposed into primitives. Fallback is handled
// automatically via InternalFusedOpCaller.
func QuantizedDense(x, packedWeights, scales, bias *Node,
	quantFormat backends.QuantFormat, groupSize, outFeatures int,
	activation ...activations.Type) *Node {

	act := activations.TypeNone
	if len(activation) > 0 {
		if len(activation) > 1 {
			Panicf("nn.QuantizedDense() can only take one optional activation, got %v", activation)
		}
		act = activation[0]
	}

	decomposed := func() *Node {
		return quantizedDenseDecomposed(x, packedWeights, scales, bias, quantFormat, groupSize, outFeatures, act)
	}

	backendAct := act.ToBackend()
	return InternalFusedOpCaller(
		func() *Node {
			return BackendFusedQuantizedDense(x, packedWeights, scales, bias, quantFormat, groupSize, outFeatures, backendAct)
		},
		decomposed,
	)
}

// quantizedDenseDecomposed implements QuantizedDense using primitive graph ops.
func quantizedDenseDecomposed(x, packedWeights, scales, bias *Node,
	quantFormat backends.QuantFormat, groupSize, outFeatures int,
	act activations.Type) *Node {

	g := x.Graph()
	xShape := x.Shape()
	K := xShape.Dimensions[xShape.Rank()-1]
	N := outFeatures

	// Step A: Dequantize packed weights to [K, N] float32.
	var dequant *Node
	switch quantFormat {
	case backends.QuantNF4:
		dequant = dequantNF4(g, packedWeights, K, N)
	case backends.QuantInt4:
		dequant = dequantInt4(g, packedWeights, K, N)
	case backends.QuantInt8:
		dequant = ConvertDType(packedWeights, dtypes.Float32)
	default:
		Panicf("nn.QuantizedDense: unsupported quantization format %v", quantFormat)
	}

	// Step B: Expand scales from [K, numGroups] to [K, N].
	groupIdxSlice := make([]int32, N)
	for j := range N {
		groupIdxSlice[j] = int32(j / groupSize)
	}
	groupIdx := Const(g, groupIdxSlice)                  // [N] int32
	scalesT := Transpose(scales, 0, 1)                   // [numGroups, K]
	groupIdxForGather := Reshape(groupIdx, N, 1)          // [N, 1]
	expandedScales := Gather(scalesT, groupIdxForGather)  // [N, K]
	expandedScales = Transpose(expandedScales, 0, 1)      // [K, N]

	// Step C: Apply scales.
	dequant = Mul(dequant, expandedScales) // [K, N]

	// Step D: Flatten x to 2D [M, K] if needed, then matmul.
	M := xShape.Size() / K
	x2d := x
	if xShape.Rank() > 2 {
		x2d = Reshape(x, M, K)
	}

	y := Dot(x2d, dequant) // [M, N]

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

// extractNibbles extracts low and high nibbles from packed [K, N/2] uint8,
// interleaving them into [K, N] of the specified dtype.
// Low nibble = even column, high nibble = odd column.
func extractNibbles(g *Graph, packed *Node, K, N int, targetDType dtypes.DType) *Node {
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	low := BitwiseAnd(packed, mask)                                            // [K, N/2]
	high := BitwiseAnd(BitwiseShiftRightLogicalScalar(packed, 4), mask)        // [K, N/2]
	indices := Stack([]*Node{low, high}, -1)                                   // [K, N/2, 2]
	indices = Reshape(indices, K, N)                                           // [K, N]
	return ConvertDType(indices, targetDType)
}

// dequantNF4 dequantizes NF4-packed weights to [K, N] float32 using the QLoRA lookup table.
func dequantNF4(g *Graph, packed *Node, K, N int) *Node {
	indices := extractNibbles(g, packed, K, N, dtypes.Int32)    // [K, N] int32
	nf4Table := Const(g, nf4LookupValues[:])                   // [16] float32
	indicesForGather := Reshape(indices, K, N, 1)               // [K, N, 1]
	return Gather(nf4Table, indicesForGather)                   // [K, N] float32
}

// dequantInt4 dequantizes Int4-packed weights to [K, N] float32.
// Each nibble is mapped to the signed range [-8, 7] via (nibble - 8).
func dequantInt4(g *Graph, packed *Node, K, N int) *Node {
	nibbles := extractNibbles(g, packed, K, N, dtypes.Float32)  // [K, N] float32
	return SubScalar(nibbles, float32(8))                       // [K, N] float32
}
