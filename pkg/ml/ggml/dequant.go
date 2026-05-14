// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package ggml provides graph-level decomposed dequantization for GGML block formats.
//
// These functions express GGML dequantization (Q4_0, Q8_0, IQ4_NL) as standard graph
// primitives (Bitcast, BitwiseAnd, ShiftRightLogical, Slice, ConvertDType, etc.) so that
// any backend — including XLA for GPU — can execute them. The simplego backend uses faster
// fused SIMD implementations when available; these decomposed versions serve as the
// automatic fallback via InternalFusedOpCaller.
package ggml

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
)

// CanDecompose returns true for GGML quantization types that have a graph-level
// decomposed implementation (Q4_0, Q8_0, IQ4_NL).
func CanDecompose(t compute.GGMLQuantType) bool {
	switch t {
	case compute.GGMLQ4_0, compute.GGMLQ8_0, compute.GGMLIQ4NL:
		return true
	default:
		return false
	}
}

// Dequantize dequantizes GGML block-format weights to [N, K] Float32 using graph primitives.
//
// weights is shaped [N, numBlocks * bytesPerBlock] Uint8 in native GGML block layout,
// where N is the number of output rows (output features for dense, vocab size for embedding lookup).
// K is the logical feature dimension: K = numBlocks * valuesPerBlock, and is fixed per
// quantization type (e.g. Q4_0 has 32 values per 18-byte block).
//
// N is passed explicitly rather than inferred from weights.Shape() because the caller may
// have reshaped gathered rows into [totalRows, bytesPerRow] where totalRows differs from
// the original table's first dimension.
func Dequantize(weights *Node, ggmlType compute.GGMLQuantType, N int) *Node {
	bytesPerRow := weights.Shape().Dimensions[1]
	bpb := ggmlType.BytesPerBlock()
	numBlocks := bytesPerRow / bpb

	switch ggmlType {
	case compute.GGMLQ8_0:
		return dequantQ8_0(weights, N, numBlocks)
	case compute.GGMLQ4_0:
		return dequantQ4_0(weights, N, numBlocks)
	case compute.GGMLIQ4NL:
		return dequantIQ4NL(weights, N, numBlocks)
	default:
		panic("ggml.Dequantize: unsupported type " + ggmlType.String())
	}
}

// dequantQ8_0 dequantizes Q8_0 blocks: 2-byte fp16 scale + 32 int8 quants per block.
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ8_0(weights *Node, N, numBlocks int) *Node {
	// Reshape to [N, numBlocks, 34] (34 bytes per Q8_0 block).
	w := Reshape(weights, N, numBlocks, 34)

	// Extract scale: first 2 bytes → fp16 → float32 → [N, numBlocks, 1].
	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	scale := Bitcast(scaleBytes, dtypes.Float16)
	scale = ConvertDType(scale, dtypes.Float32)
	scale = Reshape(scale, N, numBlocks, 1) // ensure rank-3 for broadcast

	// Extract quants: bytes 2:34 → int8 → float32 → [N, numBlocks, 32].
	quantBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 34))
	quants := Bitcast(quantBytes, dtypes.Int8)
	quants = ConvertDType(quants, dtypes.Float32)

	// Dequantize: output = scale * quants
	result := Mul(quants, scale)

	K := numBlocks * 32
	return Reshape(result, N, K)
}

// extractNibbleBlock extracts the fp16 scale and 32 combined nibble values from
// an 18-byte block layout (shared by Q4_0 and IQ4_NL).
// Returns scale [N, numBlocks, 1] Float32 and combined [N, numBlocks, 32] Uint8.
func extractNibbleBlock(weights *Node, N, numBlocks int) (scale, combined *Node) {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 18)

	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	scale = Bitcast(scaleBytes, dtypes.Float16)
	scale = ConvertDType(scale, dtypes.Float32)
	scale = Reshape(scale, N, numBlocks, 1)

	nibbleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 18))
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	lo := BitwiseAnd(nibbleBytes, mask)
	hi := BitwiseShiftRightLogicalScalar(nibbleBytes, uint8(4))
	combined = Concatenate([]*Node{lo, hi}, 2)
	return scale, combined
}

// dequantQ4_0 dequantizes Q4_0 blocks: 2-byte fp16 scale + 16 packed nibble bytes per block.
// Each byte holds two 4-bit values: low nibble → first 16 values, high nibble → last 16.
// Dequantize: output[i] = scale * (nibble - 8).
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ4_0(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	scale, combined := extractNibbleBlock(weights, N, numBlocks)

	combinedF := ConvertDType(combined, dtypes.Float32)
	eight := Scalar(g, dtypes.Float32, float32(8.0))
	combinedF = Sub(combinedF, eight)

	result := Mul(combinedF, scale)
	K := numBlocks * 32
	return Reshape(result, N, K)
}

// dequantIQ4NL dequantizes IQ4_NL blocks: same layout as Q4_0 (2-byte fp16 scale + 16 packed
// nibble bytes), but nibble values are indices into a non-linear lookup table instead of
// linear (nibble - 8).
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantIQ4NL(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	scale, combined := extractNibbleBlock(weights, N, numBlocks)

	indices := ConvertDType(combined, dtypes.Int32)
	totalElements := N * numBlocks * 32
	indicesFlat := Reshape(indices, totalElements, 1)

	lut := Const(g, compute.IQ4NLLookupTable[:])
	looked := Gather(lut, indicesFlat)
	looked = Reshape(looked, N, numBlocks, 32)

	result := Mul(looked, scale)
	K := numBlocks * 32
	return Reshape(result, N, K)
}
