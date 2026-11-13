// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package ggml provides graph-level decomposed dequantization and row quantization
// for GGML block formats.
//
// These functions express all 12 supported GGML formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
// IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) as standard graph primitives (Bitcast,
// BitwiseAnd, BitwiseShiftRightLogical, Slice, ConvertDType, etc.) so that any backend —
// including XLA for CPU/GPU — can execute them. The "go" backend uses faster fused SIMD
// implementations when available; these decomposed versions serve as the automatic
// fallback via InternalFusedOpCaller.
package ggml

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
)

// CanDecompose returns true for GGML quantization types that have a graph-level
// decomposed implementation (all 12 supported GGML types: Q4_0, Q4_1, Q5_0, Q5_1,
// Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K).
func CanDecompose(t compute.GGMLQuantType) bool {
	switch t {
	case compute.GGMLQ4_0, compute.GGMLQ4_1, compute.GGMLQ5_0, compute.GGMLQ5_1,
		compute.GGMLQ8_0, compute.GGMLIQ4NL,
		compute.GGMLQ2_K, compute.GGMLQ3_K, compute.GGMLQ4_K, compute.GGMLQ5_K, compute.GGMLQ6_K, compute.GGMLQ8_K:
		return true
	default:
		return false
	}
}

// Dequantize dequantizes GGML block-format weights to [N, K] Float32 using graph primitives.
//
// weights is shaped [N, numBlocks * bytesPerRow] Uint8 in native GGML block layout,
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
	case compute.GGMLQ4_1:
		return dequantQ4_1(weights, N, numBlocks)
	case compute.GGMLQ5_0:
		return dequantQ5_0(weights, N, numBlocks)
	case compute.GGMLQ5_1:
		return dequantQ5_1(weights, N, numBlocks)
	case compute.GGMLIQ4NL:
		return dequantIQ4NL(weights, N, numBlocks)
	case compute.GGMLQ2_K:
		return dequantQ2_K(weights, N, numBlocks)
	case compute.GGMLQ3_K:
		return dequantQ3_K(weights, N, numBlocks)
	case compute.GGMLQ4_K:
		return dequantQ4_K(weights, N, numBlocks)
	case compute.GGMLQ5_K:
		return dequantQ5_K(weights, N, numBlocks)
	case compute.GGMLQ6_K:
		return dequantQ6_K(weights, N, numBlocks)
	case compute.GGMLQ8_K:
		return dequantQ8_K(weights, N, numBlocks)
	default:
		panic("ggml.Dequantize: unsupported type " + ggmlType.String())
	}
}

// dequantQ8_K dequantizes Q8_K blocks: 4-byte fp32 scale + 256 int8 quants + 32-byte int16 bsums per 292-byte block.
// Returns [N, K] Float32 where K = numBlocks * 256.
func dequantQ8_K(weights *Node, N, numBlocks int) *Node {
	w := Reshape(weights, N, numBlocks, 292)

	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 4))
	scale := Bitcast(scaleBytes, dtypes.Float32)
	scale = Reshape(scale, N, numBlocks, 1)

	quantBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(4, 260))
	quants := Bitcast(quantBytes, dtypes.Int8)
	quants = ConvertDType(quants, dtypes.Float32)

	result := Mul(quants, scale)
	return Reshape(result, N, numBlocks*256)
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

// dequantQ4_1 dequantizes Q4_1 blocks: 2-byte fp16 d + 2-byte fp16 m + 16 packed nibble bytes per 20-byte block.
// Dequantize: output[i] = d * nibble + m.
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ4_1(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 20)

	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1)

	mBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 4))
	m := Reshape(ConvertDType(Bitcast(mBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1)

	nibbleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(4, 20))
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	lo := BitwiseAnd(nibbleBytes, mask)
	hi := BitwiseShiftRightLogicalScalar(nibbleBytes, uint8(4))
	combinedF := ConvertDType(Concatenate([]*Node{lo, hi}, 2), dtypes.Float32)

	result := Add(Mul(combinedF, d), m)
	return Reshape(result, N, numBlocks*32)
}

// extract5BitQuants extracts 32 unsigned 5-bit quantized values (in 0..31) from a 4-byte high-bit mask qhBytes
// [N, numBlocks, 4] and 16-byte packed low-nibble slice qsBytes [N, numBlocks, 16].
// In GGML Q5_0 and Q5_1, bit k (k = 0..31) of the little-endian uint32 qh (i.e. byte k/8, bit k%8) holds
// bit 4 (value 16) of element k.
func extract5BitQuants(qhBytes, qsBytes *Node, N, numBlocks int) *Node {
	g := qhBytes.Graph()
	mask := Scalar(g, dtypes.Uint8, uint8(0x0F))
	lo := BitwiseAnd(qsBytes, mask)
	hi := BitwiseShiftRightLogicalScalar(qsBytes, uint8(4))
	low4 := Concatenate([]*Node{lo, hi}, 2) // [N, numBlocks, 32] Uint8

	qhExpanded := BroadcastToDims(Reshape(qhBytes, N, numBlocks, 4, 1), N, numBlocks, 4, 8)
	shifts := BroadcastToDims(Iota(g, shapes.Make(dtypes.Uint8, 1, 1, 1, 8), 3), N, numBlocks, 4, 8)
	hiBits := BitwiseAnd(BitwiseShiftRightLogical(qhExpanded, shifts), Scalar(g, dtypes.Uint8, uint8(1)))
	hiBit16 := BitwiseShiftLeftScalar(Reshape(hiBits, N, numBlocks, 32), uint8(4))

	return BitwiseOr(low4, hiBit16)
}

// dequantQ5_0 dequantizes Q5_0 blocks: 2-byte fp16 d + 4-byte qh + 16-byte qs per 22-byte block.
// Dequantize: output[i] = d * (q5 - 16).
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ5_0(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 22)

	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1)

	qhBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 6))
	qsBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(6, 22))
	q5F := ConvertDType(extract5BitQuants(qhBytes, qsBytes, N, numBlocks), dtypes.Float32)

	sixteen := Scalar(g, dtypes.Float32, float32(16.0))
	result := Mul(Sub(q5F, sixteen), d)
	return Reshape(result, N, numBlocks*32)
}

// dequantQ5_1 dequantizes Q5_1 blocks: 2-byte fp16 d + 2-byte fp16 m + 4-byte qh + 16-byte qs per 24-byte block.
// Dequantize: output[i] = d * q5 + m.
// Returns [N, K] Float32 where K = numBlocks * 32.
func dequantQ5_1(weights *Node, N, numBlocks int) *Node {
	w := Reshape(weights, N, numBlocks, 24)

	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1)

	mBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(2, 4))
	m := Reshape(ConvertDType(Bitcast(mBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1)

	qhBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(4, 8))
	qsBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(8, 24))
	q5F := ConvertDType(extract5BitQuants(qhBytes, qsBytes, N, numBlocks), dtypes.Float32)

	result := Add(Mul(q5F, d), m)
	return Reshape(result, N, numBlocks*32)
}

// dequantQ2_K dequantizes Q2_K super-blocks (84 bytes per 256 values):
// scales[16] + qs[64] + fp16 d (2B) + fp16 dmin (2B).
func dequantQ2_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 84)

	scalesBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(0, 16))
	sc := Reshape(ConvertDType(BitwiseAnd(scalesBytes, Scalar(g, dtypes.Uint8, uint8(0x0F))), dtypes.Float32), N, numBlocks, 16, 1)
	mn := Reshape(ConvertDType(BitwiseShiftRightLogicalScalar(scalesBytes, uint8(4)), dtypes.Float32), N, numBlocks, 16, 1)

	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(80, 82))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)
	dminBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(82, 84))
	dmin := Reshape(ConvertDType(Bitcast(dminBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)

	qs := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(16, 80)), N, numBlocks, 2, 1, 32)
	qsExp := BroadcastToDims(qs, N, numBlocks, 2, 4, 32)
	shifts := BroadcastToDims(
		MulScalar(Iota(g, shapes.Make(dtypes.Uint8, 1, 1, 1, 4, 1), 3), uint8(2)),
		N, numBlocks, 2, 4, 32,
	)
	q2 := BitwiseAnd(BitwiseShiftRightLogical(qsExp, shifts), Scalar(g, dtypes.Uint8, uint8(0x03)))
	qF := ConvertDType(Reshape(q2, N, numBlocks, 16, 16), dtypes.Float32)

	result := Sub(Mul(Mul(d, sc), qF), Mul(dmin, mn))
	return Reshape(result, N, numBlocks*256)
}

// dequantQ3_K dequantizes Q3_K super-blocks (110 bytes per 256 values):
// hmask[32] + qs[64] + scales[12] + fp16 d (2B).
func dequantQ3_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 110)

	hmask := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(0, 32)), N, numBlocks, 1, 32)
	qs := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(32, 96)), N, numBlocks, 2, 1, 32)
	scaleBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(96, 108))
	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(108, 110))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)

	// Unpack 16 6-bit scales (signed offset -32) from 12 bytes.
	mask0F := Scalar(g, dtypes.Uint8, uint8(0x0F))
	mask03 := Scalar(g, dtypes.Uint8, uint8(0x03))
	sb0 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(0, 4))
	sb1 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(4, 8))
	sb2 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(8, 12))

	sc0_3 := BitwiseOr(BitwiseAnd(sb0, mask0F), BitwiseShiftLeftScalar(BitwiseAnd(sb2, mask03), uint8(4)))
	sc4_7 := BitwiseOr(BitwiseAnd(sb1, mask0F), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(sb2, uint8(2)), mask03), uint8(4)))
	sc8_11 := BitwiseOr(BitwiseAnd(BitwiseShiftRightLogicalScalar(sb0, uint8(4)), mask0F), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(sb2, uint8(4)), mask03), uint8(4)))
	sc12_15 := BitwiseOr(BitwiseAnd(BitwiseShiftRightLogicalScalar(sb1, uint8(4)), mask0F), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(sb2, uint8(6)), mask03), uint8(4)))
	scalesU6 := Concatenate([]*Node{sc0_3, sc4_7, sc8_11, sc12_15}, 2)
	scalesF := Sub(Reshape(ConvertDType(scalesU6, dtypes.Float32), N, numBlocks, 16, 1), Scalar(g, dtypes.Float32, float32(32.0)))

	// Unpack 256 signed 3-bit values: ((qs >> shift) & 3) - (hbit == 0 ? 4 : 0) = low2 + hbit*4 - 4.
	qsExp := BroadcastToDims(qs, N, numBlocks, 2, 4, 32)
	qShifts := BroadcastToDims(
		MulScalar(Iota(g, shapes.Make(dtypes.Uint8, 1, 1, 1, 4, 1), 3), uint8(2)),
		N, numBlocks, 2, 4, 32,
	)
	low2 := Reshape(BitwiseAnd(BitwiseShiftRightLogical(qsExp, qShifts), mask03), N, numBlocks, 8, 32)

	hExp := BroadcastToDims(hmask, N, numBlocks, 8, 32)
	hShifts := BroadcastToDims(Iota(g, shapes.Make(dtypes.Uint8, 1, 1, 8, 1), 2), N, numBlocks, 8, 32)
	hi4 := BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogical(hExp, hShifts), Scalar(g, dtypes.Uint8, uint8(1))), uint8(2))
	q3U := BitwiseOr(low2, hi4)
	q3F := Sub(Reshape(ConvertDType(q3U, dtypes.Float32), N, numBlocks, 16, 16), Scalar(g, dtypes.Float32, float32(4.0)))

	result := Mul(Mul(d, scalesF), q3F)
	return Reshape(result, N, numBlocks*256)
}

// unpackScaleMinK4 unpacks 8 6-bit sub-block scales and 8 6-bit sub-block mins from
// the 12-byte packed scale header shared by Q4_K and Q5_K.
// Returns sc, mn each shaped [N, numBlocks, 8, 1] Float32.
func unpackScaleMinK4(scaleBytes *Node, N, numBlocks int) (sc, mn *Node) {
	g := scaleBytes.Graph()
	mask63 := Scalar(g, dtypes.Uint8, uint8(63))
	mask0F := Scalar(g, dtypes.Uint8, uint8(0x0F))

	b0_4 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(0, 4))
	b4_8 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(4, 8))
	b8_12 := Slice(scaleBytes, AxisRange(), AxisRange(), AxisRange(8, 12))

	sc0_3 := BitwiseAnd(b0_4, mask63)
	mn0_3 := BitwiseAnd(b4_8, mask63)
	sc4_7 := BitwiseOr(BitwiseAnd(b8_12, mask0F), BitwiseShiftLeftScalar(BitwiseShiftRightLogicalScalar(b0_4, uint8(6)), uint8(4)))
	mn4_7 := BitwiseOr(BitwiseShiftRightLogicalScalar(b8_12, uint8(4)), BitwiseShiftLeftScalar(BitwiseShiftRightLogicalScalar(b4_8, uint8(6)), uint8(4)))

	sc = Reshape(ConvertDType(Concatenate([]*Node{sc0_3, sc4_7}, 2), dtypes.Float32), N, numBlocks, 8, 1)
	mn = Reshape(ConvertDType(Concatenate([]*Node{mn0_3, mn4_7}, 2), dtypes.Float32), N, numBlocks, 8, 1)
	return sc, mn
}

// dequantQ4_K dequantizes Q4_K super-blocks (144 bytes per 256 values):
// fp16 d (2B) + fp16 dmin (2B) + scales[12] + qs[128].
func dequantQ4_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 144)

	d := Reshape(ConvertDType(Bitcast(Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2)), dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)
	dmin := Reshape(ConvertDType(Bitcast(Slice(w, AxisRange(), AxisRange(), AxisRange(2, 4)), dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)
	sc, mn := unpackScaleMinK4(Slice(w, AxisRange(), AxisRange(), AxisRange(4, 16)), N, numBlocks)

	qs := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(16, 144)), N, numBlocks, 4, 1, 32)
	lo := BitwiseAnd(qs, Scalar(g, dtypes.Uint8, uint8(0x0F)))
	hi := BitwiseShiftRightLogicalScalar(qs, uint8(4))
	q4F := ConvertDType(Reshape(Concatenate([]*Node{lo, hi}, 3), N, numBlocks, 8, 32), dtypes.Float32)

	result := Sub(Mul(Mul(d, sc), q4F), Mul(dmin, mn))
	return Reshape(result, N, numBlocks*256)
}

// dequantQ5_K dequantizes Q5_K super-blocks (176 bytes per 256 values):
// fp16 d (2B) + fp16 dmin (2B) + scales[12] + qh[32] + qs[128].
func dequantQ5_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 176)

	d := Reshape(ConvertDType(Bitcast(Slice(w, AxisRange(), AxisRange(), AxisRange(0, 2)), dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)
	dmin := Reshape(ConvertDType(Bitcast(Slice(w, AxisRange(), AxisRange(), AxisRange(2, 4)), dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)
	sc, mn := unpackScaleMinK4(Slice(w, AxisRange(), AxisRange(), AxisRange(4, 16)), N, numBlocks)

	qh := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(16, 48)), N, numBlocks, 1, 32)
	qs := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(48, 176)), N, numBlocks, 4, 1, 32)
	lo := BitwiseAnd(qs, Scalar(g, dtypes.Uint8, uint8(0x0F)))
	hi := BitwiseShiftRightLogicalScalar(qs, uint8(4))
	low4 := Reshape(Concatenate([]*Node{lo, hi}, 3), N, numBlocks, 8, 32)

	qhExp := BroadcastToDims(qh, N, numBlocks, 8, 32)
	shifts := BroadcastToDims(Iota(g, shapes.Make(dtypes.Uint8, 1, 1, 8, 1), 2), N, numBlocks, 8, 32)
	hiBit16 := BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogical(qhExp, shifts), Scalar(g, dtypes.Uint8, uint8(1))), uint8(4))
	q5F := ConvertDType(BitwiseOr(low4, hiBit16), dtypes.Float32)

	result := Sub(Mul(Mul(d, sc), q5F), Mul(dmin, mn))
	return Reshape(result, N, numBlocks*256)
}

// dequantQ6_K dequantizes Q6_K super-blocks (210 bytes per 256 values):
// ql[128] + qh[64] + int8 scales[16] + fp16 d (2B).
func dequantQ6_K(weights *Node, N, numBlocks int) *Node {
	g := weights.Graph()
	w := Reshape(weights, N, numBlocks, 210)

	ql := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(0, 128)), N, numBlocks, 2, 64)
	ql0 := Slice(ql, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, 32))
	ql1 := Slice(ql, AxisRange(), AxisRange(), AxisRange(), AxisRange(32, 64))
	qh := Reshape(Slice(w, AxisRange(), AxisRange(), AxisRange(128, 192)), N, numBlocks, 2, 32)

	scBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(192, 208))
	sc := Reshape(ConvertDType(Bitcast(scBytes, dtypes.Int8), dtypes.Float32), N, numBlocks, 16, 1)
	dBytes := Slice(w, AxisRange(), AxisRange(), AxisRange(208, 210))
	d := Reshape(ConvertDType(Bitcast(dBytes, dtypes.Float16), dtypes.Float32), N, numBlocks, 1, 1)

	mask0F := Scalar(g, dtypes.Uint8, uint8(0x0F))
	mask03 := Scalar(g, dtypes.Uint8, uint8(0x03))
	q1 := BitwiseOr(BitwiseAnd(ql0, mask0F), BitwiseShiftLeftScalar(BitwiseAnd(qh, mask03), uint8(4)))
	q2 := BitwiseOr(BitwiseAnd(ql1, mask0F), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(qh, uint8(2)), mask03), uint8(4)))
	q3 := BitwiseOr(BitwiseShiftRightLogicalScalar(ql0, uint8(4)), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(qh, uint8(4)), mask03), uint8(4)))
	q4 := BitwiseOr(BitwiseShiftRightLogicalScalar(ql1, uint8(4)), BitwiseShiftLeftScalar(BitwiseAnd(BitwiseShiftRightLogicalScalar(qh, uint8(6)), mask03), uint8(4)))

	q6U := Reshape(Concatenate([]*Node{q1, q2, q3, q4}, 3), N, numBlocks, 16, 16)
	q6F := Sub(ConvertDType(q6U, dtypes.Float32), Scalar(g, dtypes.Float32, float32(32.0)))

	result := Mul(Mul(d, sc), q6F)
	return Reshape(result, N, numBlocks*256)
}
