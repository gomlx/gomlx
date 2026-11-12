// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ggml

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/exceptions"
)

// ActivationQuantTypeFor returns the paired GGML activation quantization format
// (GGMLQ8_0 for 32-value weight formats such as Q4_0/Q8_0/IQ4_NL; GGMLQ8_K for 256-value
// K-quant weight formats Q2_K..Q8_K) used for block-integer dot products.
func ActivationQuantTypeFor(weightType compute.GGMLQuantType) (compute.GGMLQuantType, bool) {
	switch weightType.ValuesPerBlock() {
	case 32:
		return compute.GGMLQ8_0, true
	case 256:
		return compute.GGMLQ8_K, true
	default:
		return 0, false
	}
}

// Quantize quantizes floating-point activations x of shape [batch..., K] into packed Uint8
// GGML blocks of shape [batch..., bytesPerRow] for GGMLQ8_0 (34 bytes per 32 values) or
// GGMLQ8_K (292 bytes per 256 values).
//
// The returned node can be shared across multiple nn.QuantizedDense projections (e.g. Q, K, V
// or Gate, Up in a GGUF Transformer block) by setting:
//
//	XQuantization: &graph.Quantization{Scheme: compute.QuantGGML, GGMLType: ggmlType}
func Quantize(x *Node, ggmlType compute.GGMLQuantType) *Node {
	xShape := x.Shape()
	if !xShape.DType.IsFloat() {
		exceptions.Panicf("ggml.Quantize: x must have floating-point DType, got %s", xShape)
	}
	if xShape.Rank() < 1 {
		exceptions.Panicf("ggml.Quantize: x must have rank >= 1, got %s", xShape)
	}
	K := xShape.Dimensions[xShape.Rank()-1]
	if K == shapes.DynamicDim {
		exceptions.Panicf("ggml.Quantize: x's last dimension K cannot be dynamic, got %s", xShape)
	}

	vpb := ggmlType.ValuesPerBlock()
	bpb := ggmlType.BytesPerBlock()
	if (ggmlType != compute.GGMLQ8_0 && ggmlType != compute.GGMLQ8_K) || vpb <= 0 || bpb <= 0 {
		exceptions.Panicf("ggml.Quantize: only GGMLQ8_0 and GGMLQ8_K are supported for row quantization, got %s", ggmlType)
	}
	if K <= 0 || K%vpb != 0 {
		exceptions.Panicf("ggml.Quantize: last dimension K (%d) must be positive and divisible by %d for %s", K, vpb, ggmlType)
	}

	numBlocks := K / vpb
	bytesPerRow := numBlocks * bpb
	x32 := x
	if x32.DType() != dtypes.Float32 {
		x32 = ConvertDType(x32, dtypes.Float32)
	}

	var packed3D *Node // [M, numBlocks, bpb] Uint8
	if xShape.IsDynamic() {
		g := x.Graph()
		blocks := DynamicReshape(x32, NamedInferredDim(g.UniqueName("ggml_q_rows")), StaticDim(numBlocks), StaticDim(vpb))
		switch ggmlType {
		case compute.GGMLQ8_0:
			packed3D = quantizeBlocksQ8_0(blocks, numBlocks, true)
		case compute.GGMLQ8_K:
			packed3D = quantizeBlocksQ8_K(blocks, numBlocks, true)
		}
		specs := DimensionSpecsFor(x)
		specs[len(specs)-1] = StaticDim(bytesPerRow)
		return DynamicReshape(packed3D, specs...)
	}

	M := xShape.Size() / K
	blocks := Reshape(x32, M, numBlocks, vpb)
	switch ggmlType {
	case compute.GGMLQ8_0:
		packed3D = quantizeBlocksQ8_0(blocks, numBlocks, false)
	case compute.GGMLQ8_K:
		packed3D = quantizeBlocksQ8_K(blocks, numBlocks, false)
	}

	outDims := slices.Clone(xShape.Dimensions)
	outDims[len(outDims)-1] = bytesPerRow
	return Reshape(packed3D, outDims...)
}

func quantizeBlocksQ8_0(blocks *Node, numBlocks int, dynamicM bool) *Node {
	g := blocks.Graph()
	maxAbs := ReduceAndKeep(Abs(blocks), ReduceMax, -1) // [M, numBlocks, 1]
	dF16 := ConvertDType(DivScalar(maxAbs, float32(127.0)), dtypes.Float16)
	dActual := ConvertDType(dF16, dtypes.Float32)

	zero := ZerosLike(dActual)
	posMask := GreaterThan(dActual, zero)
	safeD := Where(posMask, dActual, OnesLike(dActual))
	invD := Where(posMask, Reciprocal(safeD), zero)

	qRounded := roundToEvenF32(Mul(blocks, invD))
	qClamped := Clamp(Scalar(g, dtypes.Float32, float32(-127.0)), qRounded, Scalar(g, dtypes.Float32, float32(127.0)))
	qBytes := Bitcast(ConvertDType(qClamped, dtypes.Int8), dtypes.Uint8) // [M, numBlocks, 32] Uint8

	var dBytes *Node
	if dynamicM {
		d2 := DynamicReshape(dF16, NamedInferredDim(g.UniqueName("ggml_q8_0_d")), StaticDim(numBlocks))
		dBytes = Bitcast(d2, dtypes.Uint8) // [M, numBlocks, 2] Uint8
	} else {
		M := blocks.Shape().Dimensions[0]
		dBytes = Reshape(Bitcast(Reshape(dF16, M, numBlocks), dtypes.Uint8), M, numBlocks, 2)
	}
	return Concatenate([]*Node{dBytes, qBytes}, 2) // [M, numBlocks, 34] Uint8
}

func quantizeBlocksQ8_K(blocks *Node, numBlocks int, dynamicM bool) *Node {
	g := blocks.Graph()
	maxAbs := ReduceAndKeep(Abs(blocks), ReduceMax, -1) // [M, numBlocks, 1]
	d := DivScalar(maxAbs, float32(127.0))

	zero := ZerosLike(maxAbs)
	posMask := GreaterThan(maxAbs, zero)
	safeMax := Where(posMask, maxAbs, OnesLike(maxAbs))
	invD := Where(posMask, Div(Scalar(g, dtypes.Float32, float32(127.0)), safeMax), zero)

	qRounded := roundToEvenF32(Mul(blocks, invD))
	qClamped := Clamp(Scalar(g, dtypes.Float32, float32(-127.0)), qRounded, Scalar(g, dtypes.Float32, float32(127.0)))
	qInt8 := ConvertDType(qClamped, dtypes.Int8) // [M, numBlocks, 256] Int8
	qBytes := Bitcast(qInt8, dtypes.Uint8)       // [M, numBlocks, 256] Uint8

	var dBytes, bsumBytes *Node
	if dynamicM {
		d2 := DynamicReshape(d, NamedInferredDim(g.UniqueName("ggml_q8_k_d")), StaticDim(numBlocks))
		dBytes = Bitcast(d2, dtypes.Uint8) // [M, numBlocks, 4] Uint8

		qGroups := DynamicReshape(ConvertDType(qInt8, dtypes.Int16),
			NamedInferredDim(g.UniqueName("ggml_q8_k_g")), StaticDim(numBlocks), StaticDim(16), StaticDim(16))
		bsums := ReduceSum(qGroups, -1) // [M, numBlocks, 16] Int16
		bsumRaw := Bitcast(bsums, dtypes.Uint8)
		bsumBytes = DynamicReshape(bsumRaw, NamedInferredDim(g.UniqueName("ggml_q8_k_b")), StaticDim(numBlocks), StaticDim(32))
	} else {
		M := blocks.Shape().Dimensions[0]
		dBytes = Reshape(Bitcast(Reshape(d, M, numBlocks), dtypes.Uint8), M, numBlocks, 4)

		qGroups := Reshape(ConvertDType(qInt8, dtypes.Int16), M, numBlocks, 16, 16)
		bsums := ReduceSum(qGroups, -1) // [M, numBlocks, 16] Int16
		bsumBytes = Reshape(Bitcast(bsums, dtypes.Uint8), M, numBlocks, 32)
	}
	return Concatenate([]*Node{dBytes, qBytes, bsumBytes}, 2) // [M, numBlocks, 292] Uint8
}

// roundToEvenF32 implements IEEE-754 round-half-to-even in standard primitive graph ops.
func roundToEvenF32(x *Node) *Node {
	g := x.Graph()
	f := Floor(x)
	frac := Sub(x, f)
	half := Scalar(g, dtypes.Float32, float32(0.5))
	two := Scalar(g, dtypes.Float32, float32(2.0))
	one := Scalar(g, dtypes.Float32, float32(1.0))
	gtHalf := GreaterThan(frac, half)
	eqHalf := Equal(frac, half)
	isOdd := Equal(Abs(Rem(f, two)), one)
	return Where(LogicalOr(gtHalf, LogicalAnd(eqHalf, isOdd)), Add(f, one), f)
}
