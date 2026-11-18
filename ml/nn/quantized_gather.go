// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/support/envutil"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/ml/ggml"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// QuantizedGather performs a quantized row gather (embedding lookup): gathers rows from a
// quantized or packed sub-byte table and unpacks/dequantizes only the selected rows on-the-fly.
//
// This is the quantized analogue of graph.Gather for row lookups (similar to llama.cpp's
// ggml_get_rows) and is also the supported way to gather rows from packed sub-byte tables
// (Int2, Uint2, Int4, Uint4), which graph.Gather does not support directly.
//
// Supported modes:
//   - quant == nil or QuantNone: gathers rows from Int2/Uint2/Int4/Uint4 (or Int8/Uint8) tables
//     and unpacks them losslessly to Int8 (for signed Int2/Int4/Int8) or Uint8 (for unsigned
//     Uint2/Uint4/Uint8). If PackedDType is set, table may be stored bit-packed in Uint8 with
//     shape [vocabSize, K / ValuesPerStorageUnit] and logical dtype PackedDType.
//     Supports both static and dynamic batch dimensions on any backend.
//   - QuantGGML: table is [vocabSize, bytesPerRow] Uint8 with native GGML block layout;
//     output is Float32. Static batch dimensions fall back to ggml.GatherDecomposed;
//     dynamic batch dimensions require a backend with OpTypeQuantizedGather (e.g. "go").
//   - QuantLinear: table is [vocabSize, K] Int2/Uint2/Int4/Uint4/Int8/Uint8 (or [vocabSize, K/vpb]
//     Uint8 when PackedDType is set) with Scale (Float32/Float16/BFloat16) and optional ZeroPoint;
//     output matches Scale's DType. Note that PackedDType does not apply to ZeroPoint.
//     Supports both static and dynamic batch dimensions on any backend.
//   - QuantNF4: table is [vocabSize, K] Int4/Uint4/Int8/Uint8 (or [vocabSize, K/2] Uint8 when
//     PackedDType is set) with Scale (Float32/Float16/BFloat16); output matches Scale's DType.
//
// Parameters:
//   - table: quantized or sub-byte table (rank 2).
//   - indices: integer tensor with last dimension = 1 (same as Gather convention).
//     For embeddings: [batch, seqLen, 1].
//   - quant: quantization metadata, or nil to gather and unpack sub-byte/8-bit rows to Int8/Uint8.
//
// Output: tensor with shape [batch..., K].
func QuantizedGather(table, indices *Node, quant *Quantization) *Node {
	var decomposed func() *Node
	switch {
	case quant == nil, quant.Scheme == compute.QuantNone:
		decomposed = func() *Node {
			return quantizedGatherUnpackedDecomposed(table, indices, quant)
		}
	case quant.Scheme == compute.QuantGGML:
		decomposed = func() *Node {
			return ggml.GatherDecomposed(table, indices, quant.GGMLType)
		}
	case quant.Scheme == compute.QuantLinear, quant.Scheme == compute.QuantNF4:
		decomposed = func() *Node {
			return quantizedGatherLinearOrNF4Decomposed(table, indices, quant)
		}
	default:
		Panicf("nn.QuantizedGather: unsupported quantization scheme %v", quant.Scheme)
	}

	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed()
	}

	backend := table.Graph().Backend()
	if backend != nil && !backend.Capabilities().Operations[compute.OpTypeQuantizedGather] {
		return decomposed()
	}

	if quant == nil || quant.Scheme != compute.QuantGGML || ggml.CanDecompose(quant.GGMLType) {
		res, _ := graphbridge.FusedOpCaller(
			func() *Node { return graphbridge.QuantizedGather(table, indices, quant) },
			decomposed,
			false,
		)
		return res
	}
	return graphbridge.QuantizedGather(table, indices, quant)
}

func quantizedGatherUnpackedDecomposed(table, indices *Node, quant *Quantization) *Node {
	info, err := shapeinference.QuantizedGatherWithInfo(
		table.Shape(), indices.Shape(), quantizationToShapes(quant))
	if err != nil {
		panic(err)
	}
	var packedDT dtypes.DType
	if quant != nil {
		packedDT = quant.PackedDType
	}
	gathered := gatherQuantizedRows(table, indices, packedDT, info.K, info.OutputShape.DType)
	if gathered.DType() != info.OutputShape.DType {
		return ConvertDType(gathered, info.OutputShape.DType)
	}
	return gathered
}

func quantizedGatherLinearOrNF4Decomposed(table, indices *Node, quant *Quantization) *Node {
	info, err := shapeinference.QuantizedGatherWithInfo(
		table.Shape(), indices.Shape(), quantizationToShapes(quant))
	if err != nil {
		panic(err)
	}

	g := table.Graph()
	vocabSize := table.Shape().Dimensions[0]
	K := info.K
	gathered := gatherQuantizedRows(table, indices, quant.PackedDType, K, dtypes.Float32)

	var dequant *Node
	switch quant.Scheme {
	case compute.QuantNF4:
		nibbles := ConvertDType(gathered, dtypes.Int32)
		if gathered.DType() == dtypes.Int4 || gathered.DType() == dtypes.Int8 {
			nibbles = BitwiseAnd(nibbles, Scalar(g, dtypes.Int32, int32(0x0F)))
		}
		nf4Table := Const(g, compute.NF4LookupTable[:])
		dequant = Gather(nf4Table, ExpandAxes(nibbles, -1))
	case compute.QuantLinear:
		dequant = ConvertDType(gathered, dtypes.Float32)
		if quant.ZeroPoint != nil {
			zp := gatherOrExpandEmbeddingParam(quant.ZeroPoint, info.ZeroPointLayout, indices, vocabSize, K)
			dequant = Sub(dequant, zp)
		}
	}

	scale := gatherOrExpandEmbeddingParam(quant.Scale, info.ScaleLayout, indices, vocabSize, K)
	dequant = Mul(dequant, scale)
	if info.OutputShape.DType != dtypes.Float32 {
		dequant = ConvertDType(dequant, info.OutputShape.DType)
	}
	return dequant
}

// gatherQuantizedRows gathers rows along axis 0 from a 2D [rows, cols] tensor.
// When packedDType is set on a Uint8 table, it gathers the packed Uint8 rows first and unpacks
// only the selected rows in portable integer ops (no sub-byte DType or Bitcast required).
// For native packed sub-byte types (Int2/Uint2/Int4/Uint4) whose row width is byte-aligned,
// it bitcasts to Uint8 before gathering so only the gathered rows are unpacked.
// When cols is not byte-aligned, it converts table to fallbackDType before gathering.
func gatherQuantizedRows(table, indices *Node, packedDType dtypes.DType, cols int, fallbackDType dtypes.DType) *Node {
	dt := table.DType()
	if dt == dtypes.Uint8 && packedDType != dtypes.InvalidDType && packedDType.IsPacked() {
		gatheredU8 := Gather(table, indices)
		return unpackPackedUint8Rows(gatheredU8, packedDType, indices, cols)
	}
	if !dt.IsPacked() {
		return Gather(table, indices)
	}
	vpb := dt.ValuesPerStorageUnit()
	if cols%vpb != 0 {
		return Gather(ConvertDType(table, fallbackDType), indices)
	}
	rows := table.Shape().Dimensions[0]
	packedU8 := Bitcast(Reshape(table, rows, cols/vpb, vpb), dtypes.Uint8)
	gatheredU8 := Gather(packedU8, indices)
	gatheredTyped := Bitcast(gatheredU8, dt)
	specs := append(slices.Clone(DimensionSpecsFor(indices)[:indices.Rank()-1]), StaticDim(cols))
	return DynamicReshape(gatheredTyped, specs...)
}

// unpackPackedUint8Rows unpacks gathered [batch..., cols/vpb] Uint8 rows with logical sub-byte
// dtype subDT into [batch..., cols] Int8 (signed) or Uint8 (unsigned) using low-bits-first order.
func unpackPackedUint8Rows(gatheredU8 *Node, subDT dtypes.DType, indices *Node, cols int) *Node {
	g := gatheredU8.Graph()
	vpb := subDT.ValuesPerStorageUnit()
	bits := 8 / vpb
	x := ConvertDType(gatheredU8, dtypes.Int32)
	mask := Scalar(g, dtypes.Int32, int32((1<<bits)-1))
	isSigned := subDT.IsInt() && !subDT.IsUnsigned()
	var signBit *Node
	if isSigned {
		signBit = Scalar(g, dtypes.Int32, int32(1<<(bits-1)))
	}
	parts := make([]*Node, vpb)
	for i := range parts {
		part := x
		if i > 0 {
			part = BitwiseShiftRightLogicalScalar(x, i*bits)
		}
		part = BitwiseAnd(part, mask)
		if isSigned {
			part = Sub(BitwiseXor(part, signBit), signBit)
		}
		parts[i] = ExpandAxes(part, -1)
	}
	specs := append(slices.Clone(DimensionSpecsFor(indices)[:indices.Rank()-1]), StaticDim(cols))
	unpacked := DynamicReshape(Concatenate(parts, -1), specs...)
	if isSigned {
		return ConvertDType(unpacked, dtypes.Int8)
	}
	return ConvertDType(unpacked, dtypes.Uint8)
}

func gatherOrExpandEmbeddingParam(
	param *Node,
	layout shapeinference.BlockParamLayout,
	indices *Node,
	vocabSize, K int,
) *Node {
	batchRank := indices.Rank() - 1
	batchSpecs := DimensionSpecsFor(indices)[:batchRank]

	// Per-tensor scalar parameter: broadcasts across all gathered rows and columns.
	if layout.RowBlockSize == vocabSize && layout.ColBlockSize == K {
		return ConvertDType(Reshape(param), dtypes.Float32)
	}

	// Row-invariant parameter (1D per-column or [1, NumColBlocks]): expand along K once.
	if layout.RowBlockSize == vocabSize {
		p3 := ConvertDType(Reshape(param, 1, layout.NumColBlocks, 1), dtypes.Float32)
		b3 := BroadcastToDims(p3, 1, layout.NumColBlocks, layout.ColBlockSize)
		fullCols := layout.NumColBlocks * layout.ColBlockSize
		row := Reshape(b3, 1, fullCols)
		if fullCols != K {
			row = Slice(row, AxisRange(), AxisRange(0, K))
		}
		return ExpandLeftToRank(row, batchRank+1)
	}

	// Row-indexed parameter (per-row or per-row per-block).
	var p2d *Node
	var gatherIndices *Node
	if layout.RowBlockSize == 1 {
		p2d = Reshape(param, vocabSize, layout.NumColBlocks)
		gatherIndices = indices
	} else {
		numRowBlocks := (vocabSize + layout.RowBlockSize - 1) / layout.RowBlockSize
		p2d = Reshape(param, numRowBlocks, layout.NumColBlocks)
		gatherIndices = DivScalar(indices, layout.RowBlockSize)
	}

	gathered := ConvertDType(gatherQuantizedRows(p2d, gatherIndices, dtypes.InvalidDType, layout.NumColBlocks, dtypes.Float32), dtypes.Float32)
	if layout.ColBlockSize == K || layout.ColBlockSize == 1 {
		// NumColBlocks == 1 broadcasts across K; ColBlockSize == 1 means NumColBlocks == K.
		return gathered
	}

	p3 := InsertAxes(gathered, -1) // [batch..., NumColBlocks, 1]
	bSpecs := append(slices.Clone(batchSpecs), StaticDim(layout.NumColBlocks), StaticDim(layout.ColBlockSize))
	expanded := DynamicBroadcastToDims(p3, bSpecs...)
	fullCols := layout.NumColBlocks * layout.ColBlockSize
	flatSpecs := append(slices.Clone(batchSpecs), StaticDim(fullCols))
	out := DynamicReshape(expanded, flatSpecs...)
	if fullCols != K {
		ranges := make([]SliceAxisSpec, batchRank+1)
		for i := range batchRank {
			ranges[i] = AxisRange()
		}
		ranges[batchRank] = AxisRange(0, K)
		out = Slice(out, ranges...)
	}
	return out
}
