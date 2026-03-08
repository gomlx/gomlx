// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

// ConvertDType ====================================================================================================

func init() {
	setNodeExecutor(backends.OpTypeConvertDType, priorityGeneric, execConvertDType)
}

func execConvertDType(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	_ = inputsOwned // We don't reuse the inputs.
	output, err := backend.getBuffer(node.shape.DType, operand.shape.Size())
	if err != nil {
		return nil, err
	}
	output.shape = node.shape
	convertFn := convertDTypePairMap.Get(operand.shape.DType, output.shape.DType).(convertFnType)
	convertFn(operand, output)
	return output, nil
}

type convertFnType = func(operand, output *Buffer)

var convertDTypePairMap = NewDTypePairMap("ConvertDType")

func init() {
	// Register sub-byte type conversions (Int4, Uint4).
	// In simplego, Int4/Uint4 values are stored packed: 2 nibbles per uint8 byte.
	// Bitcast from uint8 produces packed buffers (flat = []uint8). ConvertDType
	// unpacks them into one value per element of the target type.
	// Low nibble (bits 0-3) is the first element, high nibble (bits 4-7) is the second.
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float32, priorityTyped, execConvertPackedInt4[float32])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float64, priorityTyped, execConvertPackedInt4[float64])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int32, priorityTyped, execConvertPackedInt4[int32])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int64, priorityTyped, execConvertPackedInt4[int64])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int8, priorityTyped, execConvertPackedInt4[int8])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float32, priorityTyped, execConvertPackedUint4[float32])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float64, priorityTyped, execConvertPackedUint4[float64])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int32, priorityTyped, execConvertPackedUint4[int32])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int64, priorityTyped, execConvertPackedUint4[int64])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Uint8, priorityTyped, execConvertPackedUint4[uint8])

	// Register mutableBytes and fillBuffer for sub-byte types.
	// Packed Int4/Uint4 buffers use []uint8 as the Go storage type.
	mutableBytesDTypeMap.Register(dtypes.Int4, priorityTyped, mutableBytesGeneric[uint8])
	mutableBytesDTypeMap.Register(dtypes.Uint4, priorityTyped, mutableBytesGeneric[uint8])
	fillBufferDTypeMap.Register(dtypes.Int4, priorityTyped, fillBufferGeneric[uint8])
	fillBufferDTypeMap.Register(dtypes.Uint4, priorityTyped, fillBufferGeneric[uint8])

	// Manually register bool x bfloat16 conversion functions.
	convertDTypePairMap.Register(dtypes.BFloat16, dtypes.Bool, priorityTyped, execConvertDTypeBFloat16ToBool)
	convertDTypePairMap.Register(dtypes.Bool, dtypes.BFloat16, priorityTyped, execConvertDTypeBoolToBFloat16)

	// Manually register bool x float16 conversion functions.
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Bool, priorityTyped, execConvertDTypeFloat16ToBool)
	convertDTypePairMap.Register(dtypes.Bool, dtypes.Float16, priorityTyped, execConvertDTypeBoolToFloat16)

	// Manually register float16 x bfloat16 conversion functions.
	convertDTypePairMap.Register(dtypes.Float16, dtypes.BFloat16, priorityTyped, execConvertDTypeFloat16ToBFloat16)
	convertDTypePairMap.Register(dtypes.BFloat16, dtypes.Float16, priorityTyped, execConvertDTypeBFloat16ToFloat16)
}

func execConvertDTypeGeneric[FromT PODNumericConstraints, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		outputFlat[idx] = ToT(value)
	}
}

func execConvertDTypeFromBFloat16[_ bfloat16.BFloat16, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		outputFlat[idx] = ToT(value.Float32())
	}
}

func execConvertDTypeToBFloat16[FromT PODNumericConstraints, _ bfloat16.BFloat16](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for idx, value := range operandFlat {
		outputFlat[idx] = bfloat16.FromFloat32(float32(value))
	}
}

func execConvertDTypeFromBool[_ bool, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]bool)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		if value {
			outputFlat[idx] = ToT(1)
		} else {
			outputFlat[idx] = ToT(0)
		}
	}
}

func execConvertDTypeToBool[FromT PODNumericConstraints, _ bool](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]bool)
	for idx, value := range operandFlat {
		outputFlat[idx] = value != 0
	}
}

func execConvertDTypeBFloat16ToBool(operand, output *Buffer) {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bool)
	for idx, value := range operandFlat {
		outputFlat[idx] = value.Float32() != 0
	}
}

func execConvertDTypeBoolToBFloat16(operand, output *Buffer) {
	operandFlat := operand.flat.([]bool)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	zero, one := bfloat16.FromFloat32(0), bfloat16.FromFloat32(1)
	for idx, value := range operandFlat {
		if value {
			outputFlat[idx] = one
		} else {
			outputFlat[idx] = zero
		}
	}
}

func execConvertDTypeFromFloat16[_ float16.Float16, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]float16.Float16)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		outputFlat[idx] = ToT(value.Float32())
	}
}

func execConvertDTypeToFloat16[FromT PODNumericConstraints, _ float16.Float16](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]float16.Float16)
	for idx, value := range operandFlat {
		outputFlat[idx] = float16.Fromfloat32(float32(value))
	}
}

func execConvertDTypeFloat16ToBool(operand, output *Buffer) {
	operandFlat := operand.flat.([]float16.Float16)
	outputFlat := output.flat.([]bool)
	for idx, value := range operandFlat {
		outputFlat[idx] = value.Float32() != 0
	}
}

func execConvertDTypeBoolToFloat16(operand, output *Buffer) {
	operandFlat := operand.flat.([]bool)
	outputFlat := output.flat.([]float16.Float16)
	zero, one := float16.Fromfloat32(0), float16.Fromfloat32(1)
	for idx, value := range operandFlat {
		if value {
			outputFlat[idx] = one
		} else {
			outputFlat[idx] = zero
		}
	}
}

func execConvertDTypeFloat16ToBFloat16(operand, output *Buffer) {
	operandFlat := operand.flat.([]float16.Float16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for idx, value := range operandFlat {
		outputFlat[idx] = bfloat16.FromFloat32(value.Float32())
	}
}

func execConvertDTypeBFloat16ToFloat16(operand, output *Buffer) {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]float16.Float16)
	for idx, value := range operandFlat {
		outputFlat[idx] = float16.Fromfloat32(value.Float32())
	}
}

// unpackInt4Nibbles unpacks packed Int4 data ([]uint8, 2 signed nibbles per byte)
// into dst (one value per element). Low nibble (bits 0-3) is the first element,
// high nibble (bits 4-7) is the second. Values 8-15 are sign-extended to -8 to -1.
func unpackInt4Nibbles(packed []uint8, dst []int8) {
	for i, b := range packed {
		lo := int8(b & 0x0F)
		hi := int8(b >> 4)
		if lo >= 8 {
			lo -= 16
		}
		if hi >= 8 {
			hi -= 16
		}
		dst[2*i] = lo
		dst[2*i+1] = hi
	}
}

// unpackUint4Nibbles unpacks packed Uint4 data ([]uint8, 2 unsigned nibbles per byte)
// into dst (one value per element). Low nibble (bits 0-3) is the first element,
// high nibble (bits 4-7) is the second.
func unpackUint4Nibbles(packed []uint8, dst []uint8) {
	for i, b := range packed {
		dst[2*i] = b & 0x0F
		dst[2*i+1] = b >> 4
	}
}

// execConvertPackedInt4 converts Int4 values to the target numeric type with sign extension.
// Handles both packed ([]uint8, 2 nibbles per byte, from Bitcast) and unpacked ([]int8,
// one value per element, from getBuffer) source buffers.
func execConvertPackedInt4[ToT PODNumericConstraints](operand, output *Buffer) {
	dstData := output.flat.([]ToT)
	switch src := operand.flat.(type) {
	case []uint8:
		// Packed: unpack nibbles then convert.
		tmp := make([]int8, len(dstData))
		unpackInt4Nibbles(src, tmp)
		for i, v := range tmp {
			dstData[i] = ToT(v)
		}
	case []int8:
		// Unpacked: one signed value per element.
		for i, v := range src {
			dstData[i] = ToT(v)
		}
	}
}

// execConvertPackedUint4 converts Uint4 values to the target numeric type.
// Handles both packed ([]uint8, 2 nibbles per byte, from Bitcast) and unpacked ([]uint8,
// one value per element, from getBuffer) source buffers. Since both forms use []uint8,
// packed vs unpacked is distinguished by length: packed has len == Size()/2.
func execConvertPackedUint4[ToT PODNumericConstraints](operand, output *Buffer) {
	srcData := operand.flat.([]uint8)
	dstData := output.flat.([]ToT)
	if len(srcData) == len(dstData) {
		// Unpacked: one value per element.
		for i, v := range srcData {
			dstData[i] = ToT(v)
		}
	} else {
		// Packed: unpack nibbles then convert.
		tmp := make([]uint8, len(dstData))
		unpackUint4Nibbles(srcData, tmp)
		for i, v := range tmp {
			dstData[i] = ToT(v)
		}
	}
}
