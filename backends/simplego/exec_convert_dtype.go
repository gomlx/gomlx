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
	// In simplego, Int4/Uint4 values are stored packed: 2 nibbles per byte.
	// Bitcast from uint8 produces packed buffers (flat = []byte). ConvertDType
	// unpacks them into one value per element of the target type.
	// Low nibble (bits 0-3) is the first element, high nibble (bits 4-7) is the second.
	//
	// Both Int4 and Uint4 unpack to []int8 first (int8 is a common denominator
	// that fits both signed [-8,7] and unsigned [0,15] 4-bit values), then the
	// shared converter promotes to the target type.
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float32, priorityTyped, execConvertPackedSubByte[float32](unpackInt4Nibbles))
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float64, priorityTyped, execConvertPackedSubByte[float64](unpackInt4Nibbles))
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int32, priorityTyped, execConvertPackedSubByte[int32](unpackInt4Nibbles))
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int64, priorityTyped, execConvertPackedSubByte[int64](unpackInt4Nibbles))
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int8, priorityTyped, execConvertPackedSubByte[int8](unpackInt4Nibbles))
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float32, priorityTyped, execConvertPackedSubByte[float32](unpackUint4Nibbles))
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float64, priorityTyped, execConvertPackedSubByte[float64](unpackUint4Nibbles))
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int32, priorityTyped, execConvertPackedSubByte[int32](unpackUint4Nibbles))
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int64, priorityTyped, execConvertPackedSubByte[int64](unpackUint4Nibbles))
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Uint8, priorityTyped, execConvertPackedSubByte[uint8](unpackUint4Nibbles))

	// Register mutableBytes and fillBuffer for sub-byte types.
	// Packed Int4/Uint4 buffers use []byte as the Go storage type.
	mutableBytesDTypeMap.Register(dtypes.Int4, priorityTyped, mutableBytesGeneric[byte])
	mutableBytesDTypeMap.Register(dtypes.Uint4, priorityTyped, mutableBytesGeneric[byte])
	fillBufferDTypeMap.Register(dtypes.Int4, priorityTyped, fillBufferGeneric[byte])
	fillBufferDTypeMap.Register(dtypes.Uint4, priorityTyped, fillBufferGeneric[byte])

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

// unpackNibblesFn is the signature for nibble unpack functions.
// All sub-byte unpack functions output []int8 as a common denominator that fits
// both signed [-8,7] and unsigned [0,15] 4-bit values.
type unpackNibblesFn = func(packed []byte, dst []int8)

// unpackInt4Nibbles unpacks packed Int4 data ([]byte, 2 signed nibbles per byte)
// into dst []int8 (one value per element). Low nibble (bits 0-3) is the first element,
// high nibble (bits 4-7) is the second. Values 8-15 are sign-extended to -8 to -1.
func unpackInt4Nibbles(packed []byte, dst []int8) {
	for i, b := range packed {
		lo := int8(b & 0x0F)
		if lo >= 8 {
			lo -= 16
		}
		hi := int8(b) >> 4 // Arithmetic right-shift preserves sign bit.
		dst[2*i] = lo
		dst[2*i+1] = hi
	}
}

// unpackUint4Nibbles unpacks packed Uint4 data ([]byte, 2 unsigned nibbles per byte)
// into dst []int8 (one value per element). Low nibble (bits 0-3) is the first element,
// high nibble (bits 4-7) is the second. Values 0-15 fit in int8.
func unpackUint4Nibbles(packed []byte, dst []int8) {
	for i, b := range packed {
		dst[2*i] = int8(b & 0x0F)
		dst[2*i+1] = int8(b >> 4)
	}
}

// execConvertPackedSubByte returns a converter for packed sub-byte types (Int4, Uint4).
// The unpackFn parameter selects signed vs unsigned nibble interpretation.
// Sub-byte types are always stored packed as []byte (2 nibbles per byte).
//
// To avoid allocating a temporary slice as large as the output, we process in
// fixed-size blocks that stay on the stack.
func execConvertPackedSubByte[OutT PODNumericConstraints](unpackFn unpackNibblesFn) convertFnType {
	const dstBlockSize = 64
	const srcBlockSize = dstBlockSize / 2 // 2 values per byte for 4-bit types.

	return func(operand, output *Buffer) {
		dstFlat := output.flat.([]OutT)
		srcFlat := operand.flat.([]byte)
		var tmp [dstBlockSize]int8

		var srcIdx, dstIdx int
		for srcIdx+srcBlockSize <= len(srcFlat) {
			unpackFn(srcFlat[srcIdx:srcIdx+srcBlockSize], tmp[:])
			for _, v := range tmp[:] {
				dstFlat[dstIdx] = OutT(v)
				dstIdx++
			}
			srcIdx += srcBlockSize
		}
		// Handle the tail.
		if srcIdx < len(srcFlat) {
			tailSrc := len(srcFlat) - srcIdx
			tailDst := tailSrc * 2
			unpackFn(srcFlat[srcIdx:], tmp[:tailDst])
			for _, v := range tmp[:tailDst] {
				dstFlat[dstIdx] = OutT(v)
				dstIdx++
			}
		}
	}
}
