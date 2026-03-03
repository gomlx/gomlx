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
	output := backend.getBuffer(node.shape.DType, operand.shape.Size())
	output.shape = node.shape
	convertFn := convertDTypePairMap.Get(operand.shape.DType, output.shape.DType).(convertFnType)
	convertFn(operand, output)
	return output, nil
}

type convertFnType = func(operand, output *Buffer)

var convertDTypePairMap = NewDTypePairMap("ConvertDType")

func init() {
	// Register sub-byte type conversions (Int4, Uint4).
	// In simplego, Int4 values are stored unpacked as int8 (one value per element),
	// Uint4 values as uint8. The generic converter works because the Go slice
	// element type matches the container type.
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float32, priorityTyped, execConvertDTypeGeneric[int8, float32])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Float64, priorityTyped, execConvertDTypeGeneric[int8, float64])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int32, priorityTyped, execConvertDTypeGeneric[int8, int32])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int64, priorityTyped, execConvertDTypeGeneric[int8, int64])
	convertDTypePairMap.Register(dtypes.Int4, dtypes.Int8, priorityTyped, execConvertDTypeGeneric[int8, int8])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float32, priorityTyped, execConvertDTypeGeneric[uint8, float32])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Float64, priorityTyped, execConvertDTypeGeneric[uint8, float64])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int32, priorityTyped, execConvertDTypeGeneric[uint8, int32])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Int64, priorityTyped, execConvertDTypeGeneric[uint8, int64])
	convertDTypePairMap.Register(dtypes.Uint4, dtypes.Uint8, priorityTyped, execConvertDTypeGeneric[uint8, uint8])

	// Register mutableBytes and fillBuffer for sub-byte types.
	mutableBytesDTypeMap.Register(dtypes.Int4, priorityTyped, mutableBytesGeneric[int8])
	mutableBytesDTypeMap.Register(dtypes.Uint4, priorityTyped, mutableBytesGeneric[uint8])
	fillBufferDTypeMap.Register(dtypes.Int4, priorityTyped, fillBufferGeneric[int8])
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
