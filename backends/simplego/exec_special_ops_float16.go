package simplego

// Float16 implementations of special operations.
// These are separated from exec_special_ops.go to keep files organized by dtype.

import (
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/x448/float16"
)

// Float16 reduce operations

func init() {
	reduceMaxDTypeMap.Register(dtypes.Float16, priorityTyped, execReduceMaxFloat16)
	reduceMinDTypeMap.Register(dtypes.Float16, priorityTyped, execReduceMinFloat16)
	reduceSumDTypeMap.Register(dtypes.Float16, priorityTyped, execReduceSumFloat16)
	reduceProductDTypeMap.Register(dtypes.Float16, priorityTyped, execReduceProductFloat16)
}

func execReduceMaxFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with the lowest value.
	initialValue := dtype.LowestValue().(float16.Float16)
	outputFlat := output.flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	// Reduce from operand.
	operandFlat := operand.flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.Fromfloat32(max(a, b))
	}
}

func execReduceMinFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with the highest value.
	initialValue := dtype.HighestValue().(float16.Float16)
	outputFlat := output.flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.Fromfloat32(min(a, b))
	}
}

func execReduceSumFloat16(operand, output *Buffer, it *reduceOutputIterator, _ dtypes.DType) {
	// Initialize with 0.
	initialValue := float16.Fromfloat32(0)
	outputFlat := output.flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.Fromfloat32(a + b)
	}
}

func execReduceProductFloat16(operand, output *Buffer, it *reduceOutputIterator, _ dtypes.DType) {
	// Initialize with 1.
	initialValue := float16.Fromfloat32(1)
	outputFlat := output.flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.Fromfloat32(a * b)
	}
}

// Float16 conversion functions

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

func init() {
	// Register Float16 conversion functions
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Int8, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, int8])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Int16, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, int16])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Int32, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, int32])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Int64, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, int64])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Uint8, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, uint8])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Uint16, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, uint16])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Uint32, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, uint32])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Uint64, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, uint64])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Float32, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, float32])
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Float64, priorityTyped, execConvertDTypeFromFloat16[float16.Float16, float64])

	convertDTypePairMap.Register(dtypes.Int8, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[int8, float16.Float16])
	convertDTypePairMap.Register(dtypes.Int16, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[int16, float16.Float16])
	convertDTypePairMap.Register(dtypes.Int32, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[int32, float16.Float16])
	convertDTypePairMap.Register(dtypes.Int64, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[int64, float16.Float16])
	convertDTypePairMap.Register(dtypes.Uint8, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[uint8, float16.Float16])
	convertDTypePairMap.Register(dtypes.Uint16, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[uint16, float16.Float16])
	convertDTypePairMap.Register(dtypes.Uint32, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[uint32, float16.Float16])
	convertDTypePairMap.Register(dtypes.Uint64, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[uint64, float16.Float16])
	convertDTypePairMap.Register(dtypes.Float32, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[float32, float16.Float16])
	convertDTypePairMap.Register(dtypes.Float64, dtypes.Float16, priorityTyped, execConvertDTypeToFloat16[float64, float16.Float16])

	convertDTypePairMap.Register(dtypes.Float16, dtypes.Bool, priorityTyped, execConvertDTypeFloat16ToBool)
	convertDTypePairMap.Register(dtypes.Bool, dtypes.Float16, priorityTyped, execConvertDTypeBoolToFloat16)

	// Float16 <-> BFloat16 conversion
	convertDTypePairMap.Register(dtypes.Float16, dtypes.BFloat16, priorityTyped, execConvertDTypeFloat16ToBFloat16)
	convertDTypePairMap.Register(dtypes.BFloat16, dtypes.Float16, priorityTyped, execConvertDTypeBFloat16ToFloat16)

	// Float16 <-> Float16 identity
	convertDTypePairMap.Register(dtypes.Float16, dtypes.Float16, priorityTyped, func(operand, output *Buffer) {
		copy(output.flat.([]float16.Float16), operand.flat.([]float16.Float16))
	})
}

// Float16 buffer operations

func mutableBytesFloat16(b *Buffer) []byte {
	flat := b.flat.([]float16.Float16)
	bytePointer := (*byte)(unsafe.Pointer(&flat[0]))
	return unsafe.Slice(bytePointer, len(flat)*2) // Float16 is 2 bytes
}

func fillBufferFloat16(b *Buffer, valueAny any) {
	var value float16.Float16
	if valueAny != nil {
		value = valueAny.(float16.Float16)
	}
	flat := b.flat.([]float16.Float16)
	for i := range flat {
		flat[i] = value
	}
}

func execWhereFloat16(conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer) {
	if conditionBuf.shape.IsScalar() {
		if conditionBuf.flat.([]bool)[0] {
			execWhereSetOutputFloat16(outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputFloat16(outputBuf, onFalseBuf)
		}
		return
	}
	conditionFlat := conditionBuf.flat.([]bool)
	onTrueFlat := onTrueBuf.flat.([]float16.Float16)
	onFalseFlat := onFalseBuf.flat.([]float16.Float16)
	outputFlat := outputBuf.flat.([]float16.Float16)
	onTrueIsScalar := onTrueBuf.shape.IsScalar()
	onFalseIsScalar := onFalseBuf.shape.IsScalar()
	onTrue := onTrueFlat[0]
	onFalse := onFalseFlat[0]
	for outputIdx, condition := range conditionFlat {
		if condition {
			if !onTrueIsScalar {
				onTrue = onTrueFlat[outputIdx]
			}
			outputFlat[outputIdx] = onTrue
		} else {
			if !onFalseIsScalar {
				onFalse = onFalseFlat[outputIdx]
			}
			outputFlat[outputIdx] = onFalse
		}
	}
}

func execWhereSetOutputFloat16(outputBuf, valueBuf *Buffer) {
	if valueBuf == outputBuf {
		return
	}
	if valueBuf.shape.Equal(outputBuf.shape) {
		copy(outputBuf.flat.([]float16.Float16), valueBuf.flat.([]float16.Float16))
		return
	}
	// Broadcast scalar
	value := valueBuf.flat.([]float16.Float16)[0]
	output := outputBuf.flat.([]float16.Float16)
	for i := range output {
		output[i] = value
	}
}

func execTransposeFloat16(operand, output *Buffer, it *transposeIterator) {
	operandFlat := operand.flat.([]float16.Float16)
	outputFlat := output.flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputFlat[it.next()] = value
	}
}

func execBroadcastFloat16(params ...any) any {
	operandFlat, outputFlat, repeats := params[0].([]float16.Float16), params[1].([]float16.Float16), params[2].(int)
	pos := 0
	for range repeats {
		copy(outputFlat[pos:], operandFlat)
		pos += len(operandFlat)
	}
	return nil
}

func execBroadcastInDimFloat16(params ...any) any {
	operandFlat, outputFlat, operandIterAny := params[0].([]float16.Float16), params[1].([]float16.Float16), params[2]
	if operandIterAny == nil {
		// Special case, where operand is a scalar that is broadcast everywhere.
		xslices.FillSlice(outputFlat, operandFlat[0])
		return nil
	}
	operandIter := operandIterAny.(*broadcastIterator)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = operandFlat[operandIter.Next()]
	}
	return nil
}

func execSliceFloat16(operand, output *Buffer, params *sliceNode) {
	rank := operand.shape.Rank()
	outputFlat := output.flat.([]float16.Float16)
	operandFlat := operand.flat.([]float16.Float16)

	// Find operandFlatIdx start value.
	var operandFlatIdx int
	operandFlatStrides := calculateStrides(operand.shape.Dimensions)
	for axis, idx := range params.starts {
		operandFlatIdx += operandFlatStrides[axis] * idx
		// Scale the flat index strides by the requested strides for this axis.
		operandFlatStrides[axis] *= params.strides[axis]
	}

	operandPerAxisIdx := make([]int, rank)
	operandPerAxisSize := output.shape.Dimensions

	for outputFlatIdx := range outputFlat {
		// Copy value at current position.
		outputFlat[outputFlatIdx] = operandFlat[operandFlatIdx]

		// Iterate to the next operand position.
		for axis := rank - 1; axis >= 0; axis-- {
			if operandPerAxisSize[axis] == 1 {
				// We don't iterate on this axis.
				continue
			}

			// Increment the current axis.
			operandPerAxisIdx[axis]++
			operandFlatIdx += operandFlatStrides[axis]
			if operandPerAxisIdx[axis] < operandPerAxisSize[axis] {
				// Done for this iteration.
				break
			}

			// Rewind the current axis: we will bump the next axis for this iteration.
			operandPerAxisIdx[axis] = 0
			operandFlatIdx -= operandPerAxisSize[axis] * operandFlatStrides[axis]
		}
	}
}

func init() {
	// Register Float16 buffer and misc operations
	mutableBytesDTypeMap.Register(dtypes.Float16, priorityTyped, mutableBytesFloat16)
	fillBufferDTypeMap.Register(dtypes.Float16, priorityTyped, fillBufferFloat16)
	whereDTypeMap.Register(dtypes.Float16, priorityTyped, execWhereFloat16)
	transposeDTypeMap.Register(dtypes.Float16, priorityTyped, execTransposeFloat16)
	dispatchBroadcast.Register(dtypes.Float16, priorityTyped, execBroadcastFloat16)
	dispatchBroadcastInDim.Register(dtypes.Float16, priorityTyped, execBroadcastInDimFloat16)
	sliceDTypeMap.Register(dtypes.Float16, priorityTyped, execSliceFloat16)
}
