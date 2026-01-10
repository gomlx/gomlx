// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

// Float16 implementations of special operations.
// These are separated from exec_special_ops.go to keep files organized by dtype.

import (
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
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
