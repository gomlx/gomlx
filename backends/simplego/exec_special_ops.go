package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

func init() {
	nodeExecutors[backends.OpTypeWhere] = execWhere
	nodeExecutors[backends.OpTypeReshape] = execReshape
}

// execWhere implements the Where op.
func execWhere(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]

	// Figure out what the outputBuffer is going to be.
	outputShape := node.shape
	var output *Buffer
	if onTrue.shape.Equal(outputShape) && inputsOwned[1] {
		output = onTrue
		inputs[1] = nil
	} else if onFalse.shape.Equal(outputShape) && inputsOwned[2] {
		output = onFalse
		inputs[2] = nil
	} else {
		output = backend.getBuffer(outputShape.DType, outputShape.Size())
		output.shape = outputShape
	}

	switch outputShape.DType {
	case dtypes.Bool:
		execWhereGeneric[bool](condition, onTrue, onFalse, output)
	case dtypes.Int8:
		execWhereGeneric[int8](condition, onTrue, onFalse, output)
	case dtypes.Int16:
		execWhereGeneric[int16](condition, onTrue, onFalse, output)
	case dtypes.Int32:
		execWhereGeneric[int32](condition, onTrue, onFalse, output)
	case dtypes.Int64:
		execWhereGeneric[int64](condition, onTrue, onFalse, output)
	case dtypes.Uint8:
		execWhereGeneric[uint8](condition, onTrue, onFalse, output)
	case dtypes.Uint16:
		execWhereGeneric[uint16](condition, onTrue, onFalse, output)
	case dtypes.Uint32:
		execWhereGeneric[uint32](condition, onTrue, onFalse, output)
	case dtypes.Uint64:
		execWhereGeneric[uint64](condition, onTrue, onFalse, output)
	case dtypes.Float32:
		execWhereGeneric[float32](condition, onTrue, onFalse, output)
	case dtypes.Float64:
		execWhereGeneric[float64](condition, onTrue, onFalse, output)
	case dtypes.BFloat16:
		execWhereGeneric[bfloat16.BFloat16](condition, onTrue, onFalse, output)
	default:
		exceptions.Panicf("unsupported DType %s for Where() operation", outputShape.DType)
	}
	return output
}

func execWhereGeneric[T supportedTypesConstraints](conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer) {
	if conditionBuf.shape.IsScalar() {
		// Case 1: condition is a scalar, either we take onTrue or onFalse as a whole (with potential broadcast).
		if conditionBuf.flat.([]bool)[0] {
			execWhereSetOutputWithValue[T](outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputWithValue[T](outputBuf, onFalseBuf)
		}
		return
	}

	conditionFlat := conditionBuf.flat.([]bool)
	onTrueFlat := onTrueBuf.flat.([]T)
	onFalseFlat := onFalseBuf.flat.([]T)
	outputFlat := outputBuf.flat.([]T)
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

func execWhereSetOutputWithValue[T supportedTypesConstraints](outputBuf, valueBuf *Buffer) {
	if valueBuf == outputBuf {
		// The output is reusing the value buffer, nothing to do.
		return
	}
	if valueBuf.shape.Equal(outputBuf.shape) {
		// Copy over values.
		copy(outputBuf.flat.([]T), valueBuf.flat.([]T))
		return
	}
	// Value must then be a scalar:
	c := valueBuf.flat.([]T)[0]
	outputSlice := outputBuf.flat.([]T)
	for outputIdx := range outputSlice {
		outputSlice[outputIdx] = c
	}
}

// execReshape implements Reshape.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func execReshape(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	var output *Buffer
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil
	} else {
		output = backend.getBuffer(operand.shape.DType, operand.shape.Size())
	}
	output.shape = node.shape
	return output
}
