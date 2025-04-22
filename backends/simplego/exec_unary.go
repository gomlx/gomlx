package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

func init() {
	nodeExecutors[backends.OpTypeNeg] = execNeg
}

// unaryOperandAndOutput is a convenience function to get the input and output -- which may be the reuse of the input
func unaryOperandAndOutput(backend *Backend, inputs []*Buffer, inputsOwned []bool) (input, output *Buffer) {
	input = inputs[0]
	if inputsOwned[0] {
		output = input
		inputs[0] = nil // This tells the executor that we took over the buffer.
		return
	}
	output = backend.getBuffer(input.shape.DType, input.shape.Size())
	output.shape = input.shape.Clone()
	return input, output
}

// execNeg executes the unary op Neg.
func execNeg(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Int8:
		execNegGeneric[int8](input.flat.([]int8), output.flat.([]int8))
	case dtypes.Int16:
		execNegGeneric[int16](input.flat.([]int16), output.flat.([]int16))
	case dtypes.Int32:
		execNegGeneric[int32](input.flat.([]int32), output.flat.([]int32))
	case dtypes.Int64:
		execNegGeneric[int64](input.flat.([]int64), output.flat.([]int64))
	case dtypes.Uint8:
		execNegGeneric[uint8](input.flat.([]uint8), output.flat.([]uint8))
	case dtypes.Uint16:
		execNegGeneric[uint16](input.flat.([]uint16), output.flat.([]uint16))
	case dtypes.Uint32:
		execNegGeneric[uint32](input.flat.([]uint32), output.flat.([]uint32))
	case dtypes.Uint64:
		execNegGeneric[uint64](input.flat.([]uint64), output.flat.([]uint64))
	case dtypes.Float32:
		execNegGeneric[float32](input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		execNegGeneric[float64](input.flat.([]float64), output.flat.([]float64))
	case dtypes.BFloat16:
		execNegBF16(input.flat.([]bfloat16.BFloat16), output.flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.shape.DType, node.opType)
	}
	return output
}

func execNegGeneric[T numericPODConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = -input
	}
}

func execNegBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(-input.Float32())
	}
}
