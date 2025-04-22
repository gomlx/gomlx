package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
)

func init() {
	nodeExecutors[backends.OpTypeNeg] = execNeg
	nodeExecutors[backends.OpTypeAbs] = execAbs
	nodeExecutors[backends.OpTypeSign] = execSign
	nodeExecutors[backends.OpTypeLogicalNot] = execLogicalNot
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

func execNegGeneric[T signedNumericPODConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = -input
	}
}

func execNegBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(-input.Float32())
	}
}

// execAbs executes the unary op Abs.
func execAbs(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Int8:
		execAbsGeneric[int8](input.flat.([]int8), output.flat.([]int8))
	case dtypes.Int16:
		execAbsGeneric[int16](input.flat.([]int16), output.flat.([]int16))
	case dtypes.Int32:
		execAbsGeneric[int32](input.flat.([]int32), output.flat.([]int32))
	case dtypes.Int64:
		execAbsGeneric[int64](input.flat.([]int64), output.flat.([]int64))
	case dtypes.Uint8:
		execAbsUnsignedGeneric[uint8](input, output)
	case dtypes.Uint16:
		execAbsUnsignedGeneric[uint16](input, output)
	case dtypes.Uint32:
		execAbsUnsignedGeneric[uint32](input, output)
	case dtypes.Uint64:
		execAbsUnsignedGeneric[uint64](input, output)
	case dtypes.Float32:
		execAbsGeneric[float32](input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		execAbsGeneric[float64](input.flat.([]float64), output.flat.([]float64))
	case dtypes.BFloat16:
		execAbsBF16(input.flat.([]bfloat16.BFloat16), output.flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.shape.DType, node.opType)
	}
	return output
}

func execAbsGeneric[T signedNumericPODConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		if input < 0 {
			outputs[ii] = -input
		} else {
			outputs[ii] = input
		}
	}
}

func execAbsUnsignedGeneric[T unsignedPODConstraints](input, output *Buffer) {
	if input == output {
		return
	}
	copy(output.flat.([]T), input.flat.([]T))
}

func execAbsBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		f := input.Float32()
		if f < 0 {
			outputs[ii] = bfloat16.FromFloat32(-f)
		} else {
			outputs[ii] = input
		}
	}
}

// execSign executes the unary op Sign.
func execSign(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	switch input.shape.DType {
	case dtypes.Int8:
		execSignGeneric[int8](input.flat.([]int8), output.flat.([]int8))
	case dtypes.Int16:
		execSignGeneric[int16](input.flat.([]int16), output.flat.([]int16))
	case dtypes.Int32:
		execSignGeneric[int32](input.flat.([]int32), output.flat.([]int32))
	case dtypes.Int64:
		execSignGeneric[int64](input.flat.([]int64), output.flat.([]int64))
	case dtypes.Uint8:
		execSignForUnsignedGeneric[uint8](input.flat.([]uint8), output.flat.([]uint8))
	case dtypes.Uint16:
		execSignForUnsignedGeneric[uint16](input.flat.([]uint16), output.flat.([]uint16))
	case dtypes.Uint32:
		execSignForUnsignedGeneric[uint32](input.flat.([]uint32), output.flat.([]uint32))
	case dtypes.Uint64:
		execSignForUnsignedGeneric[uint64](input.flat.([]uint64), output.flat.([]uint64))
	case dtypes.Float32:
		execSignGeneric[float32](input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		execSignGeneric[float64](input.flat.([]float64), output.flat.([]float64))
	case dtypes.BFloat16:
		execSignBF16(input.flat.([]bfloat16.BFloat16), output.flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.shape.DType, node.opType)
	}
	return output
}

func execSignGeneric[T signedNumericPODConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		switch {
		case input < 0:
			outputs[ii] = -1
		case input > 0:
			outputs[ii] = 1
		default:
			outputs[ii] = 0
		}
	}
}

func execSignForUnsignedGeneric[T unsignedPODConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		if input > 0 {
			outputs[ii] = 1
		} else {
			outputs[ii] = 0
		}
	}
}

func execSignBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		f := input.Float32()
		switch {
		case f < 0:
			outputs[ii] = bfloat16.FromFloat32(-1.0)
		case f > 0:
			outputs[ii] = bfloat16.FromFloat32(1.0)
		default:
			outputs[ii] = bfloat16.FromFloat32(0.0)
		}
	}
}

// execLogicalNot executes the unary op LogicalNot.
func execLogicalNot(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if input.shape.DType != dtypes.Bool {
		exceptions.Panicf("unsupported data type %s for %s", input.shape.DType, node.opType)
	}
	for ii, val := range input.flat.([]bool) {
		output.flat.([]bool)[ii] = !val
	}
	return output
}
