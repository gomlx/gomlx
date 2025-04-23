package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"slices"
)

func init() {
	nodeExecutors[backends.OpTypeIdentity] = execIdentity
	nodeExecutors[backends.OpTypeWhere] = execWhere
	nodeExecutors[backends.OpTypeReshape] = execReshape
	nodeExecutors[backends.OpTypeReduceMax] = execReduce
	nodeExecutors[backends.OpTypeReduceMin] = execReduce
	nodeExecutors[backends.OpTypeReduceSum] = execReduce
	nodeExecutors[backends.OpTypeReduceProduct] = execReduce
}

// execIdentity implements the Identity op.
func execIdentity(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	if inputsOwned[0] {
		// Trivial case, just pass the buffer forward.
		inputs[0] = nil
		return operand
	}
	output := backend.getBuffer(operand.shape.DType, operand.shape.Size())
	output.shape = operand.shape
	copyFlat(output.flat, operand.flat)
	return output
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

func execWhereGeneric[T SupportedTypesConstraints](conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer) {
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

func execWhereSetOutputWithValue[T SupportedTypesConstraints](outputBuf, valueBuf *Buffer) {
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

func execReduce(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	reduceAxes := node.data.([]int)
	if len(reduceAxes) == 0 {
		return execIdentity(backend, node, inputs, inputsOwned)
	}
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	it := newReduceOutputIterator(operand.shape.Dimensions, reduceAxes)
	dtype := output.shape.DType

	switch node.opType {
	case backends.OpTypeReduceMax:
		dispatchReduceMax.Dispatch(dtype, operand, output, it, dtype)
	case backends.OpTypeReduceMin:
		dispatchReduceMin.Dispatch(dtype, operand, output, it, dtype)
	case backends.OpTypeReduceSum:
		dispatchReduceSum.Dispatch(dtype, operand, output, it)
	case backends.OpTypeReduceProduct:
		dispatchReduceProduct.Dispatch(dtype, operand, output, it)
	default:
		exceptions.Panicf("unsupported reduce op %s", node.opType)
	}
	return output
}

type reduceOutputIterator struct {
	flatIdx int // On the output tensor.

	perAxisIdx    []int // On the operand tensor.
	dimensions    []int // Of the operand tensor.
	perAxisStride []int // It is set to 0 for the axes being reduced.
}

func newReduceOutputIterator(dimensions []int, reduceAxes []int) *reduceOutputIterator {
	inputRank := len(dimensions)
	it := &reduceOutputIterator{
		perAxisIdx: make([]int, inputRank),
		dimensions: dimensions,
	}
	it.perAxisStride = slices.Clone(dimensions)
	stride := 1
	for _, reduceAxis := range reduceAxes {
		it.perAxisStride[reduceAxis] = 0
	}
	for axis := inputRank - 1; axis >= 0; axis-- {
		if it.perAxisStride[axis] == 0 {
			// Skip reduce axes, and leave stride as 0.
			continue
		}

		// Accumulate (product) axes that are not reduced on the stride.
		newStride := stride * it.perAxisStride[axis]
		it.perAxisStride[axis] = stride
		stride = newStride
	}
	return it
}

func (it *reduceOutputIterator) next() int {
	returnIdx := it.flatIdx
	// Move pointer.
	for axis := len(it.perAxisIdx) - 1; axis >= 0; axis-- {
		it.perAxisIdx[axis]++
		it.flatIdx += it.perAxisStride[axis]
		if it.perAxisIdx[axis] < it.dimensions[axis] {
			break
		}

		// Return to the start of current axis and move to next axis.
		it.perAxisIdx[axis] = 0
		it.flatIdx -= it.perAxisStride[axis] * it.dimensions[axis]
	}
	return returnIdx
}

func execReduceInitializeGeneric[T SupportedTypesConstraints](output *Buffer, initialValue T) {
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
}

var dispatchReduceMax = NewDTypeDispatcher("ReduceMax")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceMax -generic=execReduceMaxGeneric -int -uint -float

// execReduceMaxGeneric: use dispatchReduceMax to call it.
func execReduceMaxGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with lowest value.
	initialValue := dtype.LowestValue().(T)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	// Reduce from operand.
	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = max(outputFlat[outputIdx], value)
	}
}

var dispatchReduceMin = NewDTypeDispatcher("ReduceMin")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceMin -generic=execReduceMinGeneric -int -uint -float

func execReduceMinGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with highest value.
	initialValue := dtype.HighestValue().(T)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = min(outputFlat[outputIdx], value)
	}
}

var dispatchReduceSum = NewDTypeDispatcher("ReduceSum")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceSum -generic=execReduceSumGeneric -int -uint -float

func execReduceSumGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator)

	// Initialize with 0.
	initialValue := T(0)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] + value
	}
}

var dispatchReduceProduct = NewDTypeDispatcher("ReduceProduct")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceProduct -generic=execReduceProductGeneric -int -uint -float

func execReduceProductGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator)

	// Initialize with 1.
	initialValue := T(1)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] * value
	}
}
