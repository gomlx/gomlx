package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"slices"
)

func init() {
	nodeExecutors[backends.OpTypeIdentity] = execIdentity
	nodeExecutors[backends.OpTypeWhere] = execWhere
	nodeExecutors[backends.OpTypeReshape] = execReshape
	nodeExecutors[backends.OpTypeTranspose] = execTranspose
	nodeExecutors[backends.OpTypeReduceMax] = execReduce
	nodeExecutors[backends.OpTypeReduceMin] = execReduce
	nodeExecutors[backends.OpTypeReduceSum] = execReduce
	nodeExecutors[backends.OpTypeReduceProduct] = execReduce
}

// IdentityOp ====================================================================================================

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

// WhereOp ====================================================================================================

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

	dispatchWhere.Dispatch(outputShape.DType, condition, onTrue, onFalse, output)
	return output
}

var dispatchWhere = NewDTypeDispatcher("Where")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchWhere -generic=execWhereGeneric -int -uint -float -bf16

func execWhereGeneric[T SupportedTypesConstraints](params ...any) {
	conditionBuf, onTrueBuf, onFalseBuf, outputBuf := params[0].(*Buffer), params[1].(*Buffer), params[2].(*Buffer), params[3].(*Buffer)
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

// ReshapeOp ====================================================================================================

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

// Reduce{Max,Min,Sum,Product}Op ======================================================================================

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

var dispatchReduceMax = NewDTypeDispatcher("ReduceMax")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceMax -generic=execReduceMaxGeneric -int -uint -float

// execReduceMaxGeneric: use dispatchReduceMax to call it.
func execReduceMaxGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with the lowest value.
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

func init() { dispatchReduceMax.Register(dtypes.BFloat16, execReduceMaxBFloat16) }

// execReduceMaxBFloat16: use dispatchReduceMax to call it.
func execReduceMaxBFloat16(params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with the lowest value.
	initialValue := dtype.LowestValue().(bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	// Reduce from operand.
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(max(a, b))
	}
}

var dispatchReduceMin = NewDTypeDispatcher("ReduceMin")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchReduceMin -generic=execReduceMinGeneric -int -uint -float

func execReduceMinGeneric[T PODNumericConstraints](params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with the highest value.
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

func init() { dispatchReduceMin.Register(dtypes.BFloat16, execReduceMinBFloat16) }

func execReduceMinBFloat16(params ...any) {
	operand, output, it, dtype := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator), params[3].(dtypes.DType)

	// Initialize with the highest value.
	initialValue := dtype.HighestValue().(bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(min(a, b))
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

func init() { dispatchReduceSum.Register(dtypes.BFloat16, execReduceSumBFloat16) }

func execReduceSumBFloat16(params ...any) {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator)
	// Initialize with 0.
	initialValue := bfloat16.FromFloat32(0)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(a + b)
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

func init() { dispatchReduceProduct.Register(dtypes.BFloat16, execReduceProductBFloat16) }

func execReduceProductBFloat16(params ...any) {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*reduceOutputIterator)
	// Initialize with 1.
	initialValue := bfloat16.FromFloat32(1)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(a * b)
	}
}

// TransposeOp ====================================================================================================

// execTranspose implements Transpose.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func execTranspose(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	permutations := node.data.([]int)

	// We can't write to the same buffer we read from, because it's not done with swaps.
	output := backend.getBuffer(operand.shape.DType, operand.shape.Size())
	output.shape = node.shape
	it := newTransposeIterator(operand.shape, permutations)
	dtype := node.shape.DType
	dispatchTranspose.Dispatch(dtype, operand, output, it)
	return output
}

type transposeIterator struct {
	flatIdx                                int
	perAxisIdx, perAxisStrides, dimensions []int
}

// newTransposeIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
func newTransposeIterator(operand shapes.Shape, permutations []int) *transposeIterator {
	rank := operand.Rank()

	it := &transposeIterator{
		perAxisIdx:     make([]int, rank),
		perAxisStrides: make([]int, rank),
		dimensions:     operand.Dimensions,
	}

	// First calculate strides on the output.
	stridesOnOutput := make([]int, rank)
	stride := 1
	reversePermutations := make([]int, rank)
	for reverseAxis := range rank {
		outputAxis := rank - reverseAxis - 1
		stridesOnOutput[outputAxis] = stride
		operandAxis := permutations[outputAxis]
		stride *= operand.Dimensions[operandAxis]
		reversePermutations[operandAxis] = outputAxis
	}

	// Calculate per operand axis, what is the stride on the output.
	for operandAxis := range rank {
		outputAxis := reversePermutations[operandAxis]
		it.perAxisStrides[operandAxis] = stridesOnOutput[outputAxis]
	}
	return it
}

func (it *transposeIterator) next() (nextFlatIdx int) {
	nextFlatIdx = it.flatIdx
	rank := len(it.perAxisIdx)
	for axis := rank - 1; axis >= 0; axis-- {
		it.perAxisIdx[axis]++
		it.flatIdx += it.perAxisStrides[axis]
		if it.perAxisIdx[axis] < it.dimensions[axis] {
			// We are done.
			break
		}
		// Otherwise, rewind current axis and move to next.
		it.perAxisIdx[axis] = 0
		it.flatIdx -= it.perAxisStrides[axis] * it.dimensions[axis]
	}
	return
}

var dispatchTranspose = NewDTypeDispatcher("Transpose")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchTranspose -generic=execTransposeGeneric -int -uint -float -bf16

func execTransposeGeneric[T SupportedTypesConstraints](params ...any) {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*transposeIterator)
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	for _, value := range operandFlat {
		outputFlat[it.next()] = value
	}
}
