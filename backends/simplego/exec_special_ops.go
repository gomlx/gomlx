package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"slices"
)

func init() {
	nodeExecutors[backends.OpTypeIdentity] = execIdentity
	nodeExecutors[backends.OpTypeWhere] = execWhere
	nodeExecutors[backends.OpTypeReshape] = execReshape
	nodeExecutors[backends.OpTypeTranspose] = execTranspose
	nodeExecutors[backends.OpTypeBroadcast] = execBroadcast
	nodeExecutors[backends.OpTypeBroadcastInDim] = execBroadcastInDim
	nodeExecutors[backends.OpTypeReduceMax] = execReduce
	nodeExecutors[backends.OpTypeReduceMin] = execReduce
	nodeExecutors[backends.OpTypeReduceSum] = execReduce
	nodeExecutors[backends.OpTypeReduceProduct] = execReduce
	nodeExecutors[backends.OpTypeIota] = execIota
	nodeExecutors[backends.OpTypeGather] = execGather
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

	_ = dispatchWhere.Dispatch(outputShape.DType, condition, onTrue, onFalse, output)
	return output
}

var dispatchWhere = NewDTypeDispatcher("Where")

func execWhereGeneric[T SupportedTypesConstraints](params ...any) any {
	conditionBuf, onTrueBuf, onFalseBuf, outputBuf := params[0].(*Buffer), params[1].(*Buffer), params[2].(*Buffer), params[3].(*Buffer)
	if conditionBuf.shape.IsScalar() {
		// Case 1: condition is a scalar, either we take onTrue or onFalse as a whole (with potential broadcast).
		if conditionBuf.flat.([]bool)[0] {
			execWhereSetOutputWithValue[T](outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputWithValue[T](outputBuf, onFalseBuf)
		}
		return nil
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
	return nil
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

// execReduceMaxGeneric: use dispatchReduceMax to call it.
func execReduceMaxGeneric[T PODNumericConstraints](params ...any) any {
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
	return nil
}

func init() { dispatchReduceMax.Register(dtypes.BFloat16, execReduceMaxBFloat16) }

// execReduceMaxBFloat16: use dispatchReduceMax to call it.
func execReduceMaxBFloat16(params ...any) any {
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
	return nil
}

var dispatchReduceMin = NewDTypeDispatcher("ReduceMin")

func execReduceMinGeneric[T PODNumericConstraints](params ...any) any {
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
	return nil
}

func init() { dispatchReduceMin.Register(dtypes.BFloat16, execReduceMinBFloat16) }

func execReduceMinBFloat16(params ...any) any {
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
	return nil
}

var dispatchReduceSum = NewDTypeDispatcher("ReduceSum")

func execReduceSumGeneric[T PODNumericConstraints](params ...any) any {
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
	return nil
}

func init() { dispatchReduceSum.Register(dtypes.BFloat16, execReduceSumBFloat16) }

func execReduceSumBFloat16(params ...any) any {
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
	return nil
}

var dispatchReduceProduct = NewDTypeDispatcher("ReduceProduct")

func execReduceProductGeneric[T PODNumericConstraints](params ...any) any {
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
	return nil
}

func init() { dispatchReduceProduct.Register(dtypes.BFloat16, execReduceProductBFloat16) }

func execReduceProductBFloat16(params ...any) any {
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
	return nil
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

func execTransposeGeneric[T SupportedTypesConstraints](params ...any) any {
	operand, output, it := params[0].(*Buffer), params[1].(*Buffer), params[2].(*transposeIterator)
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	for _, value := range operandFlat {
		outputFlat[it.next()] = value
	}
	return nil
}

// BroadcastOp ====================================================================================================

func execBroadcast(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	prefixDims := node.data.([]int)
	repeats := 1
	for _, dim := range prefixDims {
		repeats *= dim
	}
	dispatchBroadcast.Dispatch(node.shape.DType, operand.flat, output.flat, repeats)
	return output
}

var dispatchBroadcast = NewDTypeDispatcher("Broadcast")

func execBroadcastGeneric[T SupportedTypesConstraints](params ...any) any {
	operandFlat, outputFlat, repeats := params[0].([]T), params[1].([]T), params[2].(int)
	pos := 0
	for _ = range repeats {
		copy(outputFlat[pos:], operandFlat)
		pos += len(operandFlat)
	}
	return nil
}

// BroadcastInDimsOp ====================================================================================================

func execBroadcastInDim(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand := inputs[0]
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape

	// Special case: if operand is a scalar, we just pass a nil iterator.
	if operand.shape.Size() == 1 {
		dispatchBroadcastInDim.Dispatch(output.shape.DType, operand.flat, output.flat, nil)
		return output
	}

	// Reshape operand shape: same dimension as the operand on the corresponding axes, 1 elsewhere.
	// Notice it's the same size, the flat data doesn't change.
	reshapedOperand := shapes.Make(operand.shape.DType)
	reshapedOperand.Dimensions = make([]int, output.shape.Rank())
	xslices.FillSlice(reshapedOperand.Dimensions, 1)
	broadcastAxes := node.data.([]int)
	for operandAxis, outputAxis := range broadcastAxes {
		reshapedOperand.Dimensions[outputAxis] = operand.shape.Dimensions[operandAxis]
	}

	// Create broadcasting iterator: it requires operand and output shapes to have the same rank.
	iter := newBroadcastIterator(reshapedOperand, output.shape)
	dispatchBroadcastInDim.Dispatch(output.shape.DType, operand.flat, output.flat, iter)
	return output
}

var dispatchBroadcastInDim = NewDTypeDispatcher("BroadcastInDim")

func execBroadcastInDimGeneric[T SupportedTypesConstraints](params ...any) any {
	operandFlat, outputFlat, operandIterAny := params[0].([]T), params[1].([]T), params[2]
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

// IotaOp ====================================================================================================

func execIota(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	iotaAxis := node.data.(int)
	iotaSize := node.shape.Dimensions[iotaAxis]
	batchSize := 1
	repeatsSize := 1
	for axis, dim := range node.shape.Dimensions {
		if axis > iotaAxis {
			repeatsSize *= dim
		} else if axis < iotaAxis {
			batchSize *= dim
		}
	}
	dispatchIota.Dispatch(node.shape.DType, output, batchSize, iotaSize, repeatsSize)
	return output
}

var dispatchIota = NewDTypeDispatcher("Iota")

func execIotaGeneric[T PODNumericConstraints](params ...any) any {
	output, batchSize, iotaSize, repeatsSize := params[0].(*Buffer), params[1].(int), params[2].(int), params[3].(int)
	outputFlat := output.flat.([]T)
	flatIdx := 0
	var value T
	for _ = range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = T(0)
		for _ = range iotaSize {
			for _ = range repeatsSize {
				outputFlat[flatIdx] = value
				flatIdx++
			}
			value++
		}
	}
	return nil
}

func init() { dispatchIota.Register(dtypes.BFloat16, execIotaBFloat16) }

func execIotaBFloat16(params ...any) any {
	output, batchSize, iotaSize, repeatsSize := params[0].(*Buffer), params[1].(int), params[2].(int), params[3].(int)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	flatIdx := 0
	var value float32
	for _ = range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = 0
		for _ = range iotaSize {
			for _ = range repeatsSize {
				outputFlat[flatIdx] = bfloat16.FromFloat32(value)
				flatIdx++
			}
			value++
		}
	}
	return nil
}

// GatherOp ====================================================================================================

func execGather(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) *Buffer {
	operand, startIndices := inputs[0], inputs[1]
	gatherParams := node.data.(*gatherNode)
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	startIndicesIt := newGatherStartIndicesIterator(startIndices.shape, gatherParams.indexVectorAxis)

	operandBytes := operand.mutableBytes()
	outputBytes := output.mutableBytes()
	dispatchGather.Dispatch(startIndices.shape.DType,
		startIndices, startIndicesIt, operand.shape, operandBytes, output.shape, outputBytes, gatherParams)
	return output
}

type gatherStartIndicesIterator struct {
	flatIdx                              int
	prefixIdx, suffixIdx                 int
	prefixSize, suffixSize, prefixStride int
	numIndices, startIndicesStride       int
}

func newGatherStartIndicesIterator(startIndicesShape shapes.Shape, startVectorIndex int) *gatherStartIndicesIterator {
	it := &gatherStartIndicesIterator{
		prefixSize:         1,
		suffixSize:         1,
		prefixStride:       1,
		startIndicesStride: 1,
	}
	for axis, dim := range startIndicesShape.Dimensions {
		if axis < startVectorIndex {
			it.prefixSize *= dim
		} else {
			it.prefixStride *= dim
			if axis > startVectorIndex {
				it.suffixSize *= dim
				it.startIndicesStride *= dim
			}
		}
	}
	return it
}

func (it *gatherStartIndicesIterator) Next(startIndicesFlatIndices []int) (hasNext bool) {
	if it.prefixIdx == it.prefixSize {
		return false
	}
	flatIdx := it.flatIdx
	for ii := range startIndicesFlatIndices {
		startIndicesFlatIndices[ii] = flatIdx
		flatIdx += it.startIndicesStride
	}

	// Increment suffix index:
	if it.suffixSize > 1 {
		it.suffixIdx++
		it.flatIdx++
		if it.suffixIdx < it.suffixSize {
			return true
		}
		it.flatIdx -= it.suffixSize
		it.suffixIdx = 0
	}

	// Increment prefix index:
	it.prefixIdx++
	it.flatIdx += it.prefixStride
	return true
}

var dispatchGather = NewDTypeDispatcher("Gather")

// execGatherGeneric is specialized by startIndices DType: they need to be converted to int.
// The operand and output dtypes are treated as bytes.
func execGatherGeneric[T PODIntegerConstraints](params ...any) any {
	startIndicesFlat := params[0].([]T)
	startIndicesIt := params[1].(*gatherStartIndicesIterator)
	operandShape := params[2].(shapes.Shape)
	operandBytes := params[3].([]byte)
	outputShape := params[4].(shapes.Shape)
	outputBytes := params[5].([]byte)
	gatherParams := params[6].(*gatherNode)

	startIndexMap := gatherParams.startIndexMap
	it := params[2].(*gatherStartIndicesIterator)

	indirectStartIndices := make([]int, len(startIndexMap))
	operandStartIndices := make([]int, operandShape.Rank())
	outputIndices := make([]int, outputShape.Rank())
	for startIndicesIt.Next(indirectStartIndices) {
		// Find operand indices:
		for ii, axis := range startIndexMap {
			startIndexForAxis := startIndicesFlat[indirectStartIndices[ii]]
			operandStartIndices[axis] = int(startIndexForAxis)
		}

		// Find output indices:

	}
	return nil
}
