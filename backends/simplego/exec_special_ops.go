package simplego

import (
	"encoding/binary"
	"math/rand/v2"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
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
	nodeExecutors[backends.OpTypeReduceBitwiseAnd] = execReduce
	nodeExecutors[backends.OpTypeReduceBitwiseOr] = execReduce
	nodeExecutors[backends.OpTypeReduceBitwiseXor] = execReduce
	nodeExecutors[backends.OpTypeReduceLogicalAnd] = execReduce
	nodeExecutors[backends.OpTypeReduceLogicalOr] = execReduce
	nodeExecutors[backends.OpTypeReduceLogicalXor] = execReduce
	nodeExecutors[backends.OpTypeIota] = execIota
	nodeExecutors[backends.OpTypeGather] = execGather
	nodeExecutors[backends.OpTypeConcatenate] = execConcatenate
	nodeExecutors[backends.OpTypeConvertDType] = execConvertDType
	nodeExecutors[backends.OpTypeScatterMax] = execScatter
	nodeExecutors[backends.OpTypeScatterMin] = execScatter
	nodeExecutors[backends.OpTypeScatterSum] = execScatter
	nodeExecutors[backends.OpTypeSlice] = execSlice
	nodeExecutors[backends.OpTypeArgMinMax] = execArgMinMax
	nodeExecutors[backends.OpTypeReduceWindow] = execReduceWindow

	// For nodes with multiple outputs:
	multiOutputsNodeExecutors[backends.OpTypeRngBitGenerator] = execRngBitGenerator
}

// calculateStrides of a tensor assuming row-major order of the flat data.
func calculateStrides(dims []int) []int {
	rank := len(dims)
	stride := 1
	strides := make([]int, rank)
	for axis := rank - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= dims[axis]
	}
	return strides
}

// IdentityOp ====================================================================================================

// execIdentity implements the Identity op.
func execIdentity(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_ = node
	operand := inputs[0]
	if inputsOwned[0] {
		// Mark the input (operand) as consumed and return it as the output.
		inputs[0] = nil
		return operand, nil
	}

	// If the input is still in use, we make a copy.
	output := backend.getBuffer(operand.shape.DType, operand.shape.Size())
	output.shape = operand.shape
	copyFlat(output.flat, operand.flat)
	return output, nil
}

// WhereOp ====================================================================================================

// execWhere implements the Where op.
func execWhere(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
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

	fn := whereDTypeMap.Get(outputShape.DType).(func(conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer))
	fn(condition, onTrue, onFalse, output)
	return output, nil
}

var whereDTypeMap = NewDTypeMap("Where")

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

// ReshapeOp ====================================================================================================

// execReshape implements Reshape.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func execReshape(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	var output *Buffer
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil
	} else {
		output = backend.getBuffer(operand.shape.DType, operand.shape.Size())
		copyFlat(output.flat, operand.flat)
	}
	output.shape = node.shape
	return output, nil
}

// Reduce{Max,Min,Sum,Product}Op ======================================================================================

type genericReduceFn = func(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType)

func execReduce(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	reduceAxes := node.data.([]int)
	if len(reduceAxes) == 0 {
		return execIdentity(backend, node, inputs, inputsOwned)
	}
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	it := newReduceOutputIterator(operand.shape.Dimensions, reduceAxes)
	dtype := output.shape.DType

	var reduceFn genericReduceFn
	switch node.opType {
	case backends.OpTypeReduceMax:
		reduceFn = reduceMaxDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceMin:
		reduceFn = reduceMinDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceSum:
		reduceFn = reduceSumDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceProduct:
		reduceFn = reduceProductDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceBitwiseAnd:
		reduceFn = reduceBitwiseAndDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceBitwiseOr:
		reduceFn = reduceBitwiseOrDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceBitwiseXor:
		reduceFn = reduceBitwiseXorDTypeMap.Get(dtype).(genericReduceFn)
	case backends.OpTypeReduceLogicalAnd:
		// Logical reduction only works on boolean variables, so there is no need for a generic implementation.
		reduceFn = execReduceLogicalAnd
	case backends.OpTypeReduceLogicalOr:
		// Logical reduction only works on boolean variables, so there is no need for a generic implementation.
		reduceFn = execReduceLogicalOr
	case backends.OpTypeReduceLogicalXor:
		// Logical reduction only works on boolean variables, so there is no need for a generic implementation.
		reduceFn = execReduceLogicalXor
	default:
		return nil, errors.Errorf("unsupported reduce op %s", node.opType)
	}
	reduceFn(operand, output, it, dtype)
	return output, nil
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
			// Skip the reducing axes and leave stride as 0.
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

		// Return to the start of the current axis and move to the next axis.
		it.perAxisIdx[axis] = 0
		it.flatIdx -= it.perAxisStride[axis] * it.dimensions[axis]
	}
	return returnIdx
}

var reduceMaxDTypeMap = NewDTypeMap("ReduceMax")

// execReduceMaxGeneric: use reduceMaxDTypeMap to call it.
func execReduceMaxGeneric[T PODNumericConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {

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

func init() { reduceMaxDTypeMap.Register(dtypes.BFloat16, execReduceMaxBFloat16) }

// execReduceMaxBFloat16: use reduceMaxDTypeMa to call it.
func execReduceMaxBFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

var reduceMinDTypeMap = NewDTypeMap("ReduceMin")

func execReduceMinGeneric[T PODNumericConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

func init() { reduceMinDTypeMap.Register(dtypes.BFloat16, execReduceMinBFloat16) }

func execReduceMinBFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

var reduceSumDTypeMap = NewDTypeMap("ReduceSum")

func execReduceSumGeneric[T PODNumericConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

func init() { reduceSumDTypeMap.Register(dtypes.BFloat16, execReduceSumBFloat16) }

func execReduceSumBFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

var reduceProductDTypeMap = NewDTypeMap("ReduceProduct")

func execReduceProductGeneric[T PODNumericConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

func init() { reduceProductDTypeMap.Register(dtypes.BFloat16, execReduceProductBFloat16) }

func execReduceProductBFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
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

var (
	reduceBitwiseAndDTypeMap = NewDTypeMap("ReduceBitwiseAnd")
	reduceBitwiseOrDTypeMap  = NewDTypeMap("ReduceBitwiseOr")
	reduceBitwiseXorDTypeMap = NewDTypeMap("ReduceBitwiseXor")
)

func execReduceBitwiseAndGeneric[T PODIntegerConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	initialValue := ^T(0)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] & value
	}
}

func execReduceBitwiseOrGeneric[T PODIntegerConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	initialValue := T(0)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] | value
	}
}

func execReduceBitwiseXorGeneric[T PODIntegerConstraints](operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	initialValue := T(0)
	outputFlat := output.flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] ^ value
	}
}

func execReduceLogicalAnd(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	outputFlat := output.flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = true
	}

	operandFlat := operand.flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] && value
	}
}

func execReduceLogicalOr(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	outputFlat := output.flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = false
	}

	operandFlat := operand.flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] || value
	}
}

func execReduceLogicalXor(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with 1.
	outputFlat := output.flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = false
	}

	operandFlat := operand.flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.next()
		outputFlat[outputIdx] = outputFlat[outputIdx] != value // a != b is the same as Xor(a,b).
	}
}

// TransposeOp ====================================================================================================

// execTranspose implements Transpose.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func execTranspose(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	permutations := node.data.([]int)
	_ = inputsOwned // We don't reuse the inputs.

	// We can't write to the same buffer we read from because it's not done with swaps.
	output := backend.getBuffer(operand.shape.DType, operand.shape.Size())
	output.shape = node.shape
	it := newTransposeIterator(operand.shape, permutations)
	dtype := node.shape.DType
	transposeFn := transposeDTypeMap.Get(dtype).(func(operand, output *Buffer, it *transposeIterator))
	transposeFn(operand, output, it)
	return output, nil
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

	// First, calculate strides on the output.
	stridesOnOutput := make([]int, rank)
	stride := 1
	reversePermutations := make([]int, rank)
	for outputAxis := rank - 1; outputAxis >= 0; outputAxis-- {
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

func (it *transposeIterator) next() int {
	// Store current flatIdx first
	nextFlatIdx := it.flatIdx

	// Cache rank to avoid repeated len() calls
	rank := len(it.perAxisIdx)

	// Use local variables for array access to avoid repeated indirection
	perAxisIdx := it.perAxisIdx
	perAxisStrides := it.perAxisStrides
	dimensions := it.dimensions

	// Handle remaining axes only when needed
	for axis := rank - 1; axis >= 0; axis-- {
		perAxisIdx[axis]++
		it.flatIdx += perAxisStrides[axis]
		if perAxisIdx[axis] < dimensions[axis] {
			// We are done.
			return nextFlatIdx
		}
		perAxisIdx[axis] = 0
		it.flatIdx -= perAxisStrides[axis] * dimensions[axis]
	}

	return nextFlatIdx
}

var transposeDTypeMap = NewDTypeMap("Transpose")

func execTransposeGeneric[T SupportedTypesConstraints](operand, output *Buffer, it *transposeIterator) {
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	for _, value := range operandFlat {
		outputFlat[it.next()] = value
	}
}

// BroadcastOp ====================================================================================================

func execBroadcast(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_ = inputsOwned // We don't reuse the inputs.
	operand := inputs[0]
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
	prefixDims := node.data.([]int)
	repeats := 1
	for _, dim := range prefixDims {
		repeats *= dim
	}
	dispatchBroadcast.Dispatch(node.shape.DType, operand.flat, output.flat, repeats)
	return output, nil
}

var dispatchBroadcast = NewDTypeDispatcher("Broadcast")

func execBroadcastGeneric[T SupportedTypesConstraints](params ...any) any {
	operandFlat, outputFlat, repeats := params[0].([]T), params[1].([]T), params[2].(int)
	pos := 0
	for range repeats {
		copy(outputFlat[pos:], operandFlat)
		pos += len(operandFlat)
	}
	return nil
}

// BroadcastInDimsOp ====================================================================================================

func execBroadcastInDim(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_ = inputsOwned // We don't reuse the inputs.
	operand := inputs[0]
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape

	// Special case: if operand is a scalar, we just pass a nil iterator.
	if operand.shape.Size() == 1 {
		dispatchBroadcastInDim.Dispatch(output.shape.DType, operand.flat, output.flat, nil)
		return output, nil
	}

	// Reshape operand shape: same dimension as the operand on the corresponding axes, 1 elsewhere.
	// Notice they must have the same size; hence the flat data doesn't change.
	reshapedOperand := shapes.Make(operand.shape.DType)
	reshapedOperand.Dimensions = make([]int, output.shape.Rank())
	xslices.FillSlice(reshapedOperand.Dimensions, 1)
	broadcastAxes := node.data.([]int)
	for operandAxis, outputAxis := range broadcastAxes {
		reshapedOperand.Dimensions[outputAxis] = operand.shape.Dimensions[operandAxis]
	}

	// Create broadcasting the iterator: it requires operand and output shapes to have the same rank.
	iter := newBroadcastIterator(reshapedOperand, output.shape)
	dispatchBroadcastInDim.Dispatch(output.shape.DType, operand.flat, output.flat, iter)
	return output, nil
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

func execIota(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_, _ = inputs, inputsOwned // There are no inputs.
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
	return output, nil
}

var dispatchIota = NewDTypeDispatcher("Iota")

func execIotaGeneric[T PODNumericConstraints](params ...any) any {
	output, batchSize, iotaSize, repeatsSize := params[0].(*Buffer), params[1].(int), params[2].(int), params[3].(int)
	outputFlat := output.flat.([]T)
	flatIdx := 0
	var value T
	for range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = T(0)
		for range iotaSize {
			for range repeatsSize {
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
	for range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = 0
		for range iotaSize {
			for range repeatsSize {
				outputFlat[flatIdx] = bfloat16.FromFloat32(value)
				flatIdx++
			}
			value++
		}
	}
	return nil
}

// GatherOp ====================================================================================================

func execGather(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_ = inputsOwned // We don't reuse the inputs.
	operand, startIndices := inputs[0], inputs[1]
	gatherParams := node.data.(*gatherNode)
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape

	// Where to read/write the data.
	operandBytes := operand.mutableBytes()
	outputBytes := output.mutableBytes()

	// Outer-loop: loop over the start indices and outputBytesIdx to gather from:
	gatherIt := newGatherIterator(
		startIndices.shape, gatherParams.indexVectorAxis,
		output.shape, gatherParams.offsetOutputAxes)
	indirectStartIndices := make([]int, len(gatherParams.startIndexMap))
	operandShape := operand.shape
	operandRank := operandShape.Rank()
	dataSize := operandShape.DType.Size()
	operandStartIndices := make([]int, operandRank)

	// Inner-loop preparation: loop over the slices to copy given the starting indices.
	operandByteStrides := make([]int, operandRank)
	{
		stride := dataSize
		for axis := operandRank - 1; axis >= 0; axis-- {
			operandByteStrides[axis] = stride
			stride *= operandShape.Dimensions[axis]
		}
	}
	//fmt.Printf("operandByteStrides: %v\n", operandByteStrides)
	slicesSize := 1
	for _, sliceDim := range gatherParams.sliceSizes {
		slicesSize *= sliceDim
	}

	// For the inner-loop, calculate the strides for the output as we traverse the slices.
	sliceOutputBytesStride := make([]int, operandRank)
	{
		// - We first need to map each slice axis to the corresponding output axis: it doesn't matter if the slice size is 1,
		//   since these are not incremented.
		mapSliceToOutputAxes := make([]int, operandRank)
		offsetOutputAxesIdx := 0
		collapsedAxes := sets.MakeWith(gatherParams.collapsedSlicesAxes...)
		for sliceAxis := range operandRank {
			if collapsedAxes.Has(sliceAxis) {
				// Collapsed, we only care about the offset axes.
				continue
			}
			mapSliceToOutputAxes[sliceAxis] = gatherParams.offsetOutputAxes[offsetOutputAxesIdx]
			offsetOutputAxesIdx++
		}
		// Now we copy over the strides calculated for the gatherIterator.
		for sliceAxis := range operandRank {
			if collapsedAxes.Has(sliceAxis) {
				// Collapsed, we only care about the offset axes.
				continue
			}
			outputAxis := mapSliceToOutputAxes[sliceAxis]
			sliceOutputBytesStride[sliceAxis] = gatherIt.outputStrides[outputAxis]
		}
	}

	dispatchGather.Dispatch(startIndices.shape.DType,
		gatherParams,
		operandBytes, outputBytes, dataSize,
		gatherIt, indirectStartIndices, startIndices.flat,
		operandStartIndices, operandByteStrides,
		slicesSize, sliceOutputBytesStride,
	)
	return output, nil
}

var dispatchGather = NewDTypeDispatcher("Gather")

// execGatherGeneric is specialized by startIndices DType: they need to be converted to int.
// The operand and output dtypes are treated as bytes.
func execGatherGeneric[T PODIntegerConstraints](params ...any) any {
	paramsIdx := 0
	nextParam := func() any {
		ret := params[paramsIdx]
		paramsIdx++
		return ret
	}

	gatherParams := nextParam().(*gatherNode)
	operandBytes := nextParam().([]byte)
	outputBytes := nextParam().([]byte)
	dataSize := nextParam().(int)
	gatherIt := nextParam().(*gatherIterator)
	indirectStartIndices := nextParam().([]int)
	startIndicesFlat := nextParam().([]T) // This is specialized in this generic implementation.
	operandStartIndices := nextParam().([]int)
	operandByteStrides := nextParam().([]int)
	slicesSize := nextParam().(int)
	sliceOutputBytesStride := nextParam().([]int)

	sliceSizes := gatherParams.sliceSizes
	operandRank := len(sliceSizes)
	startIndexMap := gatherParams.startIndexMap

	// Outer-loop: loop over the start indices and outputBytesIdx to gather from.
	var operandBytesIdx, outputBytesIdx int
	sliceIndices := make([]int, operandRank)
	for gatherIt.Next(indirectStartIndices, &outputBytesIdx) {
		// Find operand indices:
		for ii, axis := range startIndexMap {
			startIndexForAxis := startIndicesFlat[indirectStartIndices[ii]]
			operandStartIndices[axis] = int(startIndexForAxis)
		}
		operandBytesIdx = 0
		for axis, idx := range operandStartIndices {
			operandBytesIdx += operandByteStrides[axis] * idx
		}
		//fmt.Printf("\toperand: start=%v, idx(bytes)=%d\n", operandStartIndices, operandBytesIdx)
		//fmt.Printf("\toutput: idx(bytes)=%d\n", outputBytesIdx)

		// Traverse sliceSizes in the operand copying over the result.
		for ii := range sliceIndices {
			sliceIndices[ii] = 0
		}
		for range slicesSize {
			// TODO: copy more than one element (dataSize) at a time, when possible.
			copy(outputBytes[outputBytesIdx:outputBytesIdx+dataSize],
				operandBytes[operandBytesIdx:operandBytesIdx+dataSize])

			// Increment index in the operand.
			for axis := operandRank - 1; axis >= 0; axis-- {
				if sliceSizes[axis] == 1 {
					// We don't iterate over sliceSizes of 1.
					continue
				}
				sliceIndices[axis]++
				operandBytesIdx += operandByteStrides[axis]
				outputBytesIdx += sliceOutputBytesStride[axis]
				if sliceIndices[axis] != sliceSizes[axis] {
					// Finished incrementing.
					break
				}

				// Rewind the current axis before trying to increment next.
				sliceIndices[axis] = 0
				operandBytesIdx -= operandByteStrides[axis] * sliceSizes[axis]
				outputBytesIdx -= sliceOutputBytesStride[axis] * sliceSizes[axis]
			}
		}
	}
	return nil
}

// gatherIterator controls iteration 2 sets of indices, that move together at each iteration.
//
//   - A. startIndices tensor, which points where to get the data from in the operand.
//   - B. the output tensor, where to store the data. It iterates over the bytes, and yields the byte position of the data.
//
// The startIndices tensor iterator (A) is split into:
//
//  1. "prefix indices": batch axes before the startVectorIndex (for startIndices)
//  2. "suffix indices": batch axes that come after the startVectorIndex (for startIndices)
//
// The output iterator (B) only iterate over the batch dimensions: the offset dimensions are all part of the slice
// that is gathered (copied over) in one go. Because the offsetOutputAxes can be interleaved with the batch dimensions
// we have to keep separate indices for each axis.
// TODO: reshape and merge axes in startIndices and operand before the gather, and later reshape back the output to separate them.
type gatherIterator struct {
	prefixIdx, suffixIdx   int
	prefixSize, suffixSize int

	// startIndices state.
	startIndicesFlatIdx      int
	startIndicesPrefixStride int

	// outputIndices state.
	outputBytesIdx     int
	outputIndices      []int // Index for each axis.
	outputDimsForBatch []int // Set to 1 for the offset axes, we are only iterating over the batch indices.
	outputStrides      []int // Calculated with the offset axes.
}

func newGatherIterator(startIndicesShape shapes.Shape, startVectorIndex int, outputShape shapes.Shape, offsetOutputAxes []int) *gatherIterator {
	it := &gatherIterator{
		prefixSize: 1,
		suffixSize: 1,

		startIndicesPrefixStride: 1,

		outputIndices:      make([]int, outputShape.Rank()),
		outputDimsForBatch: slices.Clone(outputShape.Dimensions),
		outputStrides:      make([]int, outputShape.Rank()),
	}

	// Initialize for startIndices.
	for axis, dim := range startIndicesShape.Dimensions {
		if axis < startVectorIndex {
			it.prefixSize *= dim
		} else {
			it.startIndicesPrefixStride *= dim
			if axis > startVectorIndex {
				it.suffixSize *= dim
			}
		}
	}

	// Initialize for output.
	dataSize := outputShape.DType.Size()
	outputStride := dataSize
	for axis := outputShape.Rank() - 1; axis >= 0; axis-- {
		it.outputStrides[axis] = outputStride
		outputStride *= outputShape.Dimensions[axis]
	}
	for _, outputAxis := range offsetOutputAxes {
		it.outputDimsForBatch[outputAxis] = 1 // We don't iterate over these.
	}
	return it
}

func (it *gatherIterator) Next(startIndicesFlatIndices []int, outputByteIdx *int) (hasNext bool) {
	// iterate on output bytes:
	*outputByteIdx = it.outputBytesIdx
	for axis := len(it.outputDimsForBatch) - 1; axis >= 0; axis-- {
		if it.outputDimsForBatch[axis] == 1 {
			// This axis has dimension 1, so it never changes.
			// TODO: during initialization remove this dimensions from outputDimsForBatch, outputIndices, etc.
			continue
		}
		it.outputIndices[axis]++
		it.outputBytesIdx += it.outputStrides[axis]
		if it.outputIndices[axis] < it.outputDimsForBatch[axis] {
			// If we haven't reached the end of the axis, we are done.
			break
		}
		if axis == 0 {
			// This is the last iteration.
			break
		}

		// Go back to the start of the current index.
		it.outputIndices[axis] = 0
		it.outputBytesIdx -= it.outputStrides[axis-1] // == it.outputStrides[axis] * it.outputDimsForBatch[axis]
	}

	// iterate on startIndices:
	if it.prefixIdx == it.prefixSize {
		return false
	}
	startIndicesFlatIdx := it.startIndicesFlatIdx
	for ii := range startIndicesFlatIndices {
		startIndicesFlatIndices[ii] = startIndicesFlatIdx
		startIndicesFlatIdx += it.suffixSize
	}
	if it.suffixSize > 1 {
		it.suffixIdx++
		it.startIndicesFlatIdx++
		if it.suffixIdx < it.suffixSize {
			return true
		}
		it.startIndicesFlatIdx -= it.suffixSize
		it.suffixIdx = 0
	}
	// Increment prefix index:
	it.prefixIdx++
	it.startIndicesFlatIdx += it.startIndicesPrefixStride
	return true
}

// ConcatenateOp ====================================================================================================

// execConcatenate implements the Concatenate op using direct byte copying with offsets and strides.
func execConcatenate(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	axis := node.data.(int) // Renamed from dimension
	outputShape := node.shape
	dtype := outputShape.DType
	elemSize := dtype.Size()
	rank := outputShape.Rank()
	_ = inputsOwned // We don't reuse the inputs.

	// Allocate output buffer.
	output := backend.getBuffer(dtype, outputShape.Size())
	output.shape = outputShape
	outputBytes := output.mutableBytes()

	// Calculate the size of the blocks before and after the concatenation axis.
	outerBlockSize := 1 // Number of independent blocks to copy
	for i := 0; i < axis; i++ {
		outerBlockSize *= outputShape.Dimensions[i]
	}
	innerBlockSize := 1 // Size of the innermost contiguous block (in elements)
	for i := axis + 1; i < rank; i++ {
		innerBlockSize *= outputShape.Dimensions[i]
	}
	innerBlockBytes := innerBlockSize * elemSize

	// Total size in bytes of one full "row" along the concatenation axis in the output.
	// This is the stride needed to jump from one outer block to the next in the output.
	outputConcatAxisStrideBytes := outputShape.Dimensions[axis] * innerBlockBytes

	// Current offset in bytes along the concatenation axis *within* an outer block in the output buffer.
	// This accumulates as we process each input tensor.
	outputAxisOffsetBytes := 0

	for _, inputBuf := range inputs {
		inputShape := inputBuf.shape
		inputDims := inputShape.Dimensions
		inputBytes := inputBuf.mutableBytes() // Use mutableBytes() for input

		// Size of the concatenation axis for this specific input.
		inputConcatAxisSize := inputDims[axis]

		// Total size in bytes to copy from this input *per outer block*.
		inputBlockBytes := inputConcatAxisSize * innerBlockBytes

		// Iterate through all outer dimension blocks.
		for outerIdx := 0; outerIdx < outerBlockSize; outerIdx++ {
			// Calculate the starting byte position for the current outer block in the input.
			// This is simply the outer block index times the size of the block to copy for this input.
			inputStartOffset := outerIdx * inputBlockBytes

			// Calculate the starting byte position for the current outer block in the output.
			// This is the outer block index times the total output stride along the concat axis,
			// plus the accumulated offset from previous inputs along the concat axis.
			outputStartOffset := outerIdx*outputConcatAxisStrideBytes + outputAxisOffsetBytes

			// Copy the relevant block of bytes for the current outer block.
			copy(outputBytes[outputStartOffset:outputStartOffset+inputBlockBytes], inputBytes[inputStartOffset:inputStartOffset+inputBlockBytes])
		}

		// Update the offset for the next input along the concatenation axis.
		outputAxisOffsetBytes += inputBlockBytes
	}

	return output, nil
}

// ConvertDType ====================================================================================================

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

func execConvertDTypeGeneric[FromT PODNumericConstraints, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		outputFlat[idx] = ToT(value)
	}
}

func execConvertDTypeFromBFloat16[FromT bfloat16.BFloat16, ToT PODNumericConstraints](operand, output *Buffer) {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]ToT)
	for idx, value := range operandFlat {
		outputFlat[idx] = ToT(value.Float32())
	}
}

func execConvertDTypeToBFloat16[FromT PODNumericConstraints, ToT bfloat16.BFloat16](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	for idx, value := range operandFlat {
		outputFlat[idx] = bfloat16.FromFloat32(float32(value))
	}
}

func execConvertDTypeFromBool[FromT bool, ToT PODNumericConstraints](operand, output *Buffer) {
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

func execConvertDTypeToBool[FromT PODNumericConstraints, ToT bool](operand, output *Buffer) {
	operandFlat := operand.flat.([]FromT)
	outputFlat := output.flat.([]bool)
	for idx, value := range operandFlat {
		outputFlat[idx] = value != 0
	}
}

func init() {
	// Manually register bool x bfloat16 convertion functions.
	convertDTypePairMap.Register(dtypes.BFloat16, dtypes.Bool, execConvertDTypeBFloat16ToBool)
	convertDTypePairMap.Register(dtypes.Bool, dtypes.BFloat16, execConvertDTypeBoolToBFloat16)
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

// Scatter{Max,Min,Sum}Op ==========================================================================================

// execScatter implements the Scatter operation (Max, Min, Sum variants).
func execScatter(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	scatterParams, ok := node.data.(*scatterNode)
	if !ok {
		return nil, errors.Errorf("internal error: node.data for Scatter op is not *scatterData, but %T", node.data)
	}

	// Output starts as a copy of the operand.
	// We might be able to reuse the operand buffer if it's owned.
	var output *Buffer
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil // Mark operand as consumed.
	} else {
		output = backend.cloneBuffer(operand) // Creates a new buffer with copied data.
	}
	output.shape = node.shape // Output shape is the same as operand shape.

	// Dispatch to a type-specific scatter loop based on the operation type.
	dtype := output.shape.DType
	type scatterFnT = func(opType backends.OpType, output, indices, updates *Buffer, scatterParams *scatterNode) error
	scatterFn := scatterDTypeMap.Get(dtype).(scatterFnT)
	err := scatterFn(node.opType, output, indices, updates, scatterParams)
	if err != nil {
		return nil, err
	}
	return output, nil
}

var scatterDTypeMap = NewDTypeMap("ScatterMax")

// execScatterGeneric assumes the operand is already copied to the output.
func execScatterGeneric[T SupportedTypesConstraints](opType backends.OpType, output, indices, updates *Buffer, scatterParams *scatterNode) error {
	// Get combineFn for operand's dtype.
	dtype := output.shape.DType
	type combineFnT = func(a, b T) T
	var combineFn combineFnT
	switch opType {
	case backends.OpTypeScatterMax:
		combineFn = combineMaxDTypeMap.Get(dtype).(combineFnT)
	case backends.OpTypeScatterMin:
		combineFn = combineMinDTypeMap.Get(dtype).(combineFnT)
	case backends.OpTypeScatterSum:
		combineFn = combineSumDTypeMap.Get(dtype).(combineFnT)
	default:
		return errors.Errorf("unsupported scatter op type %q", opType)
	}
	_ = combineFn

	outputShape := output.shape
	outputFlat := output.flat.([]T)
	indicesFlat := indices.flat
	updatesShape := updates.shape
	updatesFlat := updates.flat.([]T)

	// Initialize gather of the scatter indices.
	indicesShape := indices.shape
	deferenceIndicesFn := dereferenceIntsDTypeMap.Get(indicesShape.DType).(func(flat any, in, out []int))
	_, _ = indicesFlat, deferenceIndicesFn
	indicesIt := newSubIndicesIterator(indices.shape, scatterParams.indexVectorAxis)
	indexVectorStride := 1
	indexVectorSize := 1
	if scatterParams.indexVectorAxis != indicesShape.Rank() {
		indexVectorSize = indices.shape.Dimensions[scatterParams.indexVectorAxis]
		indexVectorStride = indicesIt.PerAxisStride[scatterParams.indexVectorAxis]
	}
	indirectScatterIndices := make([]int, indexVectorSize)
	elemIndices := make([]int, indexVectorSize)
	//fmt.Printf("\tindexVectorSize=%d, indexVectorStride=%d\n", numBatchAxes, indexVectorStride)

	// Initialize iterator over the updates:
	updatesIt := newSubIndicesIterator(updatesShape, scatterParams.updateWindowAxes...)
	numBatchAxes := indicesShape.Rank() - 1
	if scatterParams.indexVectorAxis == indicesShape.Rank() {
		numBatchAxes++
	}
	updatesBatchAxes := make([]int, 0, numBatchAxes)
	updatesWindowAxesSet := sets.MakeWith(scatterParams.updateWindowAxes...)
	for axis := range updatesShape.Rank() {
		if !updatesWindowAxesSet.Has(axis) {
			updatesBatchAxes = append(updatesBatchAxes, axis)
		}
	}
	innerUpdatesIt := newSubIndicesIterator(updatesShape, updatesBatchAxes...)

	// Initialize an inner iterator over the output:
	innerOutputIt := newSubIndicesIterator(outputShape, scatterParams.insertedWindowAxes...)

	// Outer-loop: range over the pointed indices
	for {
		// Find scatter indices -> where the values are going to be combined in the output:
		flatIndirectIndex := indicesIt.FlatIdx
		for ii := range indexVectorSize {
			indirectScatterIndices[ii] = flatIndirectIndex
			flatIndirectIndex += indexVectorStride
		}
		deferenceIndicesFn(indicesFlat, indirectScatterIndices, elemIndices)
		//fmt.Printf("\tindices%v = indices.flat[%d] = %v\n", indicesIt.PerAxisIdx, indicesIt.FlatIdx, elemIndices)

		// Prepare innerOutputIt to start from the position set indices.
		for axis := range innerOutputIt.PerAxisIdx {
			innerOutputIt.PerAxisIdx[axis] = 0
		}
		innerOutputIt.FlatIdx = 0
		for scatterAxis, idx := range elemIndices {
			outputAxis := scatterParams.scatterAxesToOperandAxes[scatterAxis]
			innerOutputIt.PerAxisIdx[outputAxis] = idx
			innerOutputIt.FlatIdx += idx * innerOutputIt.PerAxisStride[outputAxis]
		}

		// Prepare innerUpdatesIt to start from the indices in the updatesIt.
		innerUpdatesIt.FlatIdx = updatesIt.FlatIdx
		for ii, idx := range updatesIt.PerAxisIdx {
			innerUpdatesIt.PerAxisIdx[ii] = idx
		}

		// Inner-loop: combine slice (window) of update into output.
		for {
			outputIdx := innerOutputIt.FlatIdx
			updatesIdx := innerUpdatesIt.FlatIdx
			//fmt.Println("\t\tCombine:")
			//fmt.Printf("\t\t- updates%v (updatesFlat[%d])=%v\n", innerUpdatesIt.PerAxisIdx, updatesIdx, updatesFlat[updatesIdx])
			//fmt.Printf("\t\t-  output%v (outputFlat[%d])=%v\n", innerOutputIt.PerAxisIdx, outputIdx, outputFlat[outputIdx])
			outputFlat[outputIdx] = combineFn(outputFlat[outputIdx], updatesFlat[updatesIdx])
			//fmt.Printf("\t\t- result=%v\n", outputFlat[outputIdx])
			if !innerUpdatesIt.Increment() {
				break
			}
			innerOutputIt.Increment()
		}

		// Next in indices:
		if !indicesIt.Increment() {
			break
		}
		updatesIt.Increment()
	}
	return nil
}

type subIndicesIterator struct {
	// FlatIdx is the current flat index to the shape.
	FlatIdx int

	// PerAxisIdx is the current indices in the shape.
	PerAxisIdx []int

	PerAxisSize   []int
	PerAxisStride []int
}

func newSubIndicesIterator(shape shapes.Shape, skipAxes ...int) *subIndicesIterator {
	rank := shape.Rank()
	it := &subIndicesIterator{
		PerAxisIdx:  make([]int, rank),
		PerAxisSize: slices.Clone(shape.Dimensions),
	}
	it.PerAxisStride = calculateStrides(shape.Dimensions)
	for _, axis := range skipAxes {
		if axis < rank {
			// Set size for axis we don't want to iterate over to 1.
			it.PerAxisSize[axis] = 1
		}
	}
	return it
}

// Increment indices. It returns true if the new index is still valid, or false if it reached the end.
func (it *subIndicesIterator) Increment() bool {
	if it.FlatIdx < 0 {
		return false
	}
	rank := len(it.PerAxisSize)
	for axis := rank - 1; axis >= 0; axis-- {
		if it.PerAxisSize[axis] == 1 {
			continue
		}
		it.PerAxisIdx[axis]++
		it.FlatIdx += it.PerAxisStride[axis]
		if it.PerAxisIdx[axis] < it.PerAxisSize[axis] {
			return true
		}

		// We are going to move to the next axis.
		if axis == 0 {
			break
		}
		it.PerAxisIdx[axis] = 0
		it.FlatIdx -= it.PerAxisStride[axis-1] // Rewind FlatIdx to start of the current axis.
	}

	// Reached end.
	it.FlatIdx = -1
	return false
}

var dereferenceIntsDTypeMap = NewDTypeMap("Scatter Indices")

func dereferenceIntsGeneric[T PODIntegerConstraints](flatAny any, indicesIn, indicesOut []int) {
	flat := flatAny.([]T)
	for ii, index := range indicesIn {
		indicesOut[ii] = int(flat[index])
	}
}

var (
	combineMaxDTypeMap = NewDTypeMap("Max(a, b) for ScatterMax")
	combineMinDTypeMap = NewDTypeMap("Min(a, b) for ScatterMin")
	combineSumDTypeMap = NewDTypeMap("Sum(a, b) for ScatterSum")
)

func init() {
	combineMaxDTypeMap.Register(dtypes.BFloat16, combineForScatterMaxBFloat16)
	combineMinDTypeMap.Register(dtypes.BFloat16, combineForScatterMinBFloat16)
	combineSumDTypeMap.Register(dtypes.BFloat16, combineForScatterSumBFloat16)
}

func combineForScatterMaxGeneric[T PODNumericConstraints](a, b T) T {
	return max(a, b)
}

func combineForScatterMaxBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(max(a.Float32(), b.Float32()))
}

func combineForScatterMinGeneric[T PODNumericConstraints](a, b T) T {
	return min(a, b)
}

func combineForScatterMinBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(min(a.Float32(), b.Float32()))
}

func combineForScatterSumGeneric[T PODNumericConstraints](a, b T) T {
	return a + b
}

func combineForScatterSumBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(a.Float32() + b.Float32())
}

// SliceOp ========================================================================================================

// execSlice is the executor function registered for backends.OpTypeSlice.
func execSlice(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	sliceParams, ok := node.data.(*sliceNode)
	if !ok {
		// Assuming node.data holds the necessary slice parameters.
		// If Builder.Slice stores data differently, this needs adjustment.
		return nil, errors.Errorf("internal error: node.data for Slice op is not *sliceNode, but %T", node.data)
	}

	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape

	// Dispatch to the generic implementation based on DType.
	// Note: limits are not used in the generic exec function but passed for potential future use or consistency.
	fn := sliceDTypeMap.Get(node.shape.DType).(func(operand, output *Buffer, params *sliceNode))
	fn(operand, output, sliceParams)
	return output, nil
}

var sliceDTypeMap = NewDTypeMap("Slice")

// execSliceGeneric implements the actual slice data copying. It is called via sliceDTypeMap.Dispatch.
// It iterates through the output buffer coordinates, calculates the corresponding coordinate
// in the operand buffer using starts and strides, and copies the value.
func execSliceGeneric[T SupportedTypesConstraints](operand, output *Buffer, params *sliceNode) {
	rank := operand.shape.Rank()
	outputFlat := output.flat.([]T)
	operandFlat := operand.flat.([]T)

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

// RngBitGenerator ====================================================================================================

// execRngBitGenerator is the executor function registered for backends.OpTypeRngBitGenerator.
func execRngBitGenerator(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	state := inputs[0]
	stateFlat := state.flat.([]uint64)

	// Reserved outputs:
	rngData := backend.getBuffer(node.multiOutputsShapes[1].DType, node.multiOutputsShapes[1].Size())
	rngData.shape = node.multiOutputsShapes[1].Clone()
	rngDataBytes := rngData.mutableBytes()

	// Generate random using rand/v2:
	rng := rand.NewPCG(stateFlat[0], stateFlat[1]) // Use state and increment as seed
	var randomBits uint64
	for idx := range rngDataBytes {
		if idx%8 == 0 {
			randomBits = rng.Uint64()
		}
		// Take one byte from the randomBits.
		rngDataBytes[idx] = byte(randomBits & 0xFF)
		randomBits = randomBits >> 8
	}

	// Update state output - PCG internal state after generating random bytes
	if inputsOwned[0] {
		// We re-use the current state.
		inputs[0] = nil
	} else {
		state.shape = node.multiOutputsShapes[0]
		state = backend.getBuffer(state.shape.DType, state.shape.Size())
	}
	stateFlat = state.flat.([]uint64)

	// See details on Go source code src/math/rand/v2/pcg.go:
	rngState, err := rng.MarshalBinary()
	if err != nil {
		panic(errors.Wrapf(err, "cannot update RngBitGenerator state"))
	}
	if len(rngState) != 20 && string(rngState[:4]) != "pcg:" {
		return nil, errors.Errorf("format of PCG random number generator changed (we got %d bytes starting with %q, we wanted 20 and starting with the string 'pcg:'), pls open an issue in GoMLX", rngState[:4], len(rngState))
	}
	stateFlat[0] = binary.LittleEndian.Uint64(rngState[4 : 4+8])
	stateFlat[1] = binary.LittleEndian.Uint64(rngState[4+8 : 4+16])
	return []*Buffer{state, rngData}, nil
}

// execArgMinMax ====================================================================================================

const MaxArgMinMaxReductionSize = 0x8000_0000

// execArgMinMax is the executor function registered for backends.OpTypeArgMinMax.
func execArgMinMax(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	reduceAxis := node.data.(*argMinMaxNode).axis
	isMin := node.data.(*argMinMaxNode).isMin
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape

	// There are 3 sizes to iterate over: before and after the reduceAxis, and the size (dimension) of the reduced axis itself.
	operandDims := operand.shape.Dimensions
	operandRank := len(operandDims)
	suffixSize := 1
	for axis := reduceAxis + 1; axis < operandRank; axis++ {
		suffixSize *= operandDims[axis]
	}
	prefixSize := 1
	for axis := range reduceAxis {
		prefixSize *= operand.shape.Dimensions[axis]
	}
	reduceSize := operandDims[reduceAxis]
	if reduceSize >= MaxArgMinMaxReductionSize {
		// If we need larger, change buildArgMinMax to use int64 instead of int32.
		return nil, errors.Errorf("ArgMaxMin implementation only supports reduction on dimensions < %d, got operand shaped %s and reduce axis is %d",
			MaxArgMinMaxReductionSize, operand.shape, reduceAxis)
	}

	// Instantiate the function to copy over results from ints:
	buildCopyIntsFn := argMinMaxCopyIntsDTypeMap.Get(output.shape.DType).(func(output *Buffer) func(flatIdx int, values []int32))
	copyIntsFn := buildCopyIntsFn(output)

	// Dispatch to the generic implementation based on DType.
	argMinMaxFn := argMinMaxDTypeMap.Get(operand.shape.DType).(func(backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize, suffixSize int, isMin bool))
	argMinMaxFn(backend, operand, copyIntsFn, prefixSize, reduceSize, suffixSize, isMin)
	return output, nil
}

var (
	argMinMaxDTypeMap         = NewDTypeMap("ArgMinMaxRun")
	argMinMaxCopyIntsDTypeMap = NewDTypeMap("ArgMinMaxCopyInts")
)

// buildArgMinMaxCopyIntsFn creates a "copyInts" function to copy the given values starting at the flatIdx to
// the output buffer.
func buildArgMinMaxCopyIntsFn[T PODIntegerConstraints](output *Buffer) func(flatIdx int, values []int32) {
	outputFlat := output.flat.([]T)
	return func(flatIdx int, values []int32) {
		for _, value := range values {
			outputFlat[flatIdx] = T(value)
			flatIdx++
		}
	}
}

func execArgMinMaxGeneric[T PODNumericConstraints](
	backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize, suffixSize int, isMin bool) {
	operandFlat := operand.flat.([]T)

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer := backend.getBuffer(operand.shape.DType, suffixSize)
	currentBest := currentBestBuffer.flat.([]T)
	currentArgBestBuffer := backend.getBuffer(dtypes.Int32, suffixSize)
	currentArgBest := currentArgBestBuffer.flat.([]int32)

	outputFlatIdx := 0
	operandFlatIdx := 0
	for range prefixSize {
		// Initialize the current best with the first element of the reduced axis:
		for suffixIdx := range suffixSize {
			currentBest[suffixIdx] = operandFlat[operandFlatIdx]
			currentArgBest[suffixIdx] = 0
			operandFlatIdx++
		}

		// Iterate over the rest of the elements of reduce axis:
		if isMin {
			// ArgMin
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx]
					operandFlatIdx++
					operandValueIsNaN := operandValue != operandValue
					if operandValue < currentBest[suffixIdx] || operandValueIsNaN {
						currentBest[suffixIdx] = operandValue
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
				}
			}
		} else {
			// ArgMax
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx]
					operandFlatIdx++
					operandValueIsNaN := operandValue != operandValue
					if operandValue > currentBest[suffixIdx] || operandValueIsNaN {
						currentBest[suffixIdx] = operandValue
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
				}
			}
		}

		// Copy over the result of the whole suffix.
		copyIntsFn(outputFlatIdx, currentArgBest)
		outputFlatIdx += suffixSize
	}
	backend.putBuffer(currentBestBuffer)
	backend.putBuffer(currentArgBestBuffer)
}

func init() {
	argMinMaxDTypeMap.Register(dtypes.BFloat16, execArgMinMaxGenericBFloat16)
}

func execArgMinMaxGenericBFloat16(
	backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize, suffixSize int, isMin bool) {
	operandFlat := operand.flat.([]bfloat16.BFloat16)

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer := backend.getBuffer(operand.shape.DType, suffixSize)
	currentBest := currentBestBuffer.flat.([]bfloat16.BFloat16)
	currentArgBestBuffer := backend.getBuffer(dtypes.Int32, suffixSize)
	currentArgBest := currentArgBestBuffer.flat.([]int32)

	outputFlatIdx := 0
	operandFlatIdx := 0
	for range prefixSize {
		// Initialize the current best with the first element of reduced axis:
		for suffixIdx := range suffixSize {
			currentBest[suffixIdx] = operandFlat[operandFlatIdx]
			currentArgBest[suffixIdx] = 0
			operandFlatIdx++
		}

		// Iterate over the rest of the elements of reduce axis:
		if isMin {
			// ArgMin
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx].Float32()
					if operandValue < currentBest[suffixIdx].Float32() {
						currentBest[suffixIdx] = operandFlat[operandFlatIdx]
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
					operandFlatIdx++
				}
			}
		} else {
			// ArgMax
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx].Float32()
					if operandValue > currentBest[suffixIdx].Float32() {
						currentBest[suffixIdx] = operandFlat[operandFlatIdx]
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
					operandFlatIdx++
				}
			}
		}

		// Copy over the result of the whole suffix.
		copyIntsFn(outputFlatIdx, currentArgBest)
		outputFlatIdx += suffixSize
	}
	backend.putBuffer(currentBestBuffer)
	backend.putBuffer(currentArgBestBuffer)
}

// =================================================================================================================
// ReduceWindow ----------------------------------------------------------------------------------------------------
// =================================================================================================================
func execReduceWindow(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand := inputs[0]
	operandShape := operand.shape
	rank := operandShape.Rank()
	dtype := operandShape.DType
	outputShape := node.shape
	output := backend.getBufferForShape(outputShape)
	opData := node.data.(*reduceWindowNode)

	// resolve the effective parameters, assuming shapeinference.ReduceWindowOp handled nils by defaulting them:
	// - windowDimensions is guaranteed non-nil by the builder.
	// - strides, paddings, baseDilations, windowDilations default if their opData fields are nil.
	effWindowDimensions := opData.windowDimensions
	if effWindowDimensions == nil {
		effWindowDimensions = xslices.SliceWithValue(rank, 1)
	}
	windowShape := shapes.Make(dtype, effWindowDimensions...) // the dtype here is not used.
	effStrides := opData.strides
	if effStrides == nil {
		effStrides = effWindowDimensions
	}
	effPaddings := opData.paddings
	if effPaddings == nil {
		effPaddings = xslices.SliceWithValue(rank, [2]int{0, 0})
	}
	effBaseDilations := opData.baseDilations
	if opData.baseDilations == nil {
		effBaseDilations = xslices.SliceWithValue(rank, 1)
	}
	effWindowDilations := opData.windowDilations
	if effWindowDilations == nil {
		effWindowDilations = xslices.SliceWithValue(rank, 1)
	}

	// Initialize output and updateFn according to the reduction type
	var buildUpdateFnMap *DTypeMap
	switch opData.reductionType {
	case backends.ReduceOpMax:
		err := output.Fill(dtype.LowestValue())
		if err != nil {
			return nil, err
		}
		buildUpdateFnMap = reduceWindowMaxDTypeMap
	case backends.ReduceOpMin:
		err := output.Fill(dtype.HighestValue())
		if err != nil {
			return nil, err
		}
		buildUpdateFnMap = reduceWindowMinDTypeMap
	case backends.ReduceOpProduct:
		output.Ones()
		buildUpdateFnMap = reduceWindowProductDTypeMap
	case backends.ReduceOpSum:
		output.Zeros()
		buildUpdateFnMap = reduceWindowSumDTypeMap
	default:
		return nil, errors.Errorf("ReduceWindow: invalid reduction type: %s", opData.reductionType)
	}
	// updateFn will aggregate the operand value into the corresponding output value.
	updateFn := buildUpdateFnMap.Get(dtype).(func(operand, output *Buffer) reduceWindowUpdateFn)(operand, output)

	// Find the window effective sizes, accounting for the diffusion.
	windowSizes := make([]int, rank)
	for axis := range rank {
		windowSizes[axis] = (effWindowDimensions[axis]-1)*effWindowDilations[axis] + 1
	}
	//fmt.Printf("windowSizes=%v\n", windowSizes)

	// Find the shift from an output position to the corresponding window start in the operand.
	operandShifts := make([]int, rank)
	for axis := range rank {
		operandShifts[axis] = -effPaddings[axis][0]
	}
	//fmt.Printf("operandShifts=%v\n", operandShifts)

	// Find operand strides to convert operand indices to a flat index.
	operandStrides := make([]int, rank)
	stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		operandStrides[axis] = stride
		stride *= operandShape.Dimensions[axis]
	}

	// Main loop: loop over outputs, then over window, then calculate the corresponding operand position
	// that needs to be aggregated, and update the output correspondingly.
	//
	// TODO(optimizations):
	// - If the window will break the cache (outer dimensions of the window), probably that part of the window
	//   can be moved to the outer loop, so instead of having O(N*W) cache misses (random accesses),
	//   we will have O(W) cache misses and the O(N) part will be sequential or in local cache.
	//   More specifically we would split windowShape into "nonCachedWindowShape" and "cachedWindowShape", and
	//   iterate over the nonCachedWindowShape first.
	// - Can we refactor the check of baseDilation to outside of the loop ?
	windowIndices := make([]int, rank)
	operandIndices := make([]int, rank)
	for outputFlatIdx, outputIndices := range outputShape.Iter() {
		//fmt.Printf("Output %v:\n", outputIndices)
	iterWindowIndices:
		for _, windowIndices = range windowShape.IterOn(windowIndices) {
			//fmt.Printf("\t- window %v\n", windowIndices)
			for axis := range rank {
				operandIdx := outputIndices[axis]*effStrides[axis] + operandShifts[axis]
				operandIdx += windowIndices[axis] * effWindowDilations[axis]
				if operandIdx < 0 {
					// This index is out of the operand values (padding), nothing to update.
					continue iterWindowIndices
				}
				if effBaseDilations[axis] > 1 {
					if operandIdx%effBaseDilations[axis] != 0 {
						// This index is not aligned with the baseDilation, nothing to update.
						continue iterWindowIndices
					}
					operandIdx /= effBaseDilations[axis]
				}
				if operandIdx >= operandShape.Dimensions[axis] {
					// This index is out of the operand values (padding), nothing to update.
					continue iterWindowIndices
				}
				operandIndices[axis] = operandIdx
			}
			operandFlatIdx := 0
			for axis, operandIdx := range operandIndices {
				operandFlatIdx += operandIdx * operandStrides[axis]
			}
			updateFn(operandFlatIdx, outputFlatIdx)
		}
	}
	return output, nil
}

type reduceWindowUpdateFn func(operandFlatIdx, outputFlatIdx int)

var (
	reduceWindowMaxDTypeMap     = NewDTypeMap("reduceWindowMaxDTypeMap")
	reduceWindowMinDTypeMap     = NewDTypeMap("reduceWindowMinDTypeMap")
	reduceWindowSumDTypeMap     = NewDTypeMap("reduceWindowSumDTypeMap")
	reduceWindowProductDTypeMap = NewDTypeMap("reduceWindowProductDTypeMap")
)

func init() {
	reduceWindowMaxDTypeMap.Register(dtypes.BFloat16, reduceWindowMaxBuildUpdateFnBFloat16)
	reduceWindowMinDTypeMap.Register(dtypes.BFloat16, reduceWindowMinBuildUpdateFnBFloat16)
	reduceWindowSumDTypeMap.Register(dtypes.BFloat16, reduceWindowSumBuildUpdateFnBFloat16)
	reduceWindowProductDTypeMap.Register(dtypes.BFloat16, reduceWindowProductBuildUpdateFnBFloat16)
}

// Generic functions that build a function that will update the output at outputFlatIdx from the operand at operandFlatIdx.

func reduceWindowMaxBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = max(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMaxBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			max(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowMinBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = min(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMinBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			min(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowSumBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = outputFlat[outputFlatIdx] + operandFlat[operandFlatIdx]
	}
}

func reduceWindowSumBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() + operandFlat[operandFlatIdx].Float32())
	}
}

func reduceWindowProductBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]T)
	outputFlat := output.flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = outputFlat[outputFlatIdx] * operandFlat[operandFlatIdx]
	}
}

func reduceWindowProductBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() * operandFlat[operandFlatIdx].Float32())
	}
}
