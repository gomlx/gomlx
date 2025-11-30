package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// nodeParameter data.
type nodeParameter struct {
	name     string
	inputIdx int
}

// Parameter creates an input parameter for the computation.
// During execution of the computation, this value will need to be fed in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Op, error) {
	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	if sharding != nil {
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"sharding spec %+v not supported for %q builder", sharding, BackendName)
	}
	n := b.newNode(backends.OpTypeParameter, shape)
	n.data = &nodeParameter{
		name:     name,
		inputIdx: len(b.inputs),
	}
	b.inputs = append(b.inputs, n)
	return n, nil
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by the parameter dims.
//
// flat must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	_, err := b.checkOps("Constant")
	if err != nil {
		return nil, err
	}
	dtype, flatLen, err := checkFlat(flat)
	if err != nil {
		return nil, errors.WithMessagef(err, "Constant op")
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		return nil, errors.Errorf("flat ([%d]%s) and shape size (%d) mismatch for constant value",
			flatLen, dtype, shape.Size())
	}
	n := b.newNode(backends.OpTypeConstant, shape)
	n.data = &Buffer{
		shape: shape,
		flat:  flat,
		valid: true,
	}
	return n, nil
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (b *Builder) Iota(shape shapes.Shape, iotaAxis int) (backends.Op, error) {
	_, err := b.checkOps("Iota")
	if err != nil {
		return nil, err
	}
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node := b.newNode(backends.OpTypeIota, shape)
	node.data = iotaAxis
	return node, nil
}

// Identity implements the backends.Identity interface.
func (b *Builder) Identity(operandOp backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps("Reshape", operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	return b.newNode(backends.OpTypeIdentity, operand.shape, operand), nil
}

// Where implements the backends.Builder interface.
func (b *Builder) Where(conditionOp, onTrueOp, onFalseOp backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps("Where", conditionOp, onTrueOp, onFalseOp)
	if err != nil {
		return nil, err
	}
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape, err := shapeinference.WhereOp(condition.shape, onTrue.shape, onFalse.shape)
	if err != nil {
		return nil, err
	}
	return b.newNode(backends.OpTypeWhere, outputShape, condition, onTrue, onFalse), nil
}

// Reshape implements the backends.Builder interface.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func (b *Builder) Reshape(operandOp backends.Op, dims ...int) (backends.Op, error) {
	opType := backends.OpTypeReshape
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReshapeOp(operand.shape, dims)
	if err != nil {
		return nil, err
	}
	return b.newNode(opType, outputShape, operand), nil
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func (b *Builder) Transpose(operandOp backends.Op, permutations ...int) (backends.Op, error) {
	opType := backends.OpTypeTranspose
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.TransposeOp(operand.shape, permutations)
	if err != nil {
		panic(err)
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = permutations
	return node, nil
}

// Broadcast prefixes dimensions to an array by duplicating the data in the array.
// See BroadcastInDim for a broadcast in between the axes.
// The new dimensions dims are inserted on the left, i.e., if
// prefixDims has values `{a0, ..., aN}` and the operand shape
// has dimensions {b0, ..., bM}, then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
// The new dimensions id into copies of the operand, i.e.
//
//	output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
func (b *Builder) Broadcast(operandOp backends.Op, prefixDims ...int) (backends.Op, error) {
	opType := backends.OpTypeBroadcast
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.BroadcastOp(operand.shape, prefixDims)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = prefixDims
	return node, nil
}

// BroadcastInDim broadcasts x to an output with the given shape.
//
//   - outputShape will be the new shape after x is broadcast.
//   - broadcastAxes maps x-axes to the corresponding outputShape axes (len(broadcastAxes) == x.Shape.Rank()),
//     the i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
//     broadcastAxes must be also increasing: this operation cannot be used to transpose axes,
//     it will only broadcast and introduce new axes in-between.
//     -
//
// This also requires that the i-th input axis is either 1 or is the same as the
// output dimension it's broadcasting into.
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcastAxes will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcastAxes
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
func (b *Builder) BroadcastInDim(
	operandOp backends.Op,
	outputShape shapes.Shape,
	broadcastAxes []int,
) (backends.Op, error) {
	opType := backends.OpTypeBroadcastInDim
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	err = shapeinference.BroadcastInDimOp(operand.shape, outputShape, broadcastAxes)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = broadcastAxes
	return node, nil
}

// ReduceMax implements the backends.Builder interface.
func (b *Builder) ReduceMax(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceMax, operandOp, axis...)
}

// ReduceMin implements the backends.Builder interface.
func (b *Builder) ReduceMin(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceMin, operandOp, axis...)
}

// ReduceSum implements the backends.Builder interface.
func (b *Builder) ReduceSum(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceSum, operandOp, axis...)
}

// ReduceProduct implements the backends.Builder interface.
func (b *Builder) ReduceProduct(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceProduct, operandOp, axis...)
}

// ReduceBitwiseAnd implements the backends.Builder interface.
func (b *Builder) ReduceBitwiseAnd(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceBitwiseAnd, operandOp, axis...)
}

// ReduceBitwiseOr implements the backends.Builder interface.
func (b *Builder) ReduceBitwiseOr(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceBitwiseOr, operandOp, axis...)
}

// ReduceBitwiseXor implements the backends.Builder interface.
func (b *Builder) ReduceBitwiseXor(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceBitwiseXor, operandOp, axis...)
}

// ReduceLogicalAnd implements the backends.Builder interface.
func (b *Builder) ReduceLogicalAnd(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceLogicalAnd, operandOp, axis...)
}

// ReduceLogicalOr implements the backends.Builder interface.
func (b *Builder) ReduceLogicalOr(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceLogicalOr, operandOp, axis...)
}

// ReduceLogicalXor implements the backends.Builder interface.
func (b *Builder) ReduceLogicalXor(operandOp backends.Op, axis ...int) (backends.Op, error) {
	return b.reduceImpls(backends.OpTypeReduceLogicalXor, operandOp, axis...)
}

func (b *Builder) reduceImpls(reduceOpType backends.OpType, operandOp backends.Op, axes ...int) (backends.Op, error) {
	inputs, err := b.checkOps("ReduceOp", operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if len(axes) == 0 {
		// Default if no axes are given, is to reduce all axes.
		axes = xslices.Iota(0, operand.shape.Rank())
	}
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.shape.DType
	node := b.newNode(reduceOpType, outputShape, operand)
	node.data = axes
	return node, nil
}

// Gather implements the backends.Builder.
// It's a complex operation, fully described in the backends.Builder.Gather documentation.
func (b *Builder) Gather(
	operandOp, startIndicesOp backends.Op,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Op, error) {
	opType := backends.OpTypeGather
	inputs, err := b.checkOps(opType.String(), operandOp, startIndicesOp)
	if err != nil {
		return nil, err
	}
	operand, startIndices := inputs[0], inputs[1]
	shape, err := shapeinference.Gather(
		operand.shape,
		startIndices.shape,
		indexVectorAxis,
		offsetOutputAxes,
		collapsedSliceAxes,
		startIndexMap,
		sliceSizes,
		indicesAreSorted,
	)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, shape, operand, startIndices)
	node.data = &gatherNode{
		indexVectorAxis,
		offsetOutputAxes,
		collapsedSliceAxes,
		startIndexMap,
		sliceSizes,
		indicesAreSorted,
	}
	return node, nil
}

type gatherNode struct {
	indexVectorAxis                                                  int
	offsetOutputAxes, collapsedSlicesAxes, startIndexMap, sliceSizes []int
	indicesAreSorted                                                 bool
}

// Concatenate joins a sequence of tensors along the given axis (it must exist already).
// All input tensors must have the same shape, except potentially in the concatenation dimension.
// They must also have the same data type (DType).
// It returns an error if inputs are invalid (e.g., no inputs, mismatched graphs, shapes, dtypes, or invalid dimension).
func (b *Builder) Concatenate(axis int, operandOps ...backends.Op) (backends.Op, error) {
	if len(operandOps) == 0 {
		return nil, errors.Errorf("Concatenate requires at least one input tensor")
	}
	operands, err := b.checkOps("Concatenate", operandOps...)
	if err != nil {
		return nil, err
	}

	// Extract shapes for shape inference.
	inputShapes := make([]shapes.Shape, len(operands))
	for i, opNode := range operands {
		inputShapes[i] = opNode.shape
	}
	outputShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, err
	}
	node := b.newNode(backends.OpTypeConcatenate, outputShape, operands...)
	node.data = axis
	return node, nil
}

// ConvertDType converts operandOp to the given dtype. It implements the backends.Builder interface.
func (b *Builder) ConvertDType(operandOp backends.Op, dtype dtypes.DType) (backends.Op, error) {
	opType := backends.OpTypeConvertDType
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if operand.shape.DType == dtype {
		// No-op
		return operand, nil
	}
	outputShape := operand.shape.Clone()
	outputShape.DType = dtype
	return b.newNode(opType, outputShape, operand), nil
}

// ScatterMax implements the backends.Builder interface.
func (b *Builder) ScatterMax(
	operandOp, scatterIndicesOp, updatesOp backends.Op,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Op, error) {
	return b.scatterImpls(
		backends.OpTypeScatterMax,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

// ScatterMin implements the backends.Builder interface.
func (b *Builder) ScatterMin(
	operandOp, scatterIndicesOp, updatesOp backends.Op,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Op, error) {
	return b.scatterImpls(
		backends.OpTypeScatterMin,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

// ScatterSum implements the backends.Builder interface.
func (b *Builder) ScatterSum(
	operandOp, scatterIndicesOp, updatesOp backends.Op,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Op, error) {
	return b.scatterImpls(
		backends.OpTypeScatterSum,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

func (b *Builder) scatterImpls(
	scatterOpType backends.OpType,
	operandOp, scatterIndicesOp, updatesOp backends.Op,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (
	backends.Op, error) {
	inputs, err := b.checkOps(scatterOpType.String(), operandOp, scatterIndicesOp, updatesOp)
	if err != nil {
		return nil, err
	}
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	// Check that parameters are valid.
	outputShape, err := shapeinference.ScatterOp(
		operand.shape,
		indices.shape,
		updates.shape,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
	)
	if err != nil {
		return nil, err
	}

	// The output shape of the scatter is the operand shape.
	node := b.newNode(scatterOpType, outputShape, operand, indices, updates)
	node.data = &scatterNode{
		updateWindowAxes:         updateWindowAxes,
		insertedWindowAxes:       insertedWindowAxes,
		scatterAxesToOperandAxes: scatterAxesToOperandAxes,
		indexVectorAxis:          indexVectorAxis,
		indicesAreSorted:         indicesAreSorted,
		uniqueIndices:            uniqueIndices,
	}
	return node, nil
}

// scatterNode is attached to the Node.data field for ScatterMax, ScatterMin, ScatterSum.
type scatterNode struct {
	indexVectorAxis                                                int
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int
	indicesAreSorted, uniqueIndices                                bool
}

// Slice extracts a subarray from the input array.
// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing, it is assumed to be 1 for every dimension.
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func (b *Builder) Slice(operandOp backends.Op, starts, limits, strides []int) (backends.Op, error) {
	opType := backends.OpTypeSlice
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.SliceOp(operand.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = &sliceNode{
		starts,
		limits,
		strides,
	}
	return node, nil
}

// sliceNode is attached to the Node.data field for Slice.
type sliceNode struct {
	starts, limits, strides []int
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RngState or RngStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func (b *Builder) RNGBitGenerator(stateOp backends.Op, shape shapes.Shape) (newState, values backends.Op, err error) {
	opType := backends.OpTypeRNGBitGenerator
	inputs, err := b.checkOps(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]
	if !state.shape.Equal(backends.RNGStateShape) {
		err := errors.Errorf(
			"expected random state to be shaped %s, got state.shape=%s instead for RNGBitGenerator",
			backends.RNGStateShape,
			state.shape,
		)
		return nil, nil, err
	}
	outputShapes := []shapes.Shape{
		state.shape.Clone(),
		shape.Clone(),
	}
	node := b.newMultiOutputsNode(opType, outputShapes, state)
	newState = node.multiOutputsNodes[0]
	values = node.multiOutputsNodes[1]
	return
}

type argMinMaxNode struct {
	axis  int
	isMin bool
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
// It's a form of reduction on the given axis, and that axis goes away. So the rank of the result is one less than
// the rank of x.
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func (b *Builder) ArgMinMax(
	operandOp backends.Op,
	axis int,
	outputDType dtypes.DType,
	isMin bool,
) (backends.Op, error) {
	opType := backends.OpTypeArgMinMax
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ArgMinMaxOp(operand.shape, axis, outputDType)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = &argMinMaxNode{
		axis,
		isMin,
	}
	return node, nil
}

type reduceWindowNode struct {
	reductionType                                             backends.ReduceOpType
	windowDimensions, strides, baseDilations, windowDilations []int
	paddings                                                  [][2]int
}

// ReduceWindow runs a reduction function of reduceType (backends.ReduceOpMax, backends.ReduceOpSum or backends.ReduceOpProduct).
//
// The parameter windowDimensions must be set and have a value for each axis.
// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
// If paddings is nil, they are assumed to be 0.
func (b *Builder) ReduceWindow(
	operandOp backends.Op,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Op, error) {
	opType := backends.OpTypeReduceWindow
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReduceWindowOp(
		operand.shape,
		windowDimensions,
		strides,
		baseDilations,
		windowDilations,
		paddings,
	)
	if err != nil {
		return nil, err
	}
	node := b.newNode(opType, outputShape, operand)
	node.data = &reduceWindowNode{
		reductionType:    reductionType,
		windowDimensions: windowDimensions,
		strides:          strides,
		baseDilations:    baseDilations,
		windowDilations:  windowDilations,
		paddings:         paddings,
	}
	return node, nil
}

//======================================================================================================================
// Unary Operations ----------------------------------------------------------------------------------------------------
//======================================================================================================================

// Neg implements the backends.Builder interface.
func (b *Builder) Neg(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeNeg, operand)
}

// Sign implements the backends.Builder interface.
func (b *Builder) Sign(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSign, operand)
}

// Abs implements the backends.Builder interface.
func (b *Builder) Abs(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeAbs, operand)
}

// LogicalNot implements the backends.Builder interface.
func (b *Builder) LogicalNot(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLogicalNot, operand)
}

// BitwiseNot implements the backends.Builder interface.
func (b *Builder) BitwiseNot(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeBitwiseNot, operand)
}

// BitCount implements the backends.Builder interface.
func (b *Builder) BitCount(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeBitCount, operand)
}

// Clz implements the backends.Builder interface.
func (b *Builder) Clz(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeClz, operand)
}

// Exp implements the backends.Builder interface.
func (b *Builder) Exp(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeExp, operand)
}

// Expm1 implements the backends.Builder interface. It returns e(x)-1.
func (b *Builder) Expm1(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeExpm1, operand)
}

// Log implements the backends.Builder interface.
func (b *Builder) Log(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLog, operand)
}

// Log1p implements the backends.Builder interface.
func (b *Builder) Log1p(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLog1p, operand)
}

// Logistic implements the backends.Builder interface. Aka as sigmoid. It returns 1/(1+exp(-x)).
func (b *Builder) Logistic(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLogistic, operand)
}

// Ceil implements the backends.Builder interface.
func (b *Builder) Ceil(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeCeil, operand)
}

// Floor implements the backends.Builder interface.
func (b *Builder) Floor(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeFloor, operand)
}

// Round implements the backends.Builder interface.
func (b *Builder) Round(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeRound, operand)
}

// Rsqrt implements the backends.Builder interface.
func (b *Builder) Rsqrt(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeRsqrt, operand)
}

// Sqrt implements the backends.Builder interface.
func (b *Builder) Sqrt(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSqrt, operand)
}

// Cos implements the backends.Builder interface.
func (b *Builder) Cos(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeCos, operand)
}

// Sin implements the backends.Builder interface.
func (b *Builder) Sin(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSin, operand)
}

// Tanh implements the backends.Builder interface.
func (b *Builder) Tanh(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeTanh, operand)
}

// Erf implements the backends.Builder interface.
func (b *Builder) Erf(operand backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeErf, operand)
}

// IsFinite implements the backends.Builder interface.
func (b *Builder) IsFinite(operandOp backends.Op) (backends.Op, error) {
	opType := backends.OpTypeIsFinite
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	dtype := operand.shape.DType
	if !dtype.IsFloat() && !dtype.IsComplex() {
		return nil, errors.Errorf(
			"the operation IsFinite is only defined for float types (%s), cannot use it",
			operand.shape.DType,
		)
	}

	// Output will have the same shape but for the dtype that is bool.
	shape := operand.shape.Clone()
	shape.DType = dtypes.Bool
	return b.newNode(opType, shape, operand), nil
}

// Binary Operations:

// Add implements the backends.Builder interface.
func (b *Builder) Add(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeAdd, lhsOp, rhsOp)
}

// Mul implements the backends.Builder interface.
func (b *Builder) Mul(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMul, lhsOp, rhsOp)
}

// Sub implements the backends.Builder interface.
func (b *Builder) Sub(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeSub, lhsOp, rhsOp)
}

// Div implements the backends.Builder interface.
func (b *Builder) Div(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeDiv, lhsOp, rhsOp)
}

// Rem implements the backends.Builder interface.
func (b *Builder) Rem(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeRem, lhsOp, rhsOp)
}

// Pow implements the backends.Builder interface.
func (b *Builder) Pow(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypePow, lhsOp, rhsOp)
}

// BitwiseAnd implements the backends.Builder interface.
func (b *Builder) BitwiseAnd(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeBitwiseAnd, lhsOp, rhsOp)
}

// BitwiseOr implements the backends.Builder interface.
func (b *Builder) BitwiseOr(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeBitwiseOr, lhsOp, rhsOp)
}

// BitwiseXor implements the backends.Builder interface.
func (b *Builder) BitwiseXor(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeBitwiseXor, lhsOp, rhsOp)
}

// LogicalAnd implements the backends.Builder interface.
func (b *Builder) LogicalAnd(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeLogicalAnd, lhsOp, rhsOp)
}

// LogicalOr implements the backends.Builder interface.
func (b *Builder) LogicalOr(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeLogicalOr, lhsOp, rhsOp)
}

// LogicalXor implements the backends.Builder interface.
func (b *Builder) LogicalXor(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeLogicalXor, lhsOp, rhsOp)
}

// Max implements the backends.Builder interface.
func (b *Builder) Max(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMax, lhsOp, rhsOp)
}

// Min implements the backends.Builder interface.
func (b *Builder) Min(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMin, lhsOp, rhsOp)
}

// Equal implements the backends.Builder interface.
func (b *Builder) Equal(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeEqual, lhsOp, rhsOp)
}

// NotEqual implements the backends.Builder interface.
func (b *Builder) NotEqual(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeNotEqual, lhsOp, rhsOp)
}

// GreaterOrEqual implements the backends.Builder interface.
func (b *Builder) GreaterOrEqual(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeGreaterOrEqual, lhsOp, rhsOp)
}

// GreaterThan implements the backends.Builder interface.
func (b *Builder) GreaterThan(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeGreaterThan, lhsOp, rhsOp)
}

// LessOrEqual implements the backends.Builder interface.
func (b *Builder) LessOrEqual(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeLessOrEqual, lhsOp, rhsOp)
}

// LessThan implements the backends.Builder interface.
func (b *Builder) LessThan(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeLessThan, lhsOp, rhsOp)
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (b *Builder) Clamp(min, x, max backends.Op) (backends.Op, error) {
	clamped, err := b.Max(min, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	clamped, err = b.Min(clamped, max)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}

// IsNaN implements backends.Builder interface.
func (b *Builder) IsNaN(x backends.Op) (backends.Op, error) {
	result, err := b.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}
