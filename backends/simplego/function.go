// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// Function implements backends.Function for SimpleGo.
type Function struct {
	notimplemented.Function

	builder *Builder
	name    string

	// parent is the parent function if this is a closure.
	// For top-level functions (including main), this is nil.
	parent *Function

	// returned indicates Return() was called.
	returned bool

	// outputs stores the return values set by Return().
	outputs []*Node

	// parameters stores the parameter nodes for this function.
	parameters []*Node

	// compiled holds pre-compiled execution info.
	// This is set during Return() to allow efficient execution.
	compiled *FunctionExecutable
}

var _ backends.Function = (*Function)(nil)

// CheckValid returns an error if the builder or the function are not ok.
func (f *Function) CheckValid() error {
	if f == nil || f.builder == nil {
		return errors.Errorf("function is nil or undefined for %q", BackendName)
	}
	if f.builder.compiled {
		return errors.Errorf("cannot add new op to Function %q, builder has already been compiled", f.name)
	}
	return nil
}

// Name returns the name of this function.
// For closures, this returns "".
func (f *Function) Name() string {
	return f.name
}

// Parent returns the parent function if this is a closure.
// Returns nil for top-level functions (including main).
func (f *Function) Parent() backends.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// Closure creates a new closure function within this function.
// Closures can access values from their parent function's scope.
func (f *Function) Closure() (backends.Function, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	closure := &Function{
		builder: f.builder,
		name:    "", // Closures have empty names
		parent:  f,
	}
	return closure, nil
}

// verifyAndCastValues sanity checks that the values (backends.Op) are valid and created with this builder.
// It also verifies that all input nodes belong to the same function - using nodes from a parent
// function (closure capturing) is not yet supported.
// It returns the underlying *Node of the values.
func (f *Function) verifyAndCastValues(name string, values ...backends.Value) ([]*Node, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.builder.checkOps(name, values...)
	if err != nil {
		return nil, err
	}

	// Check that all input nodes belong to this function.
	// Capturing nodes from a parent function (closure capture) is not yet supported
	// because it introduces complexity in tracking node usage for buffer management.
	for idx, node := range nodes {
		if node.function != nil && node.function != f {
			// Check if the node is from a parent function (closure capture)
			isFromParent := false
			for parent := f.parent; parent != nil; parent = parent.parent {
				if node.function == parent {
					isFromParent = true
					break
				}
			}
			if isFromParent {
				return nil, errors.Errorf(
					"%s: input #%d uses a node from a parent function scope (closure capturing parent values). "+
						"This is not yet supported in the SimpleGo backend. "+
						"Please pass the value as a closure parameter instead. "+
						"If you need this feature, please open an issue at github.com/gomlx/gomlx",
					name, idx)
			}
			// Node from a completely different function (not parent)
			return nil, errors.Errorf(
				"%s: input #%d uses a node from a different function scope",
				name, idx)
		}
	}

	return nodes, nil
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Value, error) {
	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.builder.backend.Name())
	}
	if sharding != nil {
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"sharding spec %+v not supported for %q builder", sharding, BackendName)
	}
	data := &nodeParameter{
		name:     name,
		inputIdx: len(f.builder.inputs),
	}
	n, _ := f.builder.getOrCreateNode(f, backends.OpTypeParameter, shape, nil, data)
	f.builder.inputs = append(f.builder.inputs, n)
	f.parameters = append(f.parameters, n) // Track in function for closures
	return n, nil
}

// Constant creates a constant in the function with the given flat values and the shape defined by the dimensions.
func (f *Function) Constant(flat any, dims ...int) (backends.Value, error) {
	_, err := f.builder.checkOps("Constant")
	if err != nil {
		return nil, err
	}
	dtype, flatLen, err := checkFlat(flat)
	if err != nil {
		return nil, errors.WithMessagef(err, "Constant op")
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.builder.backend.Name())
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		return nil, errors.Errorf("flat ([%d]%s) and shape size (%d) mismatch for constant value",
			flatLen, dtype, shape.Size())
	}
	data := &Buffer{
		shape: shape,
		flat:  flat,
		valid: true,
	}
	n, _ := f.builder.getOrCreateNode(f, backends.OpTypeConstant, shape, nil, data)
	return n, nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	if err := f.CheckValid(); err != nil {
		return err
	}
	if f.returned {
		return errors.Errorf("Return() already called for function %q", f.name)
	}
	if len(outputs) == 0 {
		return errors.Errorf("Return() requires at least one output")
	}
	if len(shardings) != 0 {
		return errors.Errorf("sharding or distributed execution are not supported by SimpleGo backend")
	}

	outputNodes, err := f.verifyAndCastValues("Return", outputs...)
	if err != nil {
		return err
	}

	for _, node := range outputNodes {
		if len(node.multiOutputsShapes) != 0 {
			return errors.Errorf(
				"%s node %q is internal (with multiple-outputs) and cannot be used for output",
				f.builder.Name(),
				node.opType,
			)
		}
	}

	f.outputs = outputNodes
	f.returned = true

	// If this is a closure, pre-compile it for efficient execution.
	// Main functions are compiled later in Builder.Compile() after
	// duplicate output handling.
	if f.parent != nil {
		compiled, err := newFunctionExecutable(f)
		if err != nil {
			return errors.WithMessagef(err, "failed to compile closure")
		}
		f.compiled = compiled
	}

	return nil
}

// Compiled returns the pre-compiled function executable, or nil if not yet compiled.
func (f *Function) Compiled() *FunctionExecutable {
	return f.compiled
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (backends.Value, error) {
	_, err := f.builder.checkOps("Iota")
	if err != nil {
		return nil, err
	}
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node, _ := f.builder.getOrCreateNode(f, backends.OpTypeIota, shape, nil, iotaAxis)
	return node, nil
}

// Identity implements the backends.Identity interface.
// This operation is not de-duplicated: if you issue it twice, it will not reuse the previous instance.
func (f *Function) Identity(operandOp backends.Value) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("Reshape", operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	node := f.builder.newNode(f, backends.OpTypeIdentity, operand.shape, operand)
	return node, nil
}

// Where implements the backends.Builder interface.
func (f *Function) Where(conditionOp, onTrueOp, onFalseOp backends.Value) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("Where", conditionOp, onTrueOp, onFalseOp)
	if err != nil {
		return nil, err
	}
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape, err := shapeinference.WhereOp(condition.shape, onTrue.shape, onFalse.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, backends.OpTypeWhere, outputShape, []*Node{condition, onTrue, onFalse}, nil)
	return node, nil
}

// Reshape implements the backends.Builder interface.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func (f *Function) Reshape(operandOp backends.Value, dims ...int) (backends.Value, error) {
	opType := backends.OpTypeReshape
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReshapeOp(operand.shape, dims)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func (f *Function) Transpose(operandOp backends.Value, permutations ...int) (backends.Value, error) {
	opType := backends.OpTypeTranspose
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.TransposeOp(operand.shape, permutations)
	if err != nil {
		panic(err)
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, permutations)
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
func (f *Function) Broadcast(operandOp backends.Value, prefixDims ...int) (backends.Value, error) {
	opType := backends.OpTypeBroadcast
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.BroadcastOp(operand.shape, prefixDims)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, prefixDims)
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
func (f *Function) BroadcastInDim(
	operandOp backends.Value,
	outputShape shapes.Shape,
	broadcastAxes []int,
) (backends.Value, error) {
	opType := backends.OpTypeBroadcastInDim
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	err = shapeinference.BroadcastInDimOp(operand.shape, outputShape, broadcastAxes)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, broadcastAxes)
	return node, nil
}

// ReduceMax implements the backends.Builder interface.
func (f *Function) ReduceMax(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceMax, operandOp, axis...)
}

// ReduceMin implements the backends.Builder interface.
func (f *Function) ReduceMin(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceMin, operandOp, axis...)
}

// ReduceSum implements the backends.Builder interface.
func (f *Function) ReduceSum(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceSum, operandOp, axis...)
}

// ReduceProduct implements the backends.Builder interface.
func (f *Function) ReduceProduct(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceProduct, operandOp, axis...)
}

// ReduceBitwiseAnd implements the backends.Builder interface.
func (f *Function) ReduceBitwiseAnd(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceBitwiseAnd, operandOp, axis...)
}

// ReduceBitwiseOr implements the backends.Builder interface.
func (f *Function) ReduceBitwiseOr(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceBitwiseOr, operandOp, axis...)
}

// ReduceBitwiseXor implements the backends.Builder interface.
func (f *Function) ReduceBitwiseXor(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceBitwiseXor, operandOp, axis...)
}

// ReduceLogicalAnd implements the backends.Builder interface.
func (f *Function) ReduceLogicalAnd(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceLogicalAnd, operandOp, axis...)
}

// ReduceLogicalOr implements the backends.Builder interface.
func (f *Function) ReduceLogicalOr(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceLogicalOr, operandOp, axis...)
}

// ReduceLogicalXor implements the backends.Builder interface.
func (f *Function) ReduceLogicalXor(operandOp backends.Value, axis ...int) (backends.Value, error) {
	return f.reduceImpls(backends.OpTypeReduceLogicalXor, operandOp, axis...)
}

func (f *Function) reduceImpls(reduceOpType backends.OpType, operandOp backends.Value, axes ...int) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("ReduceOp", operandOp)
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
	node, _ := f.builder.getOrCreateNode(f, reduceOpType, outputShape, []*Node{operand}, axes)
	return node, nil
}

// Gather implements the backends.Builder.
// It's a complex operation, fully described in the backends.Builder.Gather documentation.
func (f *Function) Gather(
	operandOp, startIndicesOp backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Value, error) {
	opType := backends.OpTypeGather
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp, startIndicesOp)
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
	data := &gatherNode{
		indexVectorAxis,
		offsetOutputAxes,
		collapsedSliceAxes,
		startIndexMap,
		sliceSizes,
		indicesAreSorted,
	}
	node, _ := f.builder.getOrCreateNode(f, opType, shape, []*Node{operand, startIndices}, data)
	return node, nil
}

// Concatenate joins a sequence of tensors along the given axis (it must exist already).
// All input tensors must have the same shape, except potentially in the concatenation dimension.
// They must also have the same data type (DType).
// It returns an error if inputs are invalid (e.g., no inputs, mismatched graphs, shapes, dtypes, or invalid dimension).
func (f *Function) Concatenate(axis int, operandOps ...backends.Value) (backends.Value, error) {
	if len(operandOps) == 0 {
		return nil, errors.Errorf("Concatenate requires at least one input tensor")
	}
	operands, err := f.verifyAndCastValues("Concatenate", operandOps...)
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
	node, _ := f.builder.getOrCreateNode(f, backends.OpTypeConcatenate, outputShape, operands, axis)
	return node, nil
}

// ConvertDType converts operandOp to the given dtype. It implements the backends.Builder interface.
func (f *Function) ConvertDType(operandOp backends.Value, dtype dtypes.DType) (backends.Value, error) {
	opType := backends.OpTypeConvertDType
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
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
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// ScatterMax implements the backends.Builder interface.
func (f *Function) ScatterMax(
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpls(
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
func (f *Function) ScatterMin(
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpls(
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
func (f *Function) ScatterSum(
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpls(
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

func (f *Function) scatterImpls(
	scatterOpType backends.OpType,
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (
	backends.Value, error) {
	inputs, err := f.verifyAndCastValues(scatterOpType.String(), operandOp, scatterIndicesOp, updatesOp)
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
	data := &scatterNode{
		updateWindowAxes:         updateWindowAxes,
		insertedWindowAxes:       insertedWindowAxes,
		scatterAxesToOperandAxes: scatterAxesToOperandAxes,
		indexVectorAxis:          indexVectorAxis,
		indicesAreSorted:         indicesAreSorted,
		uniqueIndices:            uniqueIndices,
	}
	node, _ := f.builder.getOrCreateNode(f, scatterOpType, outputShape, []*Node{operand, indices, updates}, data)
	return node, nil
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
func (f *Function) Slice(operandOp backends.Value, starts, limits, strides []int) (backends.Value, error) {
	opType := backends.OpTypeSlice
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.SliceOp(operand.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	data := &sliceNode{
		starts,
		limits,
		strides,
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RNGState or RNGStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func (f *Function) RNGBitGenerator(stateOp backends.Value, shape shapes.Shape) (newState, values backends.Value, err error) {
	opType := backends.OpTypeRNGBitGenerator
	inputs, err := f.verifyAndCastValues(opType.String(), stateOp)
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
	node := f.builder.newMultiOutputsNode(f, opType, outputShapes, state)
	newState = node.multiOutputsNodes[0]
	values = node.multiOutputsNodes[1]
	return
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
// It's a form of reduction on the given axis, and that axis goes away. So the rank of the result is one less than
// the rank of x.
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func (f *Function) ArgMinMax(
	operandOp backends.Value,
	axis int,
	outputDType dtypes.DType,
	isMin bool,
) (backends.Value, error) {
	opType := backends.OpTypeArgMinMax
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ArgMinMaxOp(operand.shape, axis, outputDType)
	if err != nil {
		return nil, err
	}
	data := &argMinMaxNode{
		axis,
		isMin,
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// ReduceWindow runs a reduction function of reduceType (backends.ReduceOpMax, backends.ReduceOpSum or backends.ReduceOpProduct).
//
// The parameter windowDimensions must be set and have a value for each axis.
// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
// If paddings is nil, they are assumed to be 0.
func (f *Function) ReduceWindow(
	operandOp backends.Value,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Value, error) {
	opType := backends.OpTypeReduceWindow
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
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
	data := &reduceWindowNode{
		reductionType:    reductionType,
		windowDimensions: windowDimensions,
		strides:          strides,
		baseDilations:    baseDilations,
		windowDilations:  windowDilations,
		paddings:         paddings,
	}
	node, _ := f.builder.getOrCreateNode(f, opType, outputShape, []*Node{operand}, data)
	return node, nil
}

//======================================================================================================================
// Unary Operations ----------------------------------------------------------------------------------------------------
//======================================================================================================================

// Neg implements the backends.Builder interface.
func (f *Function) Neg(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeNeg, operand)
}

// Sign implements the backends.Builder interface.
func (f *Function) Sign(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSign, operand)
}

// Abs implements the backends.Builder interface.
func (f *Function) Abs(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeAbs, operand)
}

// LogicalNot implements the backends.Builder interface.
func (f *Function) LogicalNot(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLogicalNot, operand)
}

// BitwiseNot implements the backends.Builder interface.
func (f *Function) BitwiseNot(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeBitwiseNot, operand)
}

// BitCount implements the backends.Builder interface.
func (f *Function) BitCount(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeBitCount, operand)
}

// Clz implements the backends.Builder interface.
func (f *Function) Clz(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeClz, operand)
}

// Exp implements the backends.Builder interface.
func (f *Function) Exp(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeExp, operand)
}

// Expm1 implements the backends.Builder interface. It returns e(x)-1.
func (f *Function) Expm1(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeExpm1, operand)
}

// Log implements the backends.Builder interface.
func (f *Function) Log(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLog, operand)
}

// Log1p implements the backends.Builder interface.
func (f *Function) Log1p(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLog1p, operand)
}

// Logistic implements the backends.Builder interface. Aka as sigmoid. It returns 1/(1+exp(-x)).
func (f *Function) Logistic(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLogistic, operand)
}

// Ceil implements the backends.Builder interface.
func (f *Function) Ceil(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeCeil, operand)
}

// Floor implements the backends.Builder interface.
func (f *Function) Floor(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeFloor, operand)
}

// Round implements the backends.Builder interface.
func (f *Function) Round(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeRound, operand)
}

// Rsqrt implements the backends.Builder interface.
func (f *Function) Rsqrt(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeRsqrt, operand)
}

// Sqrt implements the backends.Builder interface.
func (f *Function) Sqrt(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSqrt, operand)
}

// Cos implements the backends.Builder interface.
func (f *Function) Cos(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeCos, operand)
}

// Sin implements the backends.Builder interface.
func (f *Function) Sin(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSin, operand)
}

// Tanh implements the backends.Builder interface.
func (f *Function) Tanh(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeTanh, operand)
}

// Erf implements the backends.Builder interface.
func (f *Function) Erf(operand backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeErf, operand)
}

// IsFinite implements the backends.Builder interface.
func (f *Function) IsFinite(operandOp backends.Value) (backends.Value, error) {
	opType := backends.OpTypeIsFinite
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
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
	node, _ := f.builder.getOrCreateNode(f, opType, shape, []*Node{operand}, nil)
	return node, nil
}

// addUnaryOp adds a generic binary op.
func (f *Function) addUnaryOp(opType backends.OpType, operandOp backends.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	shape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {

		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, shape, []*Node{operand}, nil)
	return node, nil
}

// Binary Operations:

// Add implements the backends.Builder interface.
func (f *Function) Add(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeAdd, lhsOp, rhsOp)
}

// Mul implements the backends.Builder interface.
func (f *Function) Mul(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMul, lhsOp, rhsOp)
}

// Sub implements the backends.Builder interface.
func (f *Function) Sub(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeSub, lhsOp, rhsOp)
}

// Div implements the backends.Builder interface.
func (f *Function) Div(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeDiv, lhsOp, rhsOp)
}

// Rem implements the backends.Builder interface.
func (f *Function) Rem(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeRem, lhsOp, rhsOp)
}

// Pow implements the backends.Builder interface.
func (f *Function) Pow(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypePow, lhsOp, rhsOp)
}

// BitwiseAnd implements the backends.Builder interface.
func (f *Function) BitwiseAnd(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeBitwiseAnd, lhsOp, rhsOp)
}

// BitwiseOr implements the backends.Builder interface.
func (f *Function) BitwiseOr(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeBitwiseOr, lhsOp, rhsOp)
}

// BitwiseXor implements the backends.Builder interface.
func (f *Function) BitwiseXor(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeBitwiseXor, lhsOp, rhsOp)
}

// LogicalAnd implements the backends.Builder interface.
func (f *Function) LogicalAnd(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeLogicalAnd, lhsOp, rhsOp)
}

// LogicalOr implements the backends.Builder interface.
func (f *Function) LogicalOr(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeLogicalOr, lhsOp, rhsOp)
}

// LogicalXor implements the backends.Builder interface.
func (f *Function) LogicalXor(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeLogicalXor, lhsOp, rhsOp)
}

// Max implements the backends.Builder interface.
func (f *Function) Max(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMax, lhsOp, rhsOp)
}

// Min implements the backends.Builder interface.
func (f *Function) Min(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMin, lhsOp, rhsOp)
}

// Equal implements the backends.Builder interface.
func (f *Function) Equal(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeEqual, lhsOp, rhsOp)
}

// NotEqual implements the backends.Builder interface.
func (f *Function) NotEqual(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeNotEqual, lhsOp, rhsOp)
}

// GreaterOrEqual implements the backends.Builder interface.
func (f *Function) GreaterOrEqual(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeGreaterOrEqual, lhsOp, rhsOp)
}

// GreaterThan implements the backends.Builder interface.
func (f *Function) GreaterThan(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeGreaterThan, lhsOp, rhsOp)
}

// LessOrEqual implements the backends.Builder interface.
func (f *Function) LessOrEqual(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeLessOrEqual, lhsOp, rhsOp)
}

// LessThan implements the backends.Builder interface.
func (f *Function) LessThan(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeLessThan, lhsOp, rhsOp)
}

// addBinaryOp adds a generic binary op.
func (f *Function) addBinaryOp(opType backends.OpType, lhsOp, rhsOp backends.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.BinaryOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, shape, []*Node{lhs, rhs}, nil)
	return node, nil
}

// addComparisonOp adds a generic comparison binary op.
func (f *Function) addComparisonOp(opType backends.OpType, lhsOp, rhsOp backends.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.ComparisonOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.builder.getOrCreateNode(f, opType, shape, []*Node{lhs, rhs}, nil)
	return node, nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (f *Function) Clamp(min, x, max backends.Value) (backends.Value, error) {
	clamped, err := f.Max(min, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	clamped, err = f.Min(clamped, max)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}

// IsNaN implements backends.Builder interface.
func (f *Function) IsNaN(x backends.Value) (backends.Value, error) {
	result, err := f.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}

// AllReduce implements the backends.CollectiveOps interface.
func (f *Function) AllReduce(operands []backends.Value, reductionType backends.ReduceOpType, replicaGroups [][]int) ([]backends.Value, error) {
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"AllReduce not supported for %q builder", BackendName)
}

