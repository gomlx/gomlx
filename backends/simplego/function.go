// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/pkg/errors"
)

// Function implements compute.Function for SimpleGo.
type Function struct {
	notimplemented.Function

	builder *Builder
	name    string

	// parent is the parent function if this is a closure.
	// For top-level functions (including main), this is nil.
	parent *Function

	// returned indicates Return() was called.
	returned bool

	// nodes are all nodes created within this function, in DAG order.
	// Each node's idx field is its index in this slice.
	nodes []*Node

	// outputs stores the return values set by Return().
	outputs []*Node

	// parameters stores the parameter nodes for this function.
	parameters []*Node

	// capturedParentNodes stores nodes from parent scopes that are captured by this closure.
	// The order matches capturedLocalNodes - capturedParentNodes[i] is the parent node for capturedLocalNodes[i].
	capturedParentNodes []*Node

	// capturedLocalNodes stores the proxy nodes in this closure for captured values.
	// These are OpTypeCapturedValue nodes that receive their values at execution time.
	capturedLocalNodes []*Node

	// nodeDedup provides automatic de-duplication for nodes within this function.
	nodeDedup map[nodeDedupKey][]*Node

	// compiled holds pre-compiled execution info.
	// This is set during Return() to allow efficient execution.
	compiled *FunctionExecutable
}

// capturedNodeData is the data stored in a captured value node.
// It just stores the capture index since the parent node is available
// via f.capturedParentNodes[captureIdx].
type capturedNodeData int

var _ compute.Function = (*Function)(nil)

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
func (f *Function) Parent() compute.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// IsAncestorOf checks whether f is an ancestor of leafFunc.
// It returns true if f == leafFunc.
//
// Typically, leafFunc will be a closure.
func (f *Function) IsAncestorOf(leafFunc *Function) bool {
	for ; leafFunc != nil; leafFunc = leafFunc.parent {
		if leafFunc == f {
			return true
		}
	}
	return false
}

// Closure creates a new closure function within this function.
// Closures can access values from their parent function's scope.
func (f *Function) Closure() (compute.Function, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	closure := &Function{
		Function: notimplemented.Function{
			ErrFn: notImplementedError,
		},
		builder:   f.builder,
		name:      "", // Closures have empty names
		parent:    f,
		nodeDedup: make(map[nodeDedupKey][]*Node),
	}
	return closure, nil
}

// newNode adds a new node of the given opType and shape to the function's graph.
// It's used by the other ops when creating new nodes.
// Nodes are added to the function's nodes slice.
//
// Use getOrCreateNode instead for most operations.
func (f *Function) newNode(opType compute.OpType, shape shapes.Shape, inputs ...*Node) *Node {
	n := &Node{
		builder:  f.builder,
		opType:   opType,
		idx:      len(f.nodes),
		shape:    shape,
		inputs:   slices.Clone(inputs),
		function: f,
	}
	f.nodes = append(f.nodes, n)
	return n
}

// newMultiOutputsNode creates the multi-outputs node, and its "select nodes", one per output.
// The node.multiOutputsNodes will be set with the individual outputs and can be used by the Builder to return
// to the user.
// Nodes are added to the function's nodes slice.
//
// Note: no de-duplication of multi-output nodes.
func (f *Function) newMultiOutputsNode(
	opType compute.OpType,
	outputShapes []shapes.Shape,
	inputs ...*Node,
) (node *Node) {
	node = f.newNode(opType, shapes.Invalid(), inputs...)
	node.multiOutputsShapes = outputShapes
	node.multiOutputsNodes = make([]*Node, len(outputShapes))
	for i, shape := range outputShapes {
		node.multiOutputsNodes[i] = &Node{
			builder:            f.builder,
			opType:             opType,
			idx:                len(f.nodes),
			shape:              shape,
			inputs:             []*Node{node},
			isNodeSelectOutput: true,
			selectOutputIdx:    i,
			function:           f,
		}
		f.nodes = append(f.nodes, node.multiOutputsNodes[i])
	}
	return node
}

// verifyAndCastValues sanity checks that the values (compute.Op) are valid and created with this builder.
// If a node belongs to a parent function, it creates a capture node to access the value.
// It returns the underlying *Node of the values (with capture nodes substituted for parent values).
func (f *Function) verifyAndCastValues(name string, values ...compute.Value) ([]*Node, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.builder.checkValues(name, values...)
	if err != nil {
		return nil, err
	}

	// Check each node and handle parent scope references
	for idx, node := range nodes {
		if node.function == nil {
			return nil, errors.Errorf(
				"%s: input #%d has nil function (internal error)",
				name, idx)
		}
		if node.function == f {
			continue // Same function, OK.
		}

		// Check if the node is from an ancestor function (closure capture)
		isFromAncestor := false
		for ancestor := f.parent; ancestor != nil; ancestor = ancestor.parent {
			if node.function == ancestor {
				isFromAncestor = true
				break
			}
		}
		if isFromAncestor {
			// Create or reuse a capture node for this parent value
			nodes[idx] = f.getOrCreateCaptureNode(node)
		} else {
			// Node from a completely different function (not an ancestor)
			return nil, errors.Errorf(
				"%s: input #%d uses a node from a different function scope",
				name, idx)
		}
	}

	return nodes, nil
}

// nodeParameter data.
type nodeParameter struct {
	name     string
	inputIdx int
}

// EqualNodeData implements nodeDataComparable for nodeParameter.
func (n *nodeParameter) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeParameter)
	return n.name == o.name && n.inputIdx == o.inputIdx
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *compute.ShardingSpec) (compute.Value, error) {
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
		inputIdx: len(f.parameters), // Index within this function's parameters
	}
	n, _ := f.getOrCreateNode(compute.OpTypeParameter, shape, nil, data)
	f.parameters = append(f.parameters, n)
	return n, nil
}

// Constant creates a constant in the function with the given flat values and the shape defined by the dimensions.
func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	_, err := f.verifyAndCastValues("Constant")
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
		inUse: true,
	}
	n, _ := f.getOrCreateNode(compute.OpTypeConstant, shape, nil, data)
	return n, nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []compute.Value, shardings []*compute.ShardingSpec) error {
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

	// If this is a closure or a named function (not main), pre-compile it for efficient execution.
	// Main functions are compiled later in Builder.Compile() after
	// duplicate output handling.
	if f.parent != nil || f.name != compute.MainName {
		compiled, err := newFunctionExecutable(f)
		if err != nil {
			return errors.WithMessagef(err, "failed to compile function %q", f.name)
		}
		f.compiled = compiled
	}

	return nil
}

// Compiled returns the pre-compiled function executable, or nil if not yet compiled.
func (f *Function) Compiled() *FunctionExecutable {
	return f.compiled
}

// CapturedParentNodes returns the list of parent nodes that this closure captures.
// Each entry corresponds to a node from a parent function that this closure uses.
// Returns nil for non-closures or closures that don't capture any values.
func (f *Function) CapturedParentNodes() []*Node {
	return f.capturedParentNodes
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	_, err := f.verifyAndCastValues("Iota")
	if err != nil {
		return nil, err
	}
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node, _ := f.getOrCreateNode(compute.OpTypeIota, shape, nil, iotaAxis)
	return node, nil
}

// Identity implements the compute.Identity interface.
// This operation is not de-duplicated: if you issue it twice, it will not reuse the previous instance.
func (f *Function) Identity(operandOp compute.Value) (compute.Value, error) {
	inputs, err := f.verifyAndCastValues("Reshape", operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	node := f.newNode(compute.OpTypeIdentity, operand.shape, operand)
	return node, nil
}

// Where implements the compute.Builder interface.
func (f *Function) Where(conditionOp, onTrueOp, onFalseOp compute.Value) (compute.Value, error) {
	inputs, err := f.verifyAndCastValues("Where", conditionOp, onTrueOp, onFalseOp)
	if err != nil {
		return nil, err
	}
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape, err := shapeinference.WhereOp(condition.shape, onTrue.shape, onFalse.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(compute.OpTypeWhere, outputShape, []*Node{condition, onTrue, onFalse}, nil)
	return node, nil
}

// Reshape implements the compute.Builder interface.
//
// Notice the compute.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func (f *Function) Reshape(operandOp compute.Value, dims ...int) (compute.Value, error) {
	opType := compute.OpTypeReshape
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReshapeOp(operand.shape, dims)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// Reverse returns x with the values for the given dimensions reversed, that is,
// the value indexed at `i` will be swapped with the value at indexed `(dimension_size - 1 - i)`.
// The shape remains the same.
func (f *Function) Reverse(operandOp compute.Value, axes ...int) (compute.Value, error) {
	opType := compute.OpTypeReverse
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	// Validate axes.
	for _, axis := range axes {
		if axis < 0 || axis >= operand.shape.Rank() {
			return nil, errors.Errorf("Reverse: axis %d out of range for rank %d", axis, operand.shape.Rank())
		}
	}
	// Output shape is the same as the input shape.
	node, _ := f.getOrCreateNode(opType, operand.shape, []*Node{operand}, axes)
	return node, nil
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func (f *Function) Transpose(operandOp compute.Value, permutations ...int) (compute.Value, error) {
	opType := compute.OpTypeTranspose
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.TransposeOp(operand.shape, permutations)
	if err != nil {
		panic(err)
	}
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, permutations)
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
func (f *Function) Broadcast(operandOp compute.Value, prefixDims ...int) (compute.Value, error) {
	opType := compute.OpTypeBroadcast
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.BroadcastOp(operand.shape, prefixDims)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, prefixDims)
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
	operandOp compute.Value,
	outputShape shapes.Shape,
	broadcastAxes []int,
) (compute.Value, error) {
	opType := compute.OpTypeBroadcastInDim
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	err = shapeinference.BroadcastInDimOp(operand.shape, outputShape, broadcastAxes)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, broadcastAxes)
	return node, nil
}

// ReduceMax implements the compute.Builder interface.
func (f *Function) ReduceMax(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceMax, operandOp, axis...)
}

// ReduceMin implements the compute.Builder interface.
func (f *Function) ReduceMin(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceMin, operandOp, axis...)
}

// ReduceSum implements the compute.Builder interface.
func (f *Function) ReduceSum(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceSum, operandOp, axis...)
}

// ReduceProduct implements the compute.Builder interface.
func (f *Function) ReduceProduct(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceProduct, operandOp, axis...)
}

// ReduceBitwiseAnd implements the compute.Builder interface.
func (f *Function) ReduceBitwiseAnd(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceBitwiseAnd, operandOp, axis...)
}

// ReduceBitwiseOr implements the compute.Builder interface.
func (f *Function) ReduceBitwiseOr(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceBitwiseOr, operandOp, axis...)
}

// ReduceBitwiseXor implements the compute.Builder interface.
func (f *Function) ReduceBitwiseXor(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceBitwiseXor, operandOp, axis...)
}

// ReduceLogicalAnd implements the compute.Builder interface.
func (f *Function) ReduceLogicalAnd(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceLogicalAnd, operandOp, axis...)
}

// ReduceLogicalOr implements the compute.Builder interface.
func (f *Function) ReduceLogicalOr(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceLogicalOr, operandOp, axis...)
}

// ReduceLogicalXor implements the compute.Builder interface.
func (f *Function) ReduceLogicalXor(operandOp compute.Value, axis ...int) (compute.Value, error) {
	return f.reduceImpls(compute.OpTypeReduceLogicalXor, operandOp, axis...)
}

func (f *Function) reduceImpls(reduceOpType compute.OpType, operandOp compute.Value, axes ...int) (compute.Value, error) {
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
	node, _ := f.getOrCreateNode(reduceOpType, outputShape, []*Node{operand}, axes)
	return node, nil
}

type gatherNode struct {
	indexVectorAxis                                                  int
	offsetOutputAxes, collapsedSlicesAxes, startIndexMap, sliceSizes []int
	indicesAreSorted                                                 bool
}

// EqualNodeData implements nodeDataComparable for gatherNode.
func (g *gatherNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*gatherNode)
	if g.indexVectorAxis != o.indexVectorAxis || g.indicesAreSorted != o.indicesAreSorted {
		return false
	}
	return slices.Equal(g.offsetOutputAxes, o.offsetOutputAxes) &&
		slices.Equal(g.collapsedSlicesAxes, o.collapsedSlicesAxes) &&
		slices.Equal(g.startIndexMap, o.startIndexMap) &&
		slices.Equal(g.sliceSizes, o.sliceSizes)
}

// Gather implements the compute.Builder.
// It's a complex operation, fully described in the compute.Builder.Gather documentation.
func (f *Function) Gather(
	operandOp, startIndicesOp compute.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (compute.Value, error) {
	opType := compute.OpTypeGather
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
	node, _ := f.getOrCreateNode(opType, shape, []*Node{operand, startIndices}, data)
	return node, nil
}

// Concatenate joins a sequence of tensors along the given axis (it must exist already).
// All input tensors must have the same shape, except potentially in the concatenation dimension.
// They must also have the same data type (DType).
// It returns an error if inputs are invalid (e.g., no inputs, mismatched graphs, shapes, dtypes, or invalid dimension).
func (f *Function) Concatenate(axis int, operandOps ...compute.Value) (compute.Value, error) {
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
	node, _ := f.getOrCreateNode(compute.OpTypeConcatenate, outputShape, operands, axis)
	return node, nil
}

// Bitcast reinterprets the bits of operandOp as targetDType. It implements the compute.Builder interface.
//
// If the element sizes differ, the last dimension is adjusted:
//   - Smaller target: a new trailing axis of size (srcBits / dstBits) is appended, so rank is increased by 1.
//   - Larger target: the last axis must be divisible by (dstBits / srcBits) and is divided by this ratio,
//     but not "squeezed", the rank is always preserved in this case.
func (f *Function) Bitcast(operandOp compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	opType := compute.OpTypeBitcast
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if operand.shape.DType == targetDType {
		return operand, nil
	}

	srcBits := operand.shape.DType.Bits()
	dstBits := targetDType.Bits()
	dims := slices.Clone(operand.shape.Dimensions)

	switch {
	case srcBits == dstBits:
		// Same element size: just change dtype, keep dimensions.
	case srcBits > dstBits:
		// Smaller target: append a trailing axis.
		ratio := srcBits / dstBits
		dims = append(dims, ratio)
	default:
		// Larger target: collapse the last axis.
		ratio := dstBits / srcBits
		lastDim := dims[len(dims)-1]
		if lastDim%ratio != 0 {
			return nil, errors.Errorf("Bitcast: last dim %d not divisible by element-size ratio %d", lastDim, ratio)
		}
		dims[len(dims)-1] = lastDim / ratio
	}
	outputShape := shapes.Make(targetDType, dims...)
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// ConvertDType converts operandOp to the given dtype. It implements the compute.Builder interface.
func (f *Function) ConvertDType(operandOp compute.Value, dtype dtypes.DType) (compute.Value, error) {
	opType := compute.OpTypeConvertDType
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
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// ScatterMax implements the compute.Builder interface.
func (f *Function) ScatterMax(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterMax,
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

// ScatterMin implements the compute.Builder interface.
func (f *Function) ScatterMin(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterMin,
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

// ScatterSum implements the compute.Builder interface.
func (f *Function) ScatterSum(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterSum,
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

// scatterNode is attached to the Node.data field for ScatterMax, ScatterMin, ScatterSum.
type scatterNode struct {
	indexVectorAxis                                                int
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int
	indicesAreSorted, uniqueIndices                                bool
}

// EqualNodeData implements nodeDataComparable for scatterNode.
func (s *scatterNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*scatterNode)
	if s.indexVectorAxis != o.indexVectorAxis ||
		s.indicesAreSorted != o.indicesAreSorted ||
		s.uniqueIndices != o.uniqueIndices {
		return false
	}
	return slices.Equal(s.updateWindowAxes, o.updateWindowAxes) &&
		slices.Equal(s.insertedWindowAxes, o.insertedWindowAxes) &&
		slices.Equal(s.scatterAxesToOperandAxes, o.scatterAxesToOperandAxes)
}

func (f *Function) scatterImpls(
	scatterOpType compute.OpType,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (
	compute.Value, error) {
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
	node, _ := f.getOrCreateNode(scatterOpType, outputShape, []*Node{operand, indices, updates}, data)
	return node, nil
}

// sliceNode is attached to the Node.data field for Slice.
type sliceNode struct {
	starts, limits, strides []int
}

// EqualNodeData implements nodeDataComparable for sliceNode.
func (s *sliceNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*sliceNode)
	return slices.Equal(s.starts, o.starts) &&
		slices.Equal(s.limits, o.limits) &&
		slices.Equal(s.strides, o.strides)
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
func (f *Function) Slice(operandOp compute.Value, starts, limits, strides []int) (compute.Value, error) {
	opType := compute.OpTypeSlice
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
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RNGState or RNGStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func (f *Function) RNGBitGenerator(stateOp compute.Value, shape shapes.Shape) (newState, values compute.Value, err error) {
	opType := compute.OpTypeRNGBitGenerator
	inputs, err := f.verifyAndCastValues(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]
	if !state.shape.Equal(compute.RNGStateShape) {
		err := errors.Errorf(
			"expected random state to be shaped %s, got state.shape=%s instead for RNGBitGenerator",
			compute.RNGStateShape,
			state.shape,
		)
		return nil, nil, err
	}
	outputShapes := []shapes.Shape{
		state.shape.Clone(),
		shape.Clone(),
	}
	node := f.newMultiOutputsNode(opType, outputShapes, state)
	newState = node.multiOutputsNodes[0]
	values = node.multiOutputsNodes[1]
	return
}

// argMinMaxNode with the axis and isMin fields.
type argMinMaxNode struct {
	axis  int
	isMin bool
}

// EqualNodeData implements nodeDataComparable for argMinMaxNode.
func (a *argMinMaxNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*argMinMaxNode)
	return a.axis == o.axis && a.isMin == o.isMin
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
	operandOp compute.Value,
	axis int,
	outputDType dtypes.DType,
	isMin bool,
) (compute.Value, error) {
	opType := compute.OpTypeArgMinMax
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
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

type reduceWindowNode struct {
	reductionType                                             compute.ReduceOpType
	windowDimensions, strides, baseDilations, windowDilations []int
	paddings                                                  [][2]int
}

// EqualNodeData implements nodeDataComparable for reduceWindowNode.
func (r *reduceWindowNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*reduceWindowNode)
	if r.reductionType != o.reductionType {
		return false
	}
	return slices.Equal(r.windowDimensions, o.windowDimensions) &&
		slices.Equal(r.strides, o.strides) &&
		slices.Equal(r.baseDilations, o.baseDilations) &&
		slices.Equal(r.windowDilations, o.windowDilations) &&
		slices.Equal(r.paddings, o.paddings)
}

// ReduceWindow runs a reduction function of reduceType (compute.ReduceOpMax, compute.ReduceOpSum or compute.ReduceOpProduct).
//
// The parameter windowDimensions must be set and have a value for each axis.
// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
// If paddings is nil, they are assumed to be 0.
func (f *Function) ReduceWindow(
	operandOp compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	opType := compute.OpTypeReduceWindow
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
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// ======================================================================================================================
// Unary Operations ----------------------------------------------------------------------------------------------------
// ======================================================================================================================

// Neg implements the compute.Builder interface.
func (f *Function) Neg(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeNeg, operand)
}

// Sign implements the compute.Builder interface.
func (f *Function) Sign(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSign, operand)
}

// Abs implements the compute.Builder interface.
func (f *Function) Abs(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeAbs, operand)
}

// LogicalNot implements the compute.Builder interface.
func (f *Function) LogicalNot(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLogicalNot, operand)
}

// BitwiseNot implements the compute.Builder interface.
func (f *Function) BitwiseNot(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeBitwiseNot, operand)
}

// BitCount implements the compute.Builder interface.
func (f *Function) BitCount(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeBitCount, operand)
}

// Clz implements the compute.Builder interface.
func (f *Function) Clz(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeClz, operand)
}

// Exp implements the compute.Builder interface.
func (f *Function) Exp(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeExp, operand)
}

// Expm1 implements the compute.Builder interface. It returns e(x)-1.
func (f *Function) Expm1(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeExpm1, operand)
}

// Log implements the compute.Builder interface.
func (f *Function) Log(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLog, operand)
}

// Log1p implements the compute.Builder interface.
func (f *Function) Log1p(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLog1p, operand)
}

// Logistic implements the compute.Builder interface. Aka as sigmoid. It returns 1/(1+exp(-x)).
func (f *Function) Logistic(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeLogistic, operand)
}

// Ceil implements the compute.Builder interface.
func (f *Function) Ceil(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCeil, operand)
}

// Floor implements the compute.Builder interface.
func (f *Function) Floor(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeFloor, operand)
}

// Round implements the compute.Builder interface.
func (f *Function) Round(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeRound, operand)
}

// Rsqrt implements the compute.Builder interface.
func (f *Function) Rsqrt(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeRsqrt, operand)
}

// Sqrt implements the compute.Builder interface.
func (f *Function) Sqrt(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSqrt, operand)
}

// Cos implements the compute.Builder interface.
func (f *Function) Cos(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeCos, operand)
}

// Sin implements the compute.Builder interface.
func (f *Function) Sin(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeSin, operand)
}

// Tanh implements the compute.Builder interface.
func (f *Function) Tanh(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeTanh, operand)
}

// Erf implements the compute.Builder interface.
func (f *Function) Erf(operand compute.Value) (compute.Value, error) {
	return f.addUnaryOp(compute.OpTypeErf, operand)
}

// IsFinite implements the compute.Builder interface.
func (f *Function) IsFinite(operandOp compute.Value) (compute.Value, error) {
	opType := compute.OpTypeIsFinite
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
	node, _ := f.getOrCreateNode(opType, shape, []*Node{operand}, nil)
	return node, nil
}

// addUnaryOp adds a generic binary op.
func (f *Function) addUnaryOp(opType compute.OpType, operandOp compute.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	shape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, shape, []*Node{operand}, nil)
	return node, nil
}

// Binary Operations:

// Add implements the compute.Builder interface.
func (f *Function) Add(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeAdd, lhsOp, rhsOp)
}

// Mul implements the compute.Builder interface.
func (f *Function) Mul(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMul, lhsOp, rhsOp)
}

// Sub implements the compute.Builder interface.
func (f *Function) Sub(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeSub, lhsOp, rhsOp)
}

// Div implements the compute.Builder interface.
func (f *Function) Div(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeDiv, lhsOp, rhsOp)
}

// Rem implements the compute.Builder interface.
func (f *Function) Rem(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeRem, lhsOp, rhsOp)
}

// Pow implements the compute.Builder interface.
func (f *Function) Pow(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypePow, lhsOp, rhsOp)
}

// Atan2 implements the compute.Builder interface.
func (f *Function) Atan2(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeAtan2, lhsOp, rhsOp)
}

// BitwiseAnd implements the compute.Builder interface.
func (f *Function) BitwiseAnd(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeBitwiseAnd, lhsOp, rhsOp)
}

// BitwiseOr implements the compute.Builder interface.
func (f *Function) BitwiseOr(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeBitwiseOr, lhsOp, rhsOp)
}

// BitwiseXor implements the compute.Builder interface.
func (f *Function) BitwiseXor(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeBitwiseXor, lhsOp, rhsOp)
}

// ShiftLeft implements the compute.Builder interface.
func (f *Function) ShiftLeft(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeShiftLeft, lhsOp, rhsOp)
}

// ShiftRightArithmetic implements the compute.Builder interface.
func (f *Function) ShiftRightArithmetic(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeShiftRightArithmetic, lhsOp, rhsOp)
}

// ShiftRightLogical implements the compute.Builder interface.
func (f *Function) ShiftRightLogical(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeShiftRightLogical, lhsOp, rhsOp)
}

// LogicalAnd implements the compute.Builder interface.
func (f *Function) LogicalAnd(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalAnd, lhsOp, rhsOp)
}

// LogicalOr implements the compute.Builder interface.
func (f *Function) LogicalOr(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalOr, lhsOp, rhsOp)
}

// LogicalXor implements the compute.Builder interface.
func (f *Function) LogicalXor(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeLogicalXor, lhsOp, rhsOp)
}

// Max implements the compute.Builder interface.
func (f *Function) Max(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMax, lhsOp, rhsOp)
}

// Min implements the compute.Builder interface.
func (f *Function) Min(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addBinaryOp(compute.OpTypeMin, lhsOp, rhsOp)
}

// Equal implements the compute.Builder interface.
func (f *Function) Equal(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeEqual, lhsOp, rhsOp)
}

// NotEqual implements the compute.Builder interface.
func (f *Function) NotEqual(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeNotEqual, lhsOp, rhsOp)
}

// GreaterOrEqual implements the compute.Builder interface.
func (f *Function) GreaterOrEqual(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeGreaterOrEqual, lhsOp, rhsOp)
}

// GreaterThan implements the compute.Builder interface.
func (f *Function) GreaterThan(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeGreaterThan, lhsOp, rhsOp)
}

// LessOrEqual implements the compute.Builder interface.
func (f *Function) LessOrEqual(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeLessOrEqual, lhsOp, rhsOp)
}

// LessThan implements the compute.Builder interface.
func (f *Function) LessThan(lhsOp, rhsOp compute.Value) (compute.Value, error) {
	return f.addComparisonOp(compute.OpTypeLessThan, lhsOp, rhsOp)
}

// addBinaryOp adds a generic binary op.
func (f *Function) addBinaryOp(opType compute.OpType, lhsOp, rhsOp compute.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.BinaryOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, shape, []*Node{lhs, rhs}, nil)
	return node, nil
}

// addComparisonOp adds a generic comparison binary op.
func (f *Function) addComparisonOp(opType compute.OpType, lhsOp, rhsOp compute.Value) (*Node, error) {
	inputs, err := f.verifyAndCastValues(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.ComparisonOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.getOrCreateNode(opType, shape, []*Node{lhs, rhs}, nil)
	return node, nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (f *Function) Clamp(minV, x, maxV compute.Value) (compute.Value, error) {
	clamped, err := f.Max(minV, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	clamped, err = f.Min(clamped, maxV)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}

// IsNaN implements compute.Builder interface.
func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	result, err := f.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}

// AllReduce implements the compute.CollectiveOps interface.
func (f *Function) AllReduce(_ []compute.Value, _ compute.ReduceOpType, _ [][]int) ([]compute.Value, error) {
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"AllReduce not supported for %q builder", BackendName)
}

// Sort sorts one or more tensors along the specified axis using a comparator closure.
//
// The comparator closure takes 2*N scalar parameters (lhs_0, rhs_0, lhs_1, rhs_1, ...)
// where N is the number of input tensors, and returns a scalar boolean indicating
// whether lhs should come before rhs.
func (f *Function) Sort(comparator compute.Function, axis int, isStable bool, inputs ...compute.Value) ([]compute.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	if len(inputs) == 0 {
		return nil, errors.Errorf("Sort: requires at least one input tensor")
	}

	// Validate inputs
	inputNodes, err := f.verifyAndCastValues("Sort", inputs...)
	if err != nil {
		return nil, err
	}

	// Validate comparator closure
	compFn, err := f.validateClosure("Sort", "comparator", comparator)
	if err != nil {
		return nil, err
	}

	// Verify all inputs have the same dimensions
	firstShape := inputNodes[0].shape
	for i, node := range inputNodes[1:] {
		if !shapesEqualDimensions(firstShape, node.shape) {
			return nil, errors.Errorf("Sort: all inputs must have the same dimensions, input 0 has %s, input %d has %s",
				firstShape, i+1, node.shape)
		}
	}

	// Normalize axis
	rank := firstShape.Rank()
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("Sort: axis %d out of range for rank %d", axis, rank)
	}

	// Verify comparator has 2*N scalar parameters
	expectedParams := 2 * len(inputNodes)
	if len(compFn.parameters) != expectedParams {
		return nil, errors.Errorf("Sort: comparator must have %d parameters (2 per input), got %d",
			expectedParams, len(compFn.parameters))
	}

	// Verify comparator parameters are scalars with correct dtypes
	for i, node := range inputNodes {
		expectedDType := node.shape.DType
		for j, side := range []string{"lhs", "rhs"} {
			paramIdx := 2*i + j
			param := compFn.parameters[paramIdx]
			if param.shape.Rank() != 0 {
				return nil, errors.Errorf("Sort: comparator parameter %d (%s_%d) must be scalar, got %s",
					paramIdx, side, i, param.shape)
			}
			if param.shape.DType != expectedDType {
				return nil, errors.Errorf("Sort: comparator parameter %d (%s_%d) must have dtype %s, got %s",
					paramIdx, side, i, expectedDType, param.shape.DType)
			}
		}
	}

	// Verify comparator returns a scalar boolean
	if len(compFn.outputs) != 1 {
		return nil, errors.Errorf("Sort: comparator must return exactly one value, got %d", len(compFn.outputs))
	}
	if compFn.outputs[0].shape.Rank() != 0 || compFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("Sort: comparator must return a scalar boolean, got %s", compFn.outputs[0].shape)
	}

	// Create output shapes (same as inputs)
	outputShapes := make([]shapes.Shape, len(inputNodes))
	for i, node := range inputNodes {
		outputShapes[i] = node.shape.Clone()
	}

	data := &sortNode{
		comparator: compFn,
		axis:       axis,
		isStable:   isStable,
		inputCount: len(inputNodes),
	}

	// Create multi-output node for Sort with only input tensors as regular inputs.
	// Captured values are tracked separately via AddNodeCapturedInputs.
	node := f.newMultiOutputsNode(compute.OpTypeSort, outputShapes, inputNodes...)
	node.data = data

	// Add captured values from comparator to node.capturedInputs.
	node.AddNodeCapturedInputs(compFn)

	return node.MultiOutputValues(), nil
}

// sortNode holds the data for a Sort operation.
type sortNode struct {
	comparator *Function
	axis       int
	isStable   bool
	inputCount int // Number of input tensors
}

// shapesEqualDimensions returns true if two shapes have the same dimensions (ignoring dtype).
func shapesEqualDimensions(a, b shapes.Shape) bool {
	if a.Rank() != b.Rank() {
		return false
	}
	for i := range a.Dimensions {
		if a.Dimensions[i] != b.Dimensions[i] {
			return false
		}
	}
	return true
}

// Pad implements the compute.Builder interface.
func (f *Function) Pad(operandOp, fillValueOp compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	opType := compute.OpTypePad
	inputs, err := f.verifyAndCastValues(opType.String(), operandOp, fillValueOp)
	if err != nil {
		return nil, err
	}
	operand, fillValue := inputs[0], inputs[1]

	outputShape, err := shapeinference.PadOp(operand.shape, axesConfig...)
	if err != nil {
		return nil, err
	}

	data := &padNode{axesConfig: slices.Clone(axesConfig)}
	node, _ := f.getOrCreateNode(opType, outputShape, []*Node{operand, fillValue}, data)
	return node, nil
}

type padNode struct {
	axesConfig []compute.PadAxis
}

// EqualNodeData implements nodeDataComparable for padNode.
func (p *padNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*padNode)
	return slices.Equal(p.axesConfig, o.axesConfig)
}
