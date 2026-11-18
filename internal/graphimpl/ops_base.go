// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// ReduceOpType aliases compute.ReduceOpType for internal ops.
type ReduceOpType = compute.ReduceOpType

// MinConstValueSizeToKeep defines the element-count threshold below which constant values
// are retained on the node for debugging/introspection.
var MinConstValueSizeToKeep = 32

func mustNoError[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// MustAdjustAxis returns the positive axis for operand's shape, adjusting negative axes.
func MustAdjustAxis(axis int, operand shapes.HasShape) int {
	adjustedAxis := axis
	if axis < 0 {
		adjustedAxis = operand.Shape().Rank() + axis
	}
	if adjustedAxis < 0 || adjustedAxis >= operand.Shape().Rank() {
		exceptions.Panicf("invalid axis %d, operand rank is %d", axis, operand.Shape().Rank())
	}
	return adjustedAxis
}

// ValidateBuildingGraphFromInputs checks that all inputs belong to the same building GraphImpl.
func ValidateBuildingGraphFromInputs(inputs ...*NodeImpl) (g *GraphImpl) {
	if len(inputs) == 0 {
		exceptions.Panicf("no input nodes provided, at least one is required")
	}
	for ii, n := range inputs {
		if err := exceptions.TryCatch[error](n.AssertValid); err != nil {
			panic(errors.WithMessagef(err, "invalid input[%d]", ii))
		}
		if n.NumOutputs() != 1 {
			exceptions.Panicf(
				"input[%d](%s) has multiple-outputs, it has to be split with selectOutput calls first",
				ii, n,
			)
		}
		if g == nil {
			g = n.Graph()
			g.AssertBuilding()
		} else if n.Graph() != g {
			exceptions.Panicf("combining nodes from different graphs not allowed: "+
				"input[0] graph is %q, input[%d] graph is %q", g.Name(), ii, n.Graph().Name())
		}
	}
	return
}

// NodeInputsParameter holds the inputs for compute.Parameter.
type NodeInputsParameter struct {
	Name     string
	Shape    shapes.Shape
	Sharding *distributed.ShardingSpec
	Handle   ParameterHandle
}

func (ni *NodeInputsParameter) Type() NodeType { return NodeTypeParameter }
func (ni *NodeInputsParameter) String() string {
	return fmt.Sprintf("%s(name=%q, shape=%s)", ni.Type(), ni.Name, ni.Shape)
}
func (ni *NodeInputsParameter) CloneWithInputs(originalNode *NodeImpl, _ ...*NodeImpl) *NodeImpl {
	return originalNode
}

// Parameter registers an input parameter on g.
func Parameter(g *GraphImpl, name string, shape shapes.Shape) *NodeImpl {
	return ShardedParameter(g, name, shape, nil)
}

// ShardedParameter registers a parameter with an optional sharding specification on g.
func ShardedParameter(g *GraphImpl, name string, shape shapes.Shape, sharding *distributed.ShardingSpec) (node *NodeImpl) {
	g.AssertBuilding()
	if sharding != nil && g.distStrategy == distributed.None {
		exceptions.Panicf(
			"cannot set sharding spec for parameter %q in graph %q when distributed strategy is %q",
			name, g.name, g.distStrategy,
		)
	}
	if sharding != nil && slices.Index(g.deviceMeshes, sharding.Mesh) == -1 {
		exceptions.Panicf(
			"sharding spec for parameter %q in graph %q specifies mesh %q that was not registered with the "+
				"Graph -- all meshes used in sharding specifications must be declared in the Graph upfront when "+
				"configuring AutoSharding",
			name, g.name, sharding.Mesh,
		)
	}
	handle := ParameterHandle(len(g.currentFunc.parameters))
	if name == "" {
		name = fmt.Sprintf("parameter_#%d", handle)
	}
	if _, ok := g.currentFunc.parameterNameToHandle[name]; ok {
		exceptions.Panicf("requested parameter with name %q for graph %q already exists", name, g.name)
	}
	nodeInputs := &NodeInputsParameter{
		Name:     name,
		Shape:    shape,
		Sharding: sharding,
		Handle:   handle,
	}
	result, err := g.currentFunc.backendFunc.Parameter(nodeInputs.Name, nodeInputs.Shape, sharding.ToBackendsSpec())
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create parameter %q", name))
	}
	node = &NodeImpl{
		graph:        g,
		outputOps:    []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       nodeInputs,
	}
	g.registerNode(node)
	g.currentFunc.parameters = append(g.currentFunc.parameters, node)
	g.currentFunc.parameterNameToHandle[name] = handle
	return
}

// NodeInputsSplitNode holds the inputs for one element of a multi-output node.
type NodeInputsSplitNode struct {
	MultiOutputNode *NodeImpl
	Index           int
}

func (ni *NodeInputsSplitNode) Type() NodeType { return NodeTypeSplitNode }
func (ni *NodeInputsSplitNode) String() string {
	return fmt.Sprintf("%s(multiOutputNode=[#%d], index=%d)", ni.Type(), ni.MultiOutputNode.Id(), ni.Index)
}
func (ni *NodeInputsSplitNode) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	newMultiOutputNode := newInputs[0]
	g := newMultiOutputNode.Graph()
	inputs := &NodeInputsSplitNode{
		MultiOutputNode: newMultiOutputNode,
		Index:           ni.Index,
	}
	node := &NodeImpl{
		outputOps:    []compute.Value{newMultiOutputNode.outputOps[ni.Index]},
		outputShapes: []shapes.Shape{newMultiOutputNode.outputShapes[ni.Index]},
		graph:        g,
		inputs:       inputs,
		inputNodes:   []*NodeImpl{newMultiOutputNode},
	}
	g.registerNode(node)
	return node
}

// SplitNode splits a multi-output NodeImpl into single-output NodeImpls.
func SplitNode(multiOutputNode *NodeImpl) (splitNodes []*NodeImpl) {
	if multiOutputNode.NumOutputs() == 1 {
		return []*NodeImpl{multiOutputNode}
	}
	if multiOutputNode.NumOutputs() == 0 {
		exceptions.Panicf("SplitNode expects at least one output -- got %d outputs", multiOutputNode.NumOutputs())
	}
	g := multiOutputNode.Graph()
	g.AssertBuilding()
	splitNodes = make([]*NodeImpl, 0, multiOutputNode.NumOutputs())
	for ii, op := range multiOutputNode.outputOps {
		inputs := &NodeInputsSplitNode{
			MultiOutputNode: multiOutputNode,
			Index:           ii,
		}
		inputNodes := []*NodeImpl{multiOutputNode}
		node := &NodeImpl{
			outputOps:    []compute.Value{op},
			outputShapes: []shapes.Shape{multiOutputNode.outputShapes[ii]},
			graph:        g,
			inputs:       inputs,
			inputNodes:   inputNodes,
		}
		g.registerNode(node)
		splitNodes = append(splitNodes, node)
	}
	return
}

// NodeInputsConstant holds the inputs for a constant node.
type NodeInputsConstant struct {
	Shape  shapes.Shape
	Tensor *tensors.Tensor
}

func (ni *NodeInputsConstant) Type() NodeType { return NodeTypeConstant }
func (ni *NodeInputsConstant) String() string {
	if ni.Tensor == nil {
		return fmt.Sprintf("%s(%s)", ni.Type(), ni.Shape)
	}
	return fmt.Sprintf("%s(%s: %v)", ni.Type(), ni.Shape, ni.Tensor.Value())
}
func (ni *NodeInputsConstant) CloneWithInputs(originalNode *NodeImpl, _ ...*NodeImpl) *NodeImpl {
	return originalNode
}

func newConstantNode(g *GraphImpl, result compute.Value, nodeInputs *NodeInputsConstant) *NodeImpl {
	node := &NodeImpl{
		graph:        g,
		outputOps:    []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       nodeInputs,
	}
	g.registerNode(node)
	return node
}

// ConstTensor returns a cached constant node for t in the current function scope.
func ConstTensor(g *GraphImpl, t *tensors.Tensor) (node *NodeImpl) {
	g.AssertBuilding()
	f := g.currentFunc
	if cached := f.tensorConstants.get(t); cached != nil {
		return cached
	}
	version := t.Version()
	node = constTensorUncached(g, t)
	if t != nil && t.Version() == version {
		f.tensorConstants.set(t, node, version)
	}
	return node
}

func constTensorUncached(g *GraphImpl, t *tensors.Tensor) *NodeImpl {
	g.AssertBuilding()
	nodeInputs := &NodeInputsConstant{
		Shape: t.Shape(),
	}
	if t.Size() < MinConstValueSizeToKeep {
		var err error
		nodeInputs.Tensor, err = t.LocalClone()
		if err != nil {
			panic(errors.WithMessagef(err, "ConstTensor failed to create a local clone of the tensor in the graph"))
		}
	}
	var result compute.Value
	var err error
	t.MustConstFlatData(func(flat any) {
		result, err = g.currentFunc.backendFunc.Constant(flat, nodeInputs.Shape.Dimensions...)
	})
	if err != nil {
		panic(errors.WithMessagef(err, "ConstTensor failed to create a constant in the backend"))
	}
	return newConstantNode(g, result, nodeInputs)
}

// ConstDonatedTensor creates a constant node adopting t's device buffer when possible, finalizing t.
func ConstDonatedTensor(g *GraphImpl, t *tensors.Tensor) *NodeImpl {
	g.AssertBuilding()
	if err := t.CheckValid(); err != nil {
		panic(errors.WithMessagef(err, "ConstDonatedTensor received invalid tensor"))
	}
	var node *NodeImpl
	if t.IsOnAnyDevice() {
		if tBackend, err := t.Backend(); err == nil && tBackend == g.Backend() {
			buf, err := t.DonateBuffer(g.Backend(), 0)
			if err != nil {
				panic(errors.WithMessagef(err, "ConstDonatedTensor failed to donate buffer from tensor"))
			}
			node = ConstBuffer(g, buf, true)
		}
	}
	if node == nil {
		node = constTensorUncached(g, t)
	}
	if err := t.FinalizeAll(); err != nil {
		panic(errors.WithMessagef(err, "ConstDonatedTensor failed to finalize donated tensor"))
	}
	return node
}

// ConstBuffer returns a newly created constant node backed by buf.
func ConstBuffer(g *GraphImpl, buf compute.Buffer, donate bool) *NodeImpl {
	g.AssertBuilding()
	if buf == nil {
		exceptions.Panicf("ConstBuffer received nil buffer")
	}
	if buf.Backend() != nil && buf.Backend() != g.Backend() {
		exceptions.Panicf("ConstBuffer received buffer from backend %q, but graph is on backend %q",
			buf.Backend().Name(), g.Backend().Name())
	}
	shape, err := buf.Shape()
	if err != nil {
		panic(errors.WithMessagef(err, "ConstBuffer failed to get buffer shape"))
	}
	result, err := g.currentFunc.backendFunc.ConstantFromBuffer(buf, donate)
	if err != nil && compute.IsNotImplemented(err) {
		var flat any
		if shape.IsZeroSize() {
			flat = dtypes.MakeAnySlice(shape.DType, 0)
		} else {
			flat = dtypes.MakeAnySlice(shape.DType, shape.Size())
			if toFlatErr := buf.ToFlatData(flat); toFlatErr != nil {
				panic(errors.WithMessagef(toFlatErr, "ConstBuffer fallback failed to transfer buffer data"))
			}
		}
		result, err = g.currentFunc.backendFunc.Constant(flat, shape.Dimensions...)
		if err == nil && donate {
			_ = buf.Finalize()
		}
	}
	if err != nil {
		panic(errors.WithMessagef(err, "ConstBuffer failed to create a constant in the backend"))
	}
	return newConstantNode(g, result, &NodeInputsConstant{Shape: shape})
}

// SharedConstant creates a Constant node reusing a shared backend constant compute.Value.
func SharedConstant(g *GraphImpl, val compute.Value) *NodeImpl {
	g.AssertBuilding()
	shape := mustNoError(g.builder.OpShape(val))
	return newConstantNode(g, val, &NodeInputsConstant{Shape: shape})
}

// FinalizeConstantValue releases backend resources held by a shared constant compute.Value.
func FinalizeConstantValue(val compute.Value) {
	if val == nil {
		return
	}
	if finalizer, ok := val.(interface{ Finalize() }); ok {
		finalizer.Finalize()
	}
}

// Const creates a constant node in g from a tensor, scalar, or multidimensional slice.
func Const(g *GraphImpl, x any) *NodeImpl {
	if _, ok := x.(*NodeImpl); ok {
		exceptions.Panicf(
			"Const(g, x) can only take actual values, not another computation graph node -- " +
				"for that you don't need Const(), just use it directly.")
	}
	if t, ok := x.(*tensors.Tensor); ok {
		return ConstTensor(g, t)
	}
	t := tensors.MustFromAnyValue(x)
	return constTensorUncached(g, t)
}

// StopGradient creates an Identity node with stopGradient set to true.
func StopGradient(x *NodeImpl) *NodeImpl {
	n := Identity(x)
	n.stopGradient = true
	return n
}

// OptimizationBarrier prevents the compiler from optimizing across operand.
func OptimizationBarrier(operand *NodeImpl) *NodeImpl {
	return BackendOptimizationBarrier(operand)[0]
}

// OptimizationBarriers creates an optimization barrier for multiple operands.
func OptimizationBarriers(operands ...*NodeImpl) []*NodeImpl {
	if len(operands) == 0 {
		return nil
	}
	return BackendOptimizationBarrier(operands...)
}

// BatchNormForTraining wraps the backend BatchNormForTraining operation.
func BatchNormForTraining(operand, scale, offset *NodeImpl, epsilon float32, axis int) (normalized, batchMean, batchVariance *NodeImpl) {
	_ = ValidateBuildingGraphFromInputs(operand, scale, offset)
	dtype := operand.DType()
	if scale.DType() != dtype || offset.DType() != dtype {
		exceptions.Panicf(
			"BatchNormForTraining: operand (%s), scale (%s) and offset (%s) must all have the same DType",
			operand.Shape(), scale.Shape(), offset.Shape(),
		)
	}
	axis = MustAdjustAxis(axis, operand)
	return BackendBatchNormForTraining(operand, scale, offset, epsilon, axis)
}

// BatchNormForInference wraps the backend BatchNormForInference operation.
func BatchNormForInference(operand, scale, offset, mean, variance *NodeImpl, epsilon float32, axis int) *NodeImpl {
	_ = ValidateBuildingGraphFromInputs(operand, scale, offset, mean, variance)
	dtype := operand.DType()
	if scale.DType() != dtype || offset.DType() != dtype {
		exceptions.Panicf(
			"BatchNormForInference: operand (%s), scale (%s) and offset (%s) must all have the same DType",
			operand.Shape(), scale.Shape(), offset.Shape(),
		)
	}
	axis = MustAdjustAxis(axis, operand)
	return BackendBatchNormForInference(operand, scale, offset, mean, variance, epsilon, axis)
}

// BatchNormGradient wraps the backend BatchNormGradient operation.
func BatchNormGradient(operand, scale, mean, variance, gradOutput *NodeImpl, epsilon float32, axis int) (gradOperand, gradScale, gradOffset *NodeImpl) {
	_ = ValidateBuildingGraphFromInputs(operand, scale, mean, variance)
	dtype := operand.DType()
	if scale.DType() != dtype || mean.DType() != dtype || variance.DType() != dtype {
		exceptions.Panicf(
			"BatchNormGradient: operand (%s), scale (%s), mean (%s) and variance (%s) must all have the same DType",
			operand.Shape(), scale.Shape(), mean.Shape(), variance.DType(),
		)
	}
	axis = MustAdjustAxis(axis, operand)
	return BackendBatchNormGradient(operand, scale, mean, variance, gradOutput, epsilon, axis)
}

// NodeTypeWhile is the NodeType for While operations.
const NodeTypeWhile NodeType = 1002

// NodeInputsWhile holds the inputs for a While operation.
type NodeInputsWhile struct {
	Cond         *FunctionImpl
	Body         *FunctionImpl
	InitialState []*NodeImpl
}

func (ni *NodeInputsWhile) Type() NodeType { return NodeTypeWhile }
func (ni *NodeInputsWhile) String() string { return "While" }
func (ni *NodeInputsWhile) CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	g := originalNode.Graph()
	inputValues := make([]compute.Value, len(newInputs))
	for i, input := range newInputs {
		inputValues[i] = input.outputOps[0]
	}
	results, err := g.currentFunc.backendFunc.While(ni.Cond.backendFunc, ni.Body.backendFunc, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "While operation failed"))
	}
	newNi := &NodeInputsWhile{
		Cond:         ni.Cond,
		Body:         ni.Body,
		InitialState: newInputs,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       newNi,
		inputNodes:   newInputs,
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return node
}

// While executes a loop while cond returns true.
func While(cond, body *FunctionImpl, initialState ...*NodeImpl) []*NodeImpl {
	if len(initialState) == 0 {
		exceptions.Panicf("While requires at least one initial state value")
	}
	if cond == nil {
		exceptions.Panicf("While requires a condition function")
	}
	if body == nil {
		exceptions.Panicf("While requires a body function")
	}
	if !cond.IsClosure() {
		exceptions.Panicf("While condition must be a closure (created with NewClosure)")
	}
	if !body.IsClosure() {
		exceptions.Panicf("While body must be a closure (created with NewClosure)")
	}

	g := initialState[0].graph
	g.AssertBuilding()
	ValidateBuildingGraphFromInputs(initialState...)

	inputValues := make([]compute.Value, len(initialState))
	for i, input := range initialState {
		inputValues[i] = input.outputOps[0]
	}

	results, err := g.currentFunc.backendFunc.While(cond.backendFunc, body.backendFunc, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "While operation failed"))
	}

	ni := &NodeInputsWhile{
		Cond:         cond,
		Body:         body,
		InitialState: initialState,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   initialState,
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return SplitNode(node)
}

// NodeTypeIf is the NodeType for If operations.
const NodeTypeIf NodeType = 1003

// NodeInputsIf holds the inputs for an If operation.
type NodeInputsIf struct {
	Pred        *NodeImpl
	TrueBranch  *FunctionImpl
	FalseBranch *FunctionImpl
}

func (ni *NodeInputsIf) Type() NodeType { return NodeTypeIf }
func (ni *NodeInputsIf) String() string { return "If" }
func (ni *NodeInputsIf) CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	g := originalNode.Graph()
	newPred := newInputs[0]
	results, err := g.currentFunc.backendFunc.If(newPred.outputOps[0], ni.TrueBranch.backendFunc, ni.FalseBranch.backendFunc)
	if err != nil {
		panic(errors.WithMessagef(err, "If operation failed"))
	}
	newNi := &NodeInputsIf{
		Pred:        newPred,
		TrueBranch:  ni.TrueBranch,
		FalseBranch: ni.FalseBranch,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       newNi,
		inputNodes:   []*NodeImpl{newPred},
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return node
}

// If executes trueBranch or falseBranch based on scalar boolean pred.
func If(pred *NodeImpl, trueBranch, falseBranch *FunctionImpl) []*NodeImpl {
	if pred == nil {
		exceptions.Panicf("If requires a predicate")
	}
	if trueBranch == nil {
		exceptions.Panicf("If requires a trueBranch function")
	}
	if falseBranch == nil {
		exceptions.Panicf("If requires a falseBranch function")
	}
	if !trueBranch.IsClosure() {
		exceptions.Panicf("If trueBranch must be a closure (created with NewClosure)")
	}
	if !falseBranch.IsClosure() {
		exceptions.Panicf("If falseBranch must be a closure (created with NewClosure)")
	}

	g := pred.graph
	g.AssertBuilding()

	predShape := pred.Shape()
	if !predShape.IsScalar() || predShape.DType != dtypes.Bool {
		exceptions.Panicf("If: pred must be a scalar boolean, got %s", predShape)
	}

	results, err := g.currentFunc.backendFunc.If(pred.outputOps[0], trueBranch.backendFunc, falseBranch.backendFunc)
	if err != nil {
		panic(errors.WithMessagef(err, "If operation failed"))
	}

	ni := &NodeInputsIf{
		Pred:        pred,
		TrueBranch:  trueBranch,
		FalseBranch: falseBranch,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   []*NodeImpl{pred},
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return SplitNode(node)
}

// NodeTypeSort is the NodeType for Sort operations.
const NodeTypeSort NodeType = 1001

// NodeInputsSort holds the inputs for a Sort operation.
type NodeInputsSort struct {
	Comparator *FunctionImpl
	Axis       int
	IsStable   bool
	Inputs     []*NodeImpl
}

func (ni *NodeInputsSort) Type() NodeType { return NodeTypeSort }
func (ni *NodeInputsSort) String() string { return "Sort" }
func (ni *NodeInputsSort) CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	g := originalNode.Graph()
	inputValues := make([]compute.Value, len(newInputs))
	for i, input := range newInputs {
		inputValues[i] = input.outputOps[0]
	}
	results, err := g.currentFunc.backendFunc.Sort(ni.Comparator.backendFunc, ni.Axis, ni.IsStable, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "SortFunc operation failed"))
	}
	newNi := &NodeInputsSort{
		Comparator: ni.Comparator,
		Axis:       ni.Axis,
		IsStable:   ni.IsStable,
		Inputs:     newInputs,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       newNi,
		inputNodes:   newInputs,
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return node
}

// SortFunc sorts one or more tensors along axis using comparator.
func SortFunc(comparator *FunctionImpl, axis int, isStable bool, inputs ...*NodeImpl) []*NodeImpl {
	if len(inputs) == 0 {
		exceptions.Panicf("SortFunc requires at least one input tensor")
	}
	if comparator == nil {
		exceptions.Panicf("SortFunc requires a comparator function")
	}
	if !comparator.IsClosure() {
		exceptions.Panicf("SortFunc comparator must be a closure (created with NewClosure)")
	}

	g := inputs[0].graph
	g.AssertBuilding()
	ValidateBuildingGraphFromInputs(inputs...)

	shape := inputs[0].Shape()
	for i, input := range inputs[1:] {
		if !input.Shape().EqualDimensions(shape) {
			exceptions.Panicf("SortFunc: all inputs must have the same dimensions, input 0 has %s, input %d has %s",
				shape, i+1, input.Shape())
		}
	}
	axis = MustAdjustAxis(axis, inputs[0])

	inputValues := make([]compute.Value, len(inputs))
	for i, input := range inputs {
		inputValues[i] = input.outputOps[0]
	}

	results, err := g.currentFunc.backendFunc.Sort(comparator.backendFunc, axis, isStable, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "SortFunc operation failed"))
	}

	ni := &NodeInputsSort{
		Comparator: comparator,
		Axis:       axis,
		IsStable:   isStable,
		Inputs:     inputs,
	}
	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   inputs,
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return SplitNode(node)
}
