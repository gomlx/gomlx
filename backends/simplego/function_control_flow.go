// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// getOrCreateCaptureNode returns a capture node for the given parent node.
// If the parent node has already been captured, returns the existing capture node.
// Otherwise, creates a new capture node and adds it to the captured values list.
//
// For nested closures (grandparent captures), this recursively propagates the
// capture through intermediate closures. For example, if closure C (child of B,
// child of A) wants to capture a value from A, this will:
// 1. Have B capture the value from A
// 2. Have C capture B's capture node
//
// This ensures that when If/While/Sort ops are built, they can properly set up
// their capturedInputs by looking at the closure's capturedParentNodes.
func (f *Function) getOrCreateCaptureNode(parentNode *Node) *Node {
	// Check if we've already captured this node
	for i, captured := range f.capturedParentNodes {
		if captured == parentNode {
			return f.capturedLocalNodes[i]
		}
	}

	// Determine the actual node to capture.
	// If parentNode is not from our direct parent, we need to propagate through
	// intermediate closures.
	nodeToCapture := parentNode
	if f.parent == nil {
		// This should never happen: if we're capturing a node, f must be a closure
		// with a parent function. If parent is nil, the node is not from an ancestor.
		panic(errors.Errorf(
			"getOrCreateCaptureNode: function %q has no parent but is trying to capture node from function %q",
			f.name, parentNode.function.name))
	}
	if parentNode.function != f.parent {
		// The node is from a grandparent or further ancestor.
		// First, have our parent capture it, then we capture the parent's capture node.
		parentCaptureNode := f.parent.getOrCreateCaptureNode(parentNode)
		nodeToCapture = parentCaptureNode
	}

	// Create a new capture node
	captureIdx := len(f.capturedParentNodes)
	captureNode := f.newNode(backends.OpTypeCapturedValue, parentNode.shape)
	captureNode.data = capturedNodeData(captureIdx)

	f.capturedParentNodes = append(f.capturedParentNodes, nodeToCapture)
	f.capturedLocalNodes = append(f.capturedLocalNodes, captureNode)

	return captureNode
}

// AddNodeCapturedInputs adds captured inputs from a closure to this node.
// This should be called when building ops like If, While, Sort that use closures.
// For ops with multiple closures, call this once for each closure.
// Each closure's captured values are stored as a separate slice in node.capturedInputs,
// preserving the per-closure grouping for execution.
//
// For nested closures, if the closure captures values from a grandparent,
// those values are propagated to the parent closure's required captures.
func (n *Node) AddNodeCapturedInputs(closure *Function) {
	if closure == nil {
		// Add empty slice to maintain closure index alignment.
		n.capturedInputs = append(n.capturedInputs, nil)
		return
	}

	// Append the closure's captured values as a new slice.
	// These become dependencies of the node in the parent function's DAG.
	capturedNodes := make([]*Node, len(closure.capturedParentNodes))
	copy(capturedNodes, closure.capturedParentNodes)
	n.capturedInputs = append(n.capturedInputs, capturedNodes)
}

// Call creates nodes representing a call to the target function with the given inputs.
// The target function must be a named function from the same builder that has been compiled.
func (f *Function) Call(target backends.Function, inputs ...backends.Value) ([]backends.Value, error) {
	inputNodes, err := f.verifyAndCastValues("Call", inputs...)
	if err != nil {
		return nil, err
	}

	targetFn, ok := target.(*Function)
	if !ok {
		return nil, errors.Errorf("Call: target function must be a *simplego.Function, got %T", target)
	}
	if targetFn.builder != f.builder {
		return nil, errors.Errorf("Call: target function must be from the same builder")
	}
	if !targetFn.returned {
		return nil, errors.Errorf("Call: target function %q must have Return() called", targetFn.name)
	}
	if targetFn.compiled == nil {
		return nil, errors.Errorf("Call: target function %q must be compiled", targetFn.name)
	}

	// Validate input count and shapes
	if len(inputNodes) != len(targetFn.parameters) {
		return nil, errors.Errorf("Call: function %q expects %d parameters, got %d inputs",
			targetFn.name, len(targetFn.parameters), len(inputNodes))
	}
	for i, param := range targetFn.parameters {
		if !param.shape.Equal(inputNodes[i].shape) {
			return nil, errors.Errorf("Call: function %q parameter %d shape %s doesn't match input shape %s",
				targetFn.name, i, param.shape, inputNodes[i].shape)
		}
	}

	// Create output shapes from target function's outputs
	outputShapes := make([]shapes.Shape, len(targetFn.outputs))
	for i, out := range targetFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	data := &callNode{
		target: targetFn,
	}

	node := f.newMultiOutputsNode(backends.OpTypeCall, outputShapes, inputNodes...)
	node.data = data

	return node.MultiOutputValues(), nil
}

// callNode holds the data for a Call operation.
type callNode struct {
	target *Function
}

// validateClosure validates that a backends.Function is a compiled closure of the current function.
func (f *Function) validateClosure(opName, closureName string, closure backends.Function) (*Function, error) {
	fn, ok := closure.(*Function)
	if !ok {
		return nil, errors.Errorf("%s: %s must be a *simplego.Function, got %T", opName, closureName, closure)
	}
	if fn.parent != f {
		return nil, errors.Errorf("%s: %s must be a closure of the current function", opName, closureName)
	}
	if !fn.returned {
		return nil, errors.Errorf("%s: %s must have Return() called", opName, closureName)
	}
	if fn.compiled == nil {
		return nil, errors.Errorf("%s: %s must be compiled", opName, closureName)
	}
	return fn, nil
}

// checkClosureParams verifies that a closure's parameters match expected shapes.
func checkClosureParams(opName, closureName string, fn *Function, expected []*Node) error {
	if len(fn.parameters) != len(expected) {
		return errors.Errorf("%s: %s must have %d parameters, got %d",
			opName, closureName, len(expected), len(fn.parameters))
	}
	for i, param := range fn.parameters {
		if !param.shape.Equal(expected[i].shape) {
			return errors.Errorf("%s: %s parameter %d shape %s must match expected shape %s",
				opName, closureName, i, param.shape, expected[i].shape)
		}
	}
	return nil
}

// If executes one of two branches based on a boolean predicate.
//
// The predicate must be a scalar boolean. The true and false branches are closures
// that take no parameters and return the same number of outputs with matching shapes.
func (f *Function) If(pred backends.Value, trueBranch, falseBranch backends.Function) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	// Validate predicate
	predNodes, err := f.verifyAndCastValues("If", pred)
	if err != nil {
		return nil, err
	}
	predNode := predNodes[0]

	// Verify pred is a scalar boolean
	if predNode.shape.Rank() != 0 || predNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("If: pred must be a scalar boolean, got %s", predNode.shape)
	}

	// Validate branches
	trueFn, err := f.validateClosure("If", "trueBranch", trueBranch)
	if err != nil {
		return nil, err
	}
	falseFn, err := f.validateClosure("If", "falseBranch", falseBranch)
	if err != nil {
		return nil, err
	}

	// Verify both branches have no parameters
	if len(trueFn.parameters) != 0 {
		return nil, errors.Errorf("If: trueBranch must have no parameters, got %d", len(trueFn.parameters))
	}
	if len(falseFn.parameters) != 0 {
		return nil, errors.Errorf("If: falseBranch must have no parameters, got %d", len(falseFn.parameters))
	}

	// Verify both branches have the same number of outputs with matching shapes
	if len(trueFn.outputs) != len(falseFn.outputs) {
		return nil, errors.Errorf("If: branches must return same number of outputs, trueBranch returns %d, falseBranch returns %d",
			len(trueFn.outputs), len(falseFn.outputs))
	}
	for i := range trueFn.outputs {
		if !trueFn.outputs[i].shape.Equal(falseFn.outputs[i].shape) {
			return nil, errors.Errorf("If: output %d shapes must match, trueBranch returns %s, falseBranch returns %s",
				i, trueFn.outputs[i].shape, falseFn.outputs[i].shape)
		}
	}

	// Create the If node - it will be executed at runtime
	outputShapes := make([]shapes.Shape, len(trueFn.outputs))
	for i, out := range trueFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	data := &ifNode{
		trueBranch:  trueFn,
		falseBranch: falseFn,
	}

	// Create multi-output node for If with only the predicate as regular input.
	// Captured values are tracked separately via AddNodeCapturedInputs.
	node := f.newMultiOutputsNode(backends.OpTypeIf, outputShapes, predNode)
	node.data = data

	// Add captured values from both branches to node.capturedInputs.
	// Each closure's captures are stored as a separate slice.
	node.AddNodeCapturedInputs(trueFn)
	node.AddNodeCapturedInputs(falseFn)

	return node.MultiOutputValues(), nil
}

// ifNode holds the data for an If operation.
type ifNode struct {
	trueBranch  *Function
	falseBranch *Function
}

// While executes a loop while a condition is true.
//
// The condition closure takes the current state values and returns a scalar boolean.
// The body closure takes the current state values and returns new state values.
// Both must have the same number of parameters matching the initialState count.
func (f *Function) While(cond, body backends.Function, initialState ...backends.Value) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	if len(initialState) == 0 {
		return nil, errors.Errorf("While: requires at least one initial state value")
	}

	// Validate initial state
	stateNodes, err := f.verifyAndCastValues("While", initialState...)
	if err != nil {
		return nil, err
	}

	// Validate closures and their parameters
	condFn, err := f.validateClosure("While", "cond", cond)
	if err != nil {
		return nil, err
	}
	if err = checkClosureParams("While", "cond", condFn, stateNodes); err != nil {
		return nil, err
	}

	bodyFn, err := f.validateClosure("While", "body", body)
	if err != nil {
		return nil, err
	}
	if err := checkClosureParams("While", "body", bodyFn, stateNodes); err != nil {
		return nil, err
	}

	// Verify cond returns a scalar boolean
	if len(condFn.outputs) != 1 {
		return nil, errors.Errorf("While: cond must return exactly one value, got %d", len(condFn.outputs))
	}
	if condFn.outputs[0].shape.Rank() != 0 || condFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("While: cond must return a scalar boolean, got %s", condFn.outputs[0].shape)
	}

	// Verify body returns same shapes as initialState
	if len(bodyFn.outputs) != len(stateNodes) {
		return nil, errors.Errorf("While: body must return %d values matching initialState, got %d",
			len(stateNodes), len(bodyFn.outputs))
	}
	for i, out := range bodyFn.outputs {
		if !out.shape.Equal(stateNodes[i].shape) {
			return nil, errors.Errorf("While: body output %d shape %s must match initialState shape %s",
				i, out.shape, stateNodes[i].shape)
		}
	}

	// Create output shapes (same as initial state)
	outputShapes := make([]shapes.Shape, len(stateNodes))
	for i, node := range stateNodes {
		outputShapes[i] = node.shape.Clone()
	}

	data := &whileNode{
		cond:       condFn,
		body:       bodyFn,
		stateCount: len(stateNodes),
	}

	// Create multi-output node for While with only state values as regular inputs.
	// Captured values are tracked separately via AddNodeCapturedInputs.
	node := f.newMultiOutputsNode(backends.OpTypeWhile, outputShapes, stateNodes...)
	node.data = data

	// Add captured values from both closures to node.capturedInputs.
	// Each closure's captures are stored as a separate slice.
	node.AddNodeCapturedInputs(condFn)
	node.AddNodeCapturedInputs(bodyFn)

	return node.MultiOutputValues(), nil
}

// whileNode holds the data for a While operation.
type whileNode struct {
	cond       *Function
	body       *Function
	stateCount int // Number of state values
}
