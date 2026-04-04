//go:build darwin && cgo

package metal

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// capturedNodeData is stored in OpTypeCapturedValue nodes (index into capturedParentNodes).
type capturedNodeData int

func (f *Function) getOrCreateCaptureNode(parentNode *Node) *Node {
	for i, captured := range f.capturedParentNodes {
		if captured == parentNode {
			return f.capturedLocalNodes[i]
		}
	}

	nodeToCapture := parentNode

	if f.parent == nil {
		panic(errors.Errorf(
			"getOrCreateCaptureNode: function %q has no parent but captures from %q",
			f.name, parentNode.function.name))
	}

	if parentNode.function != f.parent {
		parentCaptureNode := f.parent.getOrCreateCaptureNode(parentNode)
		nodeToCapture = parentCaptureNode
	}

	captureIdx := len(f.capturedParentNodes)
	captureNode := f.addNode(backends.OpTypeCapturedValue, parentNode.shape, nil, capturedNodeData(captureIdx))
	f.capturedParentNodes = append(f.capturedParentNodes, nodeToCapture)
	f.capturedLocalNodes = append(f.capturedLocalNodes, captureNode)

	return captureNode
}

// AddNodeCapturedInputs appends one closure's captured parent nodes for dependency tracking.
func (n *Node) AddNodeCapturedInputs(closure *Function) {
	if closure == nil {
		n.capturedInputs = append(n.capturedInputs, nil)
		return
	}

	capturedNodes := make([]*Node, len(closure.capturedParentNodes))
	copy(capturedNodes, closure.capturedParentNodes)
	n.capturedInputs = append(n.capturedInputs, capturedNodes)
}

func (f *Function) validateClosure(opName, closureName string, closure backends.Function) (*Function, error) {
	fn, ok := closure.(*Function)

	if !ok {
		return nil, errors.Errorf("%s: %s must be a *metal.Function, got %T", opName, closureName, closure)
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

func checkClosureParams(opName, closureName string, fn *Function, expected []*Node) error {
	if len(fn.parameters) != len(expected) {
		return errors.Errorf("%s: %s must have %d parameters, got %d",
			opName, closureName, len(expected), len(fn.parameters))
	}

	for i, param := range fn.parameters {
		if !param.shape.Equal(expected[i].shape) {
			return errors.Errorf("%s: %s parameter %d shape %s must match %s",
				opName, closureName, i, param.shape, expected[i].shape)
		}
	}

	return nil
}

type callNode struct {
	target *Function
}

type ifNode struct {
	trueBranch  *Function
	falseBranch *Function
}

type whileNode struct {
	cond       *Function
	body       *Function
	stateCount int
}

type sortNode struct {
	comparator *Function
	axis       int
	isStable   bool
	inputCount int
}

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

func (f *Function) Call(target backends.Function, inputs ...backends.Value) ([]backends.Value, error) {
	inputNodes, err := f.verifyAndCastValues("Call", inputs...)

	if err != nil {
		return nil, err
	}

	targetFn, ok := target.(*Function)

	if !ok {
		return nil, errors.Errorf("Call: target must be *metal.Function, got %T", target)
	}

	if targetFn.builder != f.builder {
		return nil, errors.New("Call: target must be from the same builder")
	}

	if targetFn.parent != f {
		return nil, errors.Errorf("Call: target %q must be a direct closure of the current function", targetFn.name)
	}

	if !targetFn.returned {
		return nil, errors.Errorf("Call: target %q must have Return()", targetFn.name)
	}

	if targetFn.compiled == nil {
		return nil, errors.Errorf("Call: target %q must be compiled", targetFn.name)
	}

	if len(inputNodes) != len(targetFn.parameters) {
		return nil, errors.Errorf("Call: %q expects %d parameters, got %d",
			targetFn.name, len(targetFn.parameters), len(inputNodes))
	}

	for i, param := range targetFn.parameters {
		if !param.shape.Equal(inputNodes[i].shape) {
			return nil, errors.Errorf("Call: %q param %d shape %s vs input %s",
				targetFn.name, i, param.shape, inputNodes[i].shape)
		}
	}

	outputShapes := make([]shapes.Shape, len(targetFn.outputs))

	for i, out := range targetFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	node := f.newMultiOutputsNode(backends.OpTypeCall, outputShapes, nil, inputNodes...)
	node.data = &callNode{target: targetFn}

	if len(targetFn.capturedParentNodes) > 0 {
		node.AddNodeCapturedInputs(targetFn)
	}

	return node.MultiOutputValues(), nil
}

func (f *Function) If(pred backends.Value, trueBranch, falseBranch backends.Function) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	predNodes, err := f.verifyAndCastValues("If", pred)

	if err != nil {
		return nil, err
	}

	predNode := predNodes[0]

	if predNode.shape.Rank() != 0 || predNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("If: pred must be scalar bool, got %s", predNode.shape)
	}

	trueFn, err := f.validateClosure("If", "trueBranch", trueBranch)

	if err != nil {
		return nil, err
	}

	falseFn, err := f.validateClosure("If", "falseBranch", falseBranch)

	if err != nil {
		return nil, err
	}

	if len(trueFn.parameters) != 0 || len(falseFn.parameters) != 0 {
		return nil, errors.New("If: branches must have no parameters")
	}

	if len(trueFn.outputs) != len(falseFn.outputs) {
		return nil, errors.Errorf("If: branch output count mismatch %d vs %d",
			len(trueFn.outputs), len(falseFn.outputs))
	}

	for i := range trueFn.outputs {
		if !trueFn.outputs[i].shape.Equal(falseFn.outputs[i].shape) {
			return nil, errors.Errorf("If: output %d shape mismatch", i)
		}
	}

	outputShapes := make([]shapes.Shape, len(trueFn.outputs))

	for i, out := range trueFn.outputs {
		outputShapes[i] = out.shape.Clone()
	}

	node := f.newMultiOutputsNode(backends.OpTypeIf, outputShapes, nil, predNode)
	node.data = &ifNode{trueBranch: trueFn, falseBranch: falseFn}
	node.AddNodeCapturedInputs(trueFn)
	node.AddNodeCapturedInputs(falseFn)

	return node.MultiOutputValues(), nil
}

func (f *Function) While(cond, body backends.Function, initialState ...backends.Value) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	if len(initialState) == 0 {
		return nil, errors.New("While: need at least one state value")
	}

	stateNodes, err := f.verifyAndCastValues("While", initialState...)

	if err != nil {
		return nil, err
	}

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

	if len(condFn.outputs) != 1 {
		return nil, errors.Errorf("While: cond must return 1 value, got %d", len(condFn.outputs))
	}

	if condFn.outputs[0].shape.Rank() != 0 || condFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("While: cond must return scalar bool, got %s", condFn.outputs[0].shape)
	}

	if len(bodyFn.outputs) != len(stateNodes) {
		return nil, errors.Errorf("While: body returns %d, need %d", len(bodyFn.outputs), len(stateNodes))
	}

	for i, out := range bodyFn.outputs {
		if !out.shape.Equal(stateNodes[i].shape) {
			return nil, errors.Errorf("While: body output %d shape mismatch", i)
		}
	}

	outputShapes := make([]shapes.Shape, len(stateNodes))

	for i, n := range stateNodes {
		outputShapes[i] = n.shape.Clone()
	}

	node := f.newMultiOutputsNode(backends.OpTypeWhile, outputShapes, nil, stateNodes...)
	node.data = &whileNode{cond: condFn, body: bodyFn, stateCount: len(stateNodes)}
	node.AddNodeCapturedInputs(condFn)
	node.AddNodeCapturedInputs(bodyFn)

	return node.MultiOutputValues(), nil
}

func (f *Function) Sort(comparator backends.Function, axis int, isStable bool, inputs ...backends.Value) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	if len(inputs) == 0 {
		return nil, errors.New("Sort: need at least one input")
	}

	inputNodes, err := f.verifyAndCastValues("Sort", inputs...)

	if err != nil {
		return nil, err
	}

	compFn, err := f.validateClosure("Sort", "comparator", comparator)

	if err != nil {
		return nil, err
	}

	firstShape := inputNodes[0].shape

	for i, n := range inputNodes[1:] {
		if !shapesEqualDimensions(firstShape, n.shape) {
			return nil, errors.Errorf("Sort: input %d dimensions differ from input 0", i+1)
		}
	}

	rank := firstShape.Rank()

	if axis < 0 {
		axis = rank + axis
	}

	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("Sort: axis %d out of range for rank %d", axis, rank)
	}

	expectedParams := 2 * len(inputNodes)

	if len(compFn.parameters) != expectedParams {
		return nil, errors.Errorf("Sort: comparator needs %d scalar params, got %d",
			expectedParams, len(compFn.parameters))
	}

	for i, inNode := range inputNodes {
		dt := inNode.shape.DType

		for j, side := range []string{"lhs", "rhs"} {
			pi := 2*i + j
			p := compFn.parameters[pi]

			if p.shape.Rank() != 0 || p.shape.DType != dt {
				return nil, errors.Errorf("Sort: param %d (%s_%d) must be scalar %s, got %s",
					pi, side, i, dt, p.shape)
			}
		}
	}

	if len(compFn.outputs) != 1 || compFn.outputs[0].shape.Rank() != 0 ||
		compFn.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("Sort: comparator must return scalar bool, got %v", compFn.outputs)
	}

	outputShapes := make([]shapes.Shape, len(inputNodes))

	for i, n := range inputNodes {
		outputShapes[i] = n.shape.Clone()
	}

	node := f.newMultiOutputsNode(backends.OpTypeSort, outputShapes, nil, inputNodes...)
	node.data = &sortNode{
		comparator: compFn,
		axis:       axis,
		isStable:   isStable,
		inputCount: len(inputNodes),
	}

	node.AddNodeCapturedInputs(compFn)
	return node.MultiOutputValues(), nil
}
