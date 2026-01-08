// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// NodeTypeSort is the NodeType for Sort operations.
const NodeTypeSort NodeType = 1001

// nodeInputsSort holds the inputs for a Sort operation.
type nodeInputsSort struct {
	comparator *Function
	axis       int
	isStable   bool
	inputs     []*Node
}

// Type implements NodeInputs.
func (ni *nodeInputsSort) Type() NodeType { return NodeTypeSort }

// String implements NodeInputs.
func (ni *nodeInputsSort) String() string {
	return "Sort"
}

// Sort sorts a single tensor along the specified axis.
//
// This is a simplified version of SortFunc that doesn't require a comparator closure.
// For sorting multiple tensors together or using a custom comparator, use SortFunc.
//
// Parameters:
//   - input: The tensor to sort.
//   - axis: The axis along which to sort (can be negative to count from the end).
//   - ascending: If true, sorts in ascending order; otherwise descending.
//
// Returns the sorted tensor.
//
// Example:
//
//	x := Const(g, []float32{5, 2, 8, 1, 9, 3})
//	sorted := Sort(x, 0, true)  // Returns [1, 2, 3, 5, 8, 9]
func Sort(input *Node, axis int, ascending bool) *Node {
	g := input.graph
	g.AssertBuilding()
	dtype := input.Shape().DType

	// Create a simple comparator closure
	comparator := NewClosure(g, func(g *Graph) []*Node {
		lhs := Parameter(g, "lhs", shapes.Make(dtype))
		rhs := Parameter(g, "rhs", shapes.Make(dtype))
		if ascending {
			return []*Node{LessThan(lhs, rhs)}
		}
		return []*Node{GreaterThan(lhs, rhs)}
	})

	results := SortFunc(comparator, axis, false, input)
	return results[0]
}

// SortFunc sorts one or more tensors along the specified axis using a comparator closure.
//
// The comparator closure should be created with NewClosure and must:
//   - Take 2*N scalar parameters (where N is the number of input tensors), representing
//     the left-hand side values (lhs_0, ..., lhs_N-1) and right-hand side values
//     (rhs_0, ..., rhs_N-1) being compared.
//   - Return a single boolean scalar: true if lhs should come before rhs.
//
// For a standard ascending sort on a single tensor with dtype Float32:
//
//	comparator := NewClosure(g, func(g *graph.Graph) []*Node {
//	    lhs := Parameter(g, "lhs", shapes.Make(dtypes.Float32))
//	    rhs := Parameter(g, "rhs", shapes.Make(dtypes.Float32))
//	    return []*Node{LessThan(lhs, rhs)}
//	})
//	sorted := SortFunc(comparator, 0, false, x)
//
// Parameters:
//   - comparator: A closure that compares two sets of scalar values.
//   - axis: The axis along which to sort (can be negative to count from the end).
//   - isStable: If true, maintains the relative order of equal elements.
//   - inputs: One or more tensors to sort. All must have the same shape.
//
// Returns the sorted tensors in the same order as inputs.
func SortFunc(comparator *Function, axis int, isStable bool, inputs ...*Node) []*Node {
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
	validateBuildingGraphFromInputs(inputs...)

	// Verify all inputs have the same shape
	shape := inputs[0].Shape()
	for i, input := range inputs[1:] {
		if !input.Shape().Equal(shape) {
			exceptions.Panicf("SortFunc: all inputs must have the same shape, input 0 has %s, input %d has %s",
				shape, i+1, input.Shape())
		}
	}

	// Normalize axis
	rank := shape.Rank()
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		exceptions.Panicf("SortFunc: axis %d out of range for rank %d", axis, rank)
	}

	// Convert inputs to backend values
	inputValues := make([]backends.Value, len(inputs))
	for i, input := range inputs {
		inputValues[i] = input.outputOps[0]
	}

	// Call backend Sort
	results, err := g.currentFunc.backendFunc.Sort(comparator.backendFunc, axis, isStable, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "SortFunc operation failed"))
	}

	// Create output nodes
	ni := &nodeInputsSort{
		comparator: comparator,
		axis:       axis,
		isStable:   isStable,
		inputs:     inputs,
	}

	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]backends.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}

	node := &Node{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   inputs,
		scope:        g.currentFunc,
	}
	g.registerNode(node)

	return splitNode(node)
}

// NodeTypeWhile is the NodeType for While operations.
const NodeTypeWhile NodeType = 1002

// nodeInputsWhile holds the inputs for a While operation.
type nodeInputsWhile struct {
	cond         *Function
	body         *Function
	initialState []*Node
}

// Type implements NodeInputs.
func (ni *nodeInputsWhile) Type() NodeType { return NodeTypeWhile }

// String implements NodeInputs.
func (ni *nodeInputsWhile) String() string {
	return "While"
}

// While executes a loop while a condition is true.
//
// The condition and body closures should be created with NewClosure.
//
// The condition closure must:
//   - Take N parameters (matching the number and shapes of initialState)
//   - Return a single boolean scalar
//
// The body closure must:
//   - Take N parameters (matching the number and shapes of initialState)
//   - Return N values with shapes matching initialState
//
// Example for summing numbers 1 to 10:
//
//	// State: [counter, sum]
//	cond := NewClosure(g, func(g *graph.Graph) []*Node {
//	    counter := Parameter(g, "counter", shapes.Make(dtypes.Int32))
//	    _ = Parameter(g, "sum", shapes.Make(dtypes.Int32))
//	    return []*Node{LessThan(counter, Const(g, int32(11)))}
//	})
//	body := NewClosure(g, func(g *graph.Graph) []*Node {
//	    counter := Parameter(g, "counter", shapes.Make(dtypes.Int32))
//	    sum := Parameter(g, "sum", shapes.Make(dtypes.Int32))
//	    return []*Node{
//	        Add(counter, Const(g, int32(1))),
//	        Add(sum, counter),
//	    }
//	})
//	results := While(cond, body, Const(g, int32(1)), Const(g, int32(0)))
//	// results[0] = 11, results[1] = 55
//
// Parameters:
//   - cond: Condition closure returning boolean scalar.
//   - body: Body closure returning new state.
//   - initialState: Initial values for the loop state.
//
// Returns the final state values.
func While(cond, body *Function, initialState ...*Node) []*Node {
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
	validateBuildingGraphFromInputs(initialState...)

	// Convert inputs to backend values
	inputValues := make([]backends.Value, len(initialState))
	for i, input := range initialState {
		inputValues[i] = input.outputOps[0]
	}

	// Call backend While
	results, err := g.currentFunc.backendFunc.While(cond.backendFunc, body.backendFunc, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "While operation failed"))
	}

	// Create output nodes
	ni := &nodeInputsWhile{
		cond:         cond,
		body:         body,
		initialState: initialState,
	}

	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]backends.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}

	node := &Node{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   initialState,
		scope:        g.currentFunc,
	}
	g.registerNode(node)

	return splitNode(node)
}

// NodeTypeIf is the NodeType for If operations.
const NodeTypeIf NodeType = 1003

// nodeInputsIf holds the inputs for an If operation.
type nodeInputsIf struct {
	pred        *Node
	trueBranch  *Function
	falseBranch *Function
}

// Type implements NodeInputs.
func (ni *nodeInputsIf) Type() NodeType { return NodeTypeIf }

// String implements NodeInputs.
func (ni *nodeInputsIf) String() string {
	return "If"
}

// If executes one of two branches based on a boolean predicate.
//
// The branch closures should be created with NewClosure and can capture values
// from the parent scope.
//
// Both branches must:
//   - Take no parameters
//   - Return the same number of values with matching shapes
//
// Example:
//
//	pred := IsPositive(x)
//	trueBranch := NewClosure(g, func(g *graph.Graph) []*Node {
//	    return []*Node{x}  // Return x if x > 0
//	})
//	falseBranch := NewClosure(g, func(g *graph.Graph) []*Node {
//	    return []*Node{Neg(x)}  // Return -x if x <= 0
//	})
//	result := If(pred, trueBranch, falseBranch)
//
// Parameters:
//   - pred: A scalar boolean value determining which branch to execute.
//   - trueBranch: Closure executed when pred is true.
//   - falseBranch: Closure executed when pred is false.
//
// Returns the outputs of the executed branch.
func If(pred *Node, trueBranch, falseBranch *Function) []*Node {
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

	// Verify pred is a scalar boolean
	predShape := pred.Shape()
	if !predShape.IsScalar() || predShape.DType != dtypes.Bool {
		exceptions.Panicf("If: pred must be a scalar boolean, got %s", predShape)
	}

	// Call backend If
	results, err := g.currentFunc.backendFunc.If(pred.outputOps[0], trueBranch.backendFunc, falseBranch.backendFunc)
	if err != nil {
		panic(errors.WithMessagef(err, "If operation failed"))
	}

	// Create output nodes
	ni := &nodeInputsIf{
		pred:        pred,
		trueBranch:  trueBranch,
		falseBranch: falseBranch,
	}

	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]backends.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}

	node := &Node{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   []*Node{pred},
		scope:        g.currentFunc,
	}
	g.registerNode(node)

	return splitNode(node)
}
