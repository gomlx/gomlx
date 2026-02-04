// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/pkg/errors"
)

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
