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
//   - Take 2*N scalar parameters (where N is the number of input tensors), in the order
//     (lhs_0, rhs_0, lhs_1, rhs_1, ..., lhs_N-1, rhs_N-1) where lhs_i and rhs_i are
//     scalar values from input tensor i being compared.
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
// For sorting two tensors together (e.g., values and indices):
//
//	comparator := NewClosure(g, func(g *graph.Graph) []*Node {
//	    lhsVal := Parameter(g, "lhs_val", shapes.Make(dtypes.Float32))
//	    rhsVal := Parameter(g, "rhs_val", shapes.Make(dtypes.Float32))
//	    lhsIdx := Parameter(g, "lhs_idx", shapes.Make(dtypes.Int32))
//	    rhsIdx := Parameter(g, "rhs_idx", shapes.Make(dtypes.Int32))
//	    _ = lhsIdx // unused but required
//	    _ = rhsIdx // unused but required
//	    return []*Node{LessThan(lhsVal, rhsVal)} // Compare by values only
//	})
//	sortedResults := SortFunc(comparator, 0, false, values, indices)
//
// Parameters:
//   - comparator: A closure that compares two sets of scalar values.
//   - axis: The axis along which to sort (can be negative to count from the end).
//   - isStable: If true, maintains the relative order of equal elements.
//   - inputs: One or more tensors to sort. All must have the same dimensions (dtypes can differ).
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

	// Verify all inputs have the same dimensions (dtype can differ)
	shape := inputs[0].Shape()
	for i, input := range inputs[1:] {
		if !input.Shape().EqualDimensions(shape) {
			exceptions.Panicf("SortFunc: all inputs must have the same dimensions, input 0 has %s, input %d has %s",
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

// TopK returns the top K largest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor.
//   - k: The number of top elements to retrieve.
//   - axis: The axis along which to find top K (can be negative to count from the end).
//
// Returns:
//   - values: The top K largest values, sorted in descending order.
//   - indices: The indices of the top K values in the original tensor (dtype Int32).
//
// The output shapes have the same rank as input, with the specified axis having size K.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	values, indices := TopK(x, 3, 0)
//	// values = [9, 6, 5]
//	// indices = [5, 7, 4]
func TopK(x *Node, k int, axis int) (values, indices *Node) {
	return topKImpl(x, k, axis, false)
}

// BottomK returns the K smallest elements and their indices along the specified axis.
//
// Parameters:
//   - x: The input tensor.
//   - k: The number of bottom elements to retrieve.
//   - axis: The axis along which to find bottom K (can be negative to count from the end).
//
// Returns:
//   - values: The K smallest values, sorted in ascending order.
//   - indices: The indices of the K smallest values in the original tensor (dtype Int32).
//
// The output shapes have the same rank as input, with the specified axis having size K.
//
// Example:
//
//	x := Const(g, []float32{3, 1, 4, 1, 5, 9, 2, 6})
//	values, indices := BottomK(x, 3, 0)
//	// values = [1, 1, 2]
//	// indices = [1, 3, 6]
func BottomK(x *Node, k int, axis int) (values, indices *Node) {
	return topKImpl(x, k, axis, true)
}

// topKImpl is the shared implementation for TopK and BottomK.
func topKImpl(x *Node, k int, axis int, ascending bool) (values, indices *Node) {
	g := x.graph
	g.AssertBuilding()

	shape := x.Shape()
	rank := shape.Rank()
	dtype := shape.DType

	// Normalize axis
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		exceptions.Panicf("TopK/BottomK: axis %d out of range for rank %d", axis, rank)
	}

	// Validate k
	axisSize := shape.Dimensions[axis]
	if k <= 0 {
		exceptions.Panicf("TopK/BottomK: k must be positive, got %d", k)
	}
	if k > axisSize {
		exceptions.Panicf("TopK/BottomK: k=%d exceeds axis size %d", k, axisSize)
	}

	// Create indices tensor using Iota along the sort axis
	// Indices will have Int32 dtype
	indicesShape := shape.Clone()
	indicesShape.DType = dtypes.Int32
	indicesInput := Iota(g, indicesShape, axis)

	// Create comparator that compares based on values
	// StableHLO sort expects parameters in order: (lhs_0, rhs_0, lhs_1, rhs_1, ...)
	// where input 0 is values and input 1 is indices
	comparator := NewClosure(g, func(g *Graph) []*Node {
		lhsVal := Parameter(g, "lhs_val", shapes.Make(dtype))
		rhsVal := Parameter(g, "rhs_val", shapes.Make(dtype))
		_ = Parameter(g, "lhs_idx", shapes.Make(dtypes.Int32))
		_ = Parameter(g, "rhs_idx", shapes.Make(dtypes.Int32))
		if ascending {
			return []*Node{LessThan(lhsVal, rhsVal)}
		}
		return []*Node{GreaterThan(lhsVal, rhsVal)}
	})

	// Sort values and indices together
	sortedResults := SortFunc(comparator, axis, true, x, indicesInput)
	sortedValues := sortedResults[0]
	sortedIndices := sortedResults[1]

	// Slice the first K elements along the sort axis
	sliceSpecs := make([]SliceAxisSpec, rank)
	for i := 0; i < rank; i++ {
		if i == axis {
			sliceSpecs[i] = AxisRange(0, k)
		} else {
			sliceSpecs[i] = AxisRange() // Full range
		}
	}

	values = Slice(sortedValues, sliceSpecs...)
	indices = Slice(sortedIndices, sliceSpecs...)

	return values, indices
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
