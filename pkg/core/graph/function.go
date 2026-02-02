// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// MainName is the name of the main function -- the default function created when the Graph is created.
const MainName = backends.MainName

// Function represents a function in the graph.
//
// The Graph always is associated with a "current" function, where it is operating.
// It defaults to the main function (named "main") automatically created when the graph is created.
//
// But if the backend supports it, other top-level functions or closures (also a type of function) can be created.
// See NewClosure and NewFunction functions.
type Function struct {
	graph       *Graph
	backendFunc backends.Function

	name     string
	parent   *Function
	returned bool

	inputs, outputs []shapes.Shape

	// parameters keeps track of parameter nodes its names and a mapping of name to index.
	parameters            []*Node
	parametersNames       []string
	parameterNameToHandle map[string]ParameterHandle
}

// newMainFunc is called during the creation of a Graph.
func (g *Graph) newMainFunc() *Function {
	f := &Function{
		graph:                 g,
		backendFunc:           g.builder.Main(),
		name:                  MainName,
		parent:                nil,
		returned:              false,
		parameterNameToHandle: make(map[string]ParameterHandle),
	}
	g.functions[MainName] = f
	return f
}

// Name returns the name of the function.
//
// For top-level functions the name is unique.
// For closures the name is "".
func (f *Function) Name() string {
	return f.name
}

// Path returns the full path of the function, following its parents to the root.
// Empty names (closures) are replaced by "closure".
func (f *Function) Path() string {
	parts := []string{}
	for fn := f; fn != nil; fn = fn.parent {
		name := fn.name
		if name == "" {
			name = "closure"
		}
		parts = append(parts, name)
	}
	slices.Reverse(parts)
	return strings.Join(parts, "/")
}

// Parent return the parent function of a closure.
// It returns nil if this is a top-level function.
func (f *Function) Parent() *Function {
	return f.parent
}

// IsMain returns whether this is the main function.
func (f *Function) IsMain() bool {
	return f.name == MainName
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

// IsClosure returns whether this is a closure.
func (f *Function) IsClosure() bool {
	return f.name == "" && f.parent != nil
}

// Return from the given function. This is the last operation of a function.
//
// Don't call this directly, this is mostly for internal use and testing:
// Compile, NewFunction and NewClosure will call this automatically.
func (f *Function) Return(outputs []*Node, outputShardings []*distributed.ShardingSpec) {
	g := f.graph
	g.AssertValid()
	g.AssertBuilding()
	if len(outputs) == 0 {
		exceptions.Panicf("no outputs selected when Graph.Compile graph %q", g.name)
	}
	if len(outputShardings) > 0 && len(outputShardings) != len(outputs) {
		exceptions.Panicf("if outputShardings are given, there must be one for each output, "+
			"but got %d outputs and %d shardings", len(outputs), len(outputShardings))
	}

	// Sanity check on the output nodes.
	for ii, node := range outputs {
		if node == nil {
			exceptions.Panicf("output node %d is nil when returning in graph %q", ii, g.name)
		}
		if node.Graph() != g {
			exceptions.Panicf("output node %d is part of a different graph (name=%q) than the one being "+
				"returned (name=%q)", ii, node.graph.name, g.name)
		}
		if !node.scope.IsAncestorOf(f) {
			exceptions.Panicf("output node #%d is not part of the function %q scope when returning in graph %q",
				ii, f.name, g.name)
		}
		if node.NumOutputs() != 1 {
			exceptions.Panicf("Graph(%q).Compile cannot take multi-output nodes (output #%d: %s), this type of Node"+
				" is internal only", g.name, ii, node)
		}
	}

	// Create "identities" for duplicate outputs and create a mapping if there are any:
	outputsSet := sets.Make[*Node]()
	for ii, node := range outputs {
		if outputsSet.Has(node) {
			outputs[ii] = Identity(node)
		} else {
			outputsSet.Insert(node)
		}
	}

	if klog.V(1).Enabled() {
		start := time.Now()
		defer func() {
			elapsed := time.Since(start)
			klog.Infof("Graph.Compile time for graph %q: %s", g.Name(), elapsed)
		}()
	}

	outputsOps := xslices.Map(outputs, func(node *Node) backends.Value { return node.outputOps[0] })
	backendShardings := xslices.Map(outputShardings, func(s *distributed.ShardingSpec) *backends.ShardingSpec {
		return s.ToBackendsSpec()
	})

	// Call Return on the main function with outputs and shardings
	if err := g.currentFunc.backendFunc.Return(outputsOps, backendShardings); err != nil {
		panic(errors.WithMessagef(err, "Graph failed to set return values"))
	}
	f.returned = true
	f.outputs = xslices.Map(outputs, func(n *Node) shapes.Shape { return n.Shape() })
}

// innermostFunction finds the "innermost" (deepest) function scope among the inputs.
//
// It returns an error if the scopes are incompatible (i.e., they are not on the same branch of the function tree).
//
// If inputs is empty, it returns nil -- the caller should handle this case (usually by assigning the current function).
func innermostFunction(inputs []*Node) (*Function, error) {
	if len(inputs) == 0 {
		return nil, nil // No inputs, no scope inferred.
	}

	var candidate *Function
	for i, node := range inputs {
		if node == nil {
			return nil, errors.Errorf("input node #%d is nil", i)
		}
		if node.scope == nil {
			return nil, errors.Errorf("input node #%d (%s) has a nil scope", i, node)
		}

		if candidate == nil {
			candidate = node.scope
			continue
		}

		other := node.scope
		if other == candidate {
			continue
		}

		if candidate.IsAncestorOf(other) {
			// candidate is ancestor of other, so other is deeper (or they are disjoint, checked later).
			// If candidate is ancestor of other, then they are compatible, and other is the new candidate.
			candidate = other
		} else if !other.IsAncestorOf(candidate) {
			// candidate is NOT ancestor of other, AND other is NOT ancestor of candidate.
			// Disjoint branches.
			return nil, errors.Errorf("incompatible scopes for inputs: scope %q and scope %q are not in the same ancestry line", candidate.name, other.name)
		}
		// else: other is ancestor of candidate, so candidate remains the deeper one.
	}
	return candidate, nil
}

// NewClosure creates a new closure in the current graph.
//
// Closures are used as parameters for special operations like While, If, etc;
// they cannot be called directly.
//
// Closures, unlike top-level functions (created with NewFunction), can access variables from the parent scope.
// Because of that they are not named and not registered in the Graph (so they cannot be retrieved by name).
//
// It creates the new Function object, sets it temporarily as the current for the Graph (reversed on exit)
// and call the funcDef with it.
//
// The signature of the created function will have the inputs based on the parameters created by funcDef, and
// the outputs based on the return values of funcDef.
func NewClosure(g *Graph, funcDef func(g *Graph) []*Node) *Function {
	g.AssertBuilding()
	return NewClosureWithSharding(g, func(g *Graph) ([]*Node, []*distributed.ShardingSpec) {
		return funcDef(g), nil
	})
}

// NewClosureWithSharding creates a new closure in the current graph.
//
// Closures are used as parameters for special operations like While, If, etc;
// they cannot be called directly.
//
// It is similar to NewClosure, but funcDef also returns the sharding information, and it is used on f.Return in the end.
//
// If the returned sharding information if not nil, it validates the sharding to the output nodes.
// Otherwise (sharding is nil), it behaves just like NewClosure.
func NewClosureWithSharding(g *Graph, funcDef func(g *Graph) ([]*Node, []*distributed.ShardingSpec)) *Function {
	g.AssertBuilding()
	if !g.backend.Capabilities().Functions {
		exceptions.Panicf("backend %q does not support functions (needed for closures)", g.backend.Name())
	}

	// Create the function.
	f := &Function{
		graph:                 g,
		name:                  "",
		parent:                g.currentFunc,
		parameterNameToHandle: make(map[string]ParameterHandle),
	}
	var err error
	f.backendFunc, err = g.builder.Main().Closure()
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create new closure"))
	}

	// Temporarily set the current function.
	prevFunc := g.currentFunc
	g.currentFunc = f
	defer func() {
		g.currentFunc = prevFunc
	}()

	// Call the definition.
	outputs, shardings := funcDef(g)

	// Validate shardings
	if len(shardings) > 0 {
		if len(shardings) != len(outputs) {
			exceptions.Panicf("NewClosureWithSharding: got %d outputs but %d sharding specs", len(outputs), len(shardings))
		}
		for i, node := range outputs {
			spec := shardings[i]
			if spec == nil {
				continue
			}
			shape := node.Shape()
			if spec.Rank() != shape.Rank() {
				exceptions.Panicf("NewClosureWithSharding: output #%d has shape %s (rank %d) but sharding spec has rank %d -- they must match",
					i, shape, shape.Rank(), spec.Rank())
			}
			shardShape := spec.ShardShape(shape)
			if !shardShape.Ok() {
				exceptions.Panicf("NewClosureWithSharding: output #%d shape %s is not compatible with sharding spec %s: shard shape is invalid",
					i, shape, spec)
			}
		}
	}

	// Return the outputs.
	f.Return(outputs, shardings)
	return f
}

// NewFunction creates a new top-level function in the current graph.
//
// It creates the new Function object, sets it temporarily as the current for the Graph (reversed on exit)
// and call the funcDef with it.
//
// The signature of the created function will have the inputs based on the parameters created by funcDef, and
// the outputs based on the return values of funcDef.
func NewFunction(g *Graph, name string, funcDef func(g *Graph) []*Node) *Function {
	g.AssertBuilding()
	return NewFunctionWithSharding(g, name, func(g *Graph) ([]*Node, []*distributed.ShardingSpec) {
		return funcDef(g), nil
	})
}

// NewFunctionWithSharding creates a new top-level function in the current graph.
//
// It is similar to NewFunction, but funcDef also returns the sharding information, and it is used on f.Return in the end.
//
// If the returned sharding information if not nil, it validates the sharding to the output nodes.
// Otherwise (sharding is nil), it behaves just like NewFunction.
func NewFunctionWithSharding(g *Graph, name string, funcDef func(g *Graph) ([]*Node, []*distributed.ShardingSpec)) *Function {
	g.AssertBuilding()
	if !g.backend.Capabilities().Functions {
		exceptions.Panicf("backend %q does not support other top-level functions (while attempting to create function %q)", g.backend.Name(), name)
	}
	if name == "" {
		exceptions.Panicf("function name cannot be empty")
	}
	if _, found := g.functions[name]; found {
		exceptions.Panicf("function name %q already exists in graph %q", name, g.name)
	}

	// Create the function.
	f := &Function{
		graph:                 g,
		name:                  name,
		parameterNameToHandle: make(map[string]ParameterHandle),
	}
	var err error
	f.backendFunc, err = g.builder.NewFunction(name)
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create new function %q", name))
	}

	// Temporarily set the current function.
	prevFunc := g.currentFunc
	g.currentFunc = f
	defer func() {
		g.currentFunc = prevFunc
	}()

	// Call the definition.
	outputs, shardings := funcDef(g)

	// Validate shardings
	if len(shardings) > 0 {
		if len(shardings) != len(outputs) {
			exceptions.Panicf("NewFunctionWithSharding: got %d outputs but %d sharding specs", len(outputs), len(shardings))
		}
		for i, node := range outputs {
			spec := shardings[i]
			if spec == nil {
				continue
			}
			shape := node.Shape()
			if spec.Rank() != shape.Rank() {
				exceptions.Panicf("NewFunctionWithSharding: output #%d has shape %s (rank %d) but sharding spec has rank %d -- they must match",
					i, shape, shape.Rank(), spec.Rank())
			}
			shardShape := spec.ShardShape(shape)
			if !shardShape.Ok() {
				exceptions.Panicf("NewFunctionWithSharding: output #%d shape %s is not compatible with sharding spec %s: shard shape is invalid",
					i, shape, spec)
			}
		}
	}

	g.functions[name] = f

	// Return the outputs.
	f.Return(outputs, shardings)
	return f
}

// NodeTypeCall is the NodeType for function calls.
// We use a large number to avoid conflict with generated NodeTypes in gen_backend_ops.go.
const NodeTypeCall NodeType = 1000

// nodeInputsCall holds the inputs used for the call to backends.Function.Call.
type nodeInputsCall struct {
	f      *Function
	inputs []*Node
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsCall) Type() NodeType {
	return NodeTypeCall
}

// String implements the interface NodeInputs.
func (ni *nodeInputsCall) String() string {
	return fmt.Sprintf("Call(%s, %d inputs)", ni.f.name, len(ni.inputs))
}

// Call a function with the given inputs.
//
// The function f must be from the same graph.
func (f *Function) Call(inputs ...*Node) []*Node {
	g := f.graph
	g.AssertBuilding()
	if f.IsClosure() {
		exceptions.Panicf("closure functions (%q) cannot be called directly, only used as "+
			"inputs to other functions", f.Path())
	}

	// Validate inputs
	if len(inputs) != len(f.parameters) {
		exceptions.Panicf("Function %q expects %d inputs, but got %d", f.name, len(f.parameters), len(inputs))
	}

	// Combine inputs for validation
	validateBuildingGraphFromInputs(inputs...) // Ensures inputs are from same graph g.

	for i, input := range inputs {
		param := f.parameters[i]
		// Check shape compatibility
		// For now, strict check on Rank and DType.
		if input.Rank() != param.Rank() {
			exceptions.Panicf("Function %q input #%d expects rank %d (param shape %s), got rank %d (input shape %s)",
				f.name, i, param.Rank(), param.Shape(), input.Rank(), input.Shape())
		}
		if input.DType() != param.DType() {
			exceptions.Panicf("Function %q input #%d expects dtype %s, got %s",
				f.name, i, param.DType(), input.DType())
		}
	}

	// Create node inputs
	ni := &nodeInputsCall{
		f:      f,
		inputs: inputs,
	}

	// Prepare backend inputs
	inputOps := make([]backends.Value, len(inputs))
	for i, n := range inputs {
		inputOps[i] = n.outputOps[0]
	}

	// Call backend
	results, err := g.currentFunc.backendFunc.Call(f.backendFunc, inputOps...)
	if err != nil {
		panic(errors.WithMessagef(err, "Failed to call function %q", f.name))
	}

	// Create output node
	outputShapes := make([]shapes.Shape, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
	}

	node := &Node{
		graph:        g,
		outputOps:    results,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   inputs,
	}
	g.registerNode(node)

	return splitNode(node)
}
