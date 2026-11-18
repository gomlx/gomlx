// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// MainName is the name of the main function created with each graph.
const MainName = compute.MainName

// FunctionImpl is the internal representation of a function or closure in a graph.
type FunctionImpl struct {
	public any // Back-link to canonical *graph.Function wrapper.

	graph       *GraphImpl
	backendFunc compute.Function

	name     string
	parent   *FunctionImpl
	returned bool

	inputs, outputs []shapes.Shape

	parameters            []*NodeImpl
	parametersNames       []string
	parameterNameToHandle map[string]ParameterHandle

	scalars         scalarCache
	tensorConstants *tensorConstCache
}

// Public returns the canonical *graph.Function wrapper (stored as any), or nil.
func (f *FunctionImpl) Public() any {
	if f == nil {
		return nil
	}
	return f.public
}

// SetPublic records the canonical *graph.Function wrapper on f.
func (f *FunctionImpl) SetPublic(pub any) {
	f.public = pub
}

// BackendFunc returns the underlying compute.Function.
func (f *FunctionImpl) BackendFunc() compute.Function {
	return f.backendFunc
}

func (g *GraphImpl) newMainFunc() *FunctionImpl {
	f := &FunctionImpl{
		graph:                 g,
		backendFunc:           g.builder.Main(),
		name:                  MainName,
		parent:                nil,
		returned:              false,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
		tensorConstants:       newTensorConstCache(),
	}
	g.functions[MainName] = f
	return f
}

// Name returns the name of the function ("" for closures).
func (f *FunctionImpl) Name() string {
	return f.name
}

// Path returns the slash-separated scope path from the root function.
func (f *FunctionImpl) Path() string {
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

// Parent returns the parent function of a closure, or nil.
func (f *FunctionImpl) Parent() *FunctionImpl {
	return f.parent
}

// IsMain returns whether this is the main function.
func (f *FunctionImpl) IsMain() bool {
	return f.name == MainName
}

// IsAncestorOf checks whether f is an ancestor of leafFunc (including f == leafFunc).
func (f *FunctionImpl) IsAncestorOf(leafFunc *FunctionImpl) bool {
	for ; leafFunc != nil; leafFunc = leafFunc.parent {
		if leafFunc == f {
			return true
		}
	}
	return false
}

// IsClosure returns whether this is a closure.
func (f *FunctionImpl) IsClosure() bool {
	return f.name == "" && f.parent != nil
}

// Return sets the return values of this function.
func (f *FunctionImpl) Return(outputs []*NodeImpl, outputShardings []*distributed.ShardingSpec) {
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

	outputsSet := sets.Make[*NodeImpl]()
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

	outputsOps := xslices.Map(outputs, func(node *NodeImpl) compute.Value { return node.outputOps[0] })
	backendShardings := xslices.Map(outputShardings, func(s *distributed.ShardingSpec) *compute.ShardingSpec {
		return s.ToBackendsSpec()
	})

	if err := g.currentFunc.backendFunc.Return(outputsOps, backendShardings); err != nil {
		panic(errors.WithMessagef(err, "Graph failed to set return values"))
	}
	f.returned = true
	f.outputs = xslices.Map(outputs, func(n *NodeImpl) shapes.Shape { return n.Shape() })
}

func innermostFunction(inputs []*NodeImpl) (*FunctionImpl, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	var candidate *FunctionImpl
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
			candidate = other
		} else if !other.IsAncestorOf(candidate) {
			return nil, errors.Errorf("incompatible scopes for inputs: scope %q and scope %q are not in the same ancestry line", candidate.name, other.name)
		}
	}
	return candidate, nil
}

// NewClosure creates a new closure in g.
func NewClosure(g *GraphImpl, funcDef func(g *GraphImpl) []*NodeImpl) *FunctionImpl {
	g.AssertBuilding()
	return NewClosureWithSharding(g, func(g *GraphImpl) ([]*NodeImpl, []*distributed.ShardingSpec) {
		return funcDef(g), nil
	})
}

// NewClosureWithSharding creates a new closure in g with optional output shardings.
func NewClosureWithSharding(g *GraphImpl, funcDef func(g *GraphImpl) ([]*NodeImpl, []*distributed.ShardingSpec)) *FunctionImpl {
	g.AssertBuilding()
	if !g.backend.Capabilities().Functions {
		panic(errors.Wrapf(compute.ErrNotImplemented, "backend %q does not support functions (needed for closures)", g.backend.Name()))
	}

	f := &FunctionImpl{
		graph:                 g,
		name:                  "",
		parent:                g.currentFunc,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
		tensorConstants:       newTensorConstCache(),
	}
	var err error
	f.backendFunc, err = g.builder.Main().Closure()
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create new closure"))
	}

	prevFunc := g.currentFunc
	g.currentFunc = f
	defer func() {
		g.currentFunc = prevFunc
	}()

	outputs, shardings := funcDef(g)

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

	f.Return(outputs, shardings)
	return f
}

// NewFunction creates a new top-level function in g.
func NewFunction(g *GraphImpl, name string, funcDef func(g *GraphImpl) []*NodeImpl) *FunctionImpl {
	g.AssertBuilding()
	return NewFunctionWithSharding(g, name, func(g *GraphImpl) ([]*NodeImpl, []*distributed.ShardingSpec) {
		return funcDef(g), nil
	})
}

// NewFunctionWithSharding creates a new top-level function in g with optional output shardings.
func NewFunctionWithSharding(g *GraphImpl, name string, funcDef func(g *GraphImpl) ([]*NodeImpl, []*distributed.ShardingSpec)) *FunctionImpl {
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

	f := &FunctionImpl{
		graph:                 g,
		name:                  name,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
		tensorConstants:       newTensorConstCache(),
	}
	var err error
	f.backendFunc, err = g.builder.NewFunction(name)
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create new function %q", name))
	}

	prevFunc := g.currentFunc
	g.currentFunc = f
	defer func() {
		g.currentFunc = prevFunc
	}()

	outputs, shardings := funcDef(g)

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
	f.Return(outputs, shardings)
	return f
}

// NodeTypeCall is the NodeType for function calls.
const NodeTypeCall NodeType = 1000

// NodeInputsCall holds the inputs for compute.Function.Call.
type NodeInputsCall struct {
	F      *FunctionImpl
	Inputs []*NodeImpl
}

func (ni *NodeInputsCall) Type() NodeType { return NodeTypeCall }

func (ni *NodeInputsCall) String() string {
	return fmt.Sprintf("Call(%s, %d inputs)", ni.F.name, len(ni.Inputs))
}

func (ni *NodeInputsCall) CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	g := originalNode.Graph()
	inputOps := make([]compute.Value, len(newInputs))
	for i, n := range newInputs {
		inputOps[i] = n.outputOps[0]
	}
	results, err := g.currentFunc.backendFunc.Call(ni.F.backendFunc, inputOps...)
	if err != nil {
		panic(errors.WithMessagef(err, "Failed to call function %q", ni.F.name))
	}
	outputShapes := make([]shapes.Shape, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
	}
	newNi := &NodeInputsCall{
		F:      ni.F,
		Inputs: newInputs,
	}
	node := &NodeImpl{
		graph:        g,
		outputOps:    results,
		outputShapes: outputShapes,
		inputs:       newNi,
		inputNodes:   newInputs,
	}
	g.registerNode(node)
	return node
}

// Call invokes function f with the given inputs.
func (f *FunctionImpl) Call(inputs ...*NodeImpl) []*NodeImpl {
	g := f.graph
	g.AssertBuilding()
	if f.IsClosure() {
		exceptions.Panicf("closure functions (%q) cannot be called directly, only used as "+
			"inputs to other functions", f.Path())
	}

	if len(inputs) != len(f.parameters) {
		exceptions.Panicf("Function %q expects %d inputs, but got %d", f.name, len(f.parameters), len(inputs))
	}

	ValidateBuildingGraphFromInputs(inputs...)

	for i, input := range inputs {
		param := f.parameters[i]
		if input.Rank() != param.Rank() {
			exceptions.Panicf("Function %q input #%d expects rank %d (param shape %s), got rank %d (input shape %s)",
				f.name, i, param.Rank(), param.Shape(), input.Rank(), input.Shape())
		}
		if input.DType() != param.DType() {
			exceptions.Panicf("Function %q input #%d expects dtype %s, got %s",
				f.name, i, param.DType(), input.DType())
		}
	}

	ni := &NodeInputsCall{
		F:      f,
		Inputs: inputs,
	}

	inputOps := make([]compute.Value, len(inputs))
	for i, n := range inputs {
		inputOps[i] = n.outputOps[0]
	}

	results, err := g.currentFunc.backendFunc.Call(f.backendFunc, inputOps...)
	if err != nil {
		panic(errors.WithMessagef(err, "Failed to call function %q", f.name))
	}

	outputShapes := make([]shapes.Shape, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
	}

	node := &NodeImpl{
		graph:        g,
		outputOps:    results,
		outputShapes: outputShapes,
		inputs:       ni,
		inputNodes:   inputs,
	}
	g.registerNode(node)

	return SplitNode(node)
}
