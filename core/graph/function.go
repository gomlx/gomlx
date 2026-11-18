// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// MainName is the name of the main function -- the default function created when the Graph is created.
const MainName = graphimpl.MainName

// Function represents a function in the graph.
//
// The Graph always is associated with a "current" function, where it is operating.
// It defaults to the main function (named "main") automatically created when the graph is created.
//
// But if the backend supports it, other top-level functions or closures (also a type of function) can be created.
// See NewClosure and NewFunction functions.
type Function struct {
	impl *graphimpl.FunctionImpl
}

func wrapFunction(impl *graphimpl.FunctionImpl) *Function {
	if impl == nil {
		return nil
	}
	if pub := impl.Public(); pub != nil {
		return pub.(*Function)
	}
	f := &Function{impl: impl}
	impl.SetPublic(f)
	return f
}

func unwrapFunction(f *Function) *graphimpl.FunctionImpl {
	if f == nil {
		return nil
	}
	return f.impl
}

// Name returns the name of the function ("" for closures).
func (f *Function) Name() string {
	return f.impl.Name()
}

// Path returns the full path of the function, following its parents to the root.
func (f *Function) Path() string {
	return f.impl.Path()
}

// Parent returns the parent function of a closure, or nil if this is a top-level function.
func (f *Function) Parent() *Function {
	return wrapFunction(f.impl.Parent())
}

// IsMain returns whether this is the main function.
func (f *Function) IsMain() bool {
	return f.impl.IsMain()
}

// IsAncestorOf checks whether f is an ancestor of leafFunc.
func (f *Function) IsAncestorOf(leafFunc *Function) bool {
	return f.impl.IsAncestorOf(unwrapFunction(leafFunc))
}

// IsClosure returns whether this is a closure.
func (f *Function) IsClosure() bool {
	return f.impl.IsClosure()
}

// NewClosure creates a new closure in the current graph.
func NewClosure(g *Graph, funcDef func(g *Graph) []*Node) *Function {
	return wrapFunction(graphimpl.NewClosure(unwrapGraph(g), func(ig *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
		return unwrapNodes(funcDef(wrapGraph(ig)))
	}))
}

// NewClosureWithSharding creates a new closure in the current graph with sharding outputs.
func NewClosureWithSharding(g *Graph, funcDef func(g *Graph) ([]*Node, []*distributed.ShardingSpec)) *Function {
	return wrapFunction(graphimpl.NewClosureWithSharding(unwrapGraph(g), func(ig *graphimpl.GraphImpl) ([]*graphimpl.NodeImpl, []*distributed.ShardingSpec) {
		nodes, shardings := funcDef(wrapGraph(ig))
		return unwrapNodes(nodes), shardings
	}))
}

// NewFunction creates a new top-level function in the current graph.
func NewFunction(g *Graph, name string, funcDef func(g *Graph) []*Node) *Function {
	return wrapFunction(graphimpl.NewFunction(unwrapGraph(g), name, func(ig *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
		return unwrapNodes(funcDef(wrapGraph(ig)))
	}))
}

// NewFunctionWithSharding creates a new top-level function in the current graph with sharding outputs.
func NewFunctionWithSharding(g *Graph, name string, funcDef func(g *Graph) ([]*Node, []*distributed.ShardingSpec)) *Function {
	return wrapFunction(graphimpl.NewFunctionWithSharding(unwrapGraph(g), name, func(ig *graphimpl.GraphImpl) ([]*graphimpl.NodeImpl, []*distributed.ShardingSpec) {
		nodes, shardings := funcDef(wrapGraph(ig))
		return unwrapNodes(nodes), shardings
	}))
}

// Call a function with the given inputs.
func (f *Function) Call(inputs ...*Node) []*Node {
	return wrapNodes(f.impl.Call(unwrapNodes(inputs)...))
}
