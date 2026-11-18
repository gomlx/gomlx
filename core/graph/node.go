// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

const MaxSizeToPrint = graphimpl.MaxSizeToPrint

// VJP is a vector-jacobian product (reverse-mode automatic differentiation) function for a Node.
type VJP func(node *Node, vjpOutputs []*Node, outputShape shapes.Shape) []*Node

// Node represents the result of an operation in the computation graph, and can be used as input to further operations.
//
// Internally, it keeps tracks of all parameters used for the computation: this is later used for auto-differentiation.
//
// It also stores meta-information: see Node.SetLogged, Node.StopGradient.
//
// Notice some complex methods offered in this package may be implemented with several instances of simpler operations
// and yield several nodes in the graph, that's normal.
//
// Node.String allows for a pretty-printing of node. To see the full graph with all nodes, use Graph.String.
type Node struct {
	impl *graphimpl.NodeImpl
}

func wrapNode(impl *graphimpl.NodeImpl) *Node {
	if impl == nil {
		return nil
	}
	if pub := impl.Public(); pub != nil {
		return pub.(*Node)
	}
	n := &Node{impl: impl}
	impl.SetPublic(n)
	return n
}

func unwrapNode(n *Node) *graphimpl.NodeImpl {
	if n == nil {
		return nil
	}
	return n.impl
}

func wrapNodes(impls []*graphimpl.NodeImpl) []*Node {
	if impls == nil {
		return nil
	}
	nodes := make([]*Node, len(impls))
	for i, impl := range impls {
		nodes[i] = wrapNode(impl)
	}
	return nodes
}

func unwrapNodes(nodes []*Node) []*graphimpl.NodeImpl {
	if nodes == nil {
		return nil
	}
	impls := make([]*graphimpl.NodeImpl, len(nodes))
	for i, n := range nodes {
		impls[i] = unwrapNode(n)
	}
	return impls
}

// Graph that holds this Node.
func (n *Node) Graph() *Graph {
	if n == nil {
		return nil
	}
	return wrapGraph(n.impl.Graph())
}

// Function scope that holds this Node.
func (n *Node) Function() *Function {
	if n == nil {
		return nil
	}
	return wrapFunction(n.impl.Function())
}

// Shape of the Node's output. It can be empty for multi-output internal nodes.
func (n *Node) Shape() shapes.Shape {
	if n == nil {
		return shapes.Shape{}
	}
	return n.impl.Shape()
}

// DType returns the DType of the node's shape.
func (n *Node) DType() dtypes.DType {
	return n.Shape().DType
}

// Rank returns the rank of the node's shape.
func (n *Node) Rank() int {
	return n.Shape().Rank()
}

// IsScalar returns whether the node's shape is a scalar.
func (n *Node) IsScalar() bool {
	return n.Shape().IsScalar()
}

// Id is the unique id of this node within the Graph.
func (n *Node) Id() NodeId {
	if n == nil {
		return InvalidNodeId
	}
	return n.impl.Id()
}

// GetParameterHandle returns the parameter id in the graph.
// It panics if node is not a parameter.
func (n *Node) GetParameterHandle() ParameterHandle {
	n.AssertValid()
	return n.impl.GetParameterHandle()
}

// GetParameterName returns the parameter name.
// If node is not a parameter, it panics.
func (n *Node) GetParameterName() string {
	n.AssertValid()
	return n.impl.GetParameterName()
}

// Inputs are the other nodes that are direct inputs to the node.
// This doesn't include static inputs for some operations that are not given by other Graph nodes.
func (n *Node) Inputs() []*Node {
	if n == nil {
		return nil
	}
	return wrapNodes(n.impl.Inputs())
}

// NumOutputs returns the number of outputs for a node.
func (n *Node) NumOutputs() int {
	if n == nil {
		return 0
	}
	return n.impl.NumOutputs()
}

// AssertValid panics if n is nil, or if its graph is invalid.
func (n *Node) AssertValid() {
	if n == nil {
		(*graphimpl.NodeImpl)(nil).AssertValid()
	}
	n.impl.AssertValid()
}

// SetLogged indicates that a node should be logged by executors, with the given message.
func (n *Node) SetLogged(message string) {
	n.impl.SetLogged(message)
}

// SetLoggedf indicates that a node should be logged by executors, with the given formatted message.
// See SetLogged.
func (n *Node) SetLoggedf(format string, args ...any) {
	n.impl.SetLoggedf(format, args...)
}

// IsLogged returns whether node is marked to be logged.
func (n *Node) IsLogged() bool {
	return n.impl.IsLogged()
}

// LogMessage associated with node, if any.
func (n *Node) LogMessage() string {
	return n.impl.LogMessage()
}

// Trace returns stack-trace in form of an error, of when the node was created.
// Only available if enabled by `Graph.SetTraced(true)`.
func (n *Node) Trace() error {
	return n.impl.Trace()
}

// String implements the `fmt.Stringer` interface.
func (n *Node) String() string {
	if n == nil {
		return "Node(nil)"
	}
	return n.impl.String()
}

// StopGradient returns whether node is a StopGradient.
func (n *Node) StopGradient() bool {
	if n == nil {
		return false
	}
	return n.impl.StopGradient()
}

// CustomGradient returns a registered custom gradient for the Node. See IdentityWithCustomGradient.
func (n *Node) CustomGradient() VJP {
	if n == nil || n.impl.CustomVJP() == nil {
		return nil
	}
	if vjp, ok := n.impl.CustomVJP().(VJP); ok {
		return vjp
	}
	fn, ok := n.impl.CustomVJP().(func(node *graphimpl.NodeImpl, vjpForOutputs []*graphimpl.NodeImpl, outputShape shapes.Shape) []*graphimpl.NodeImpl)
	if !ok || fn == nil {
		return nil
	}
	return func(node *Node, vjpOutputs []*Node, outputShape shapes.Shape) []*Node {
		return wrapNodes(fn(unwrapNode(node), unwrapNodes(vjpOutputs), outputShape))
	}
}
