/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	xla "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
	"strings"
)

const (
	MaxSizeToPrint = 5
)

// Node implements Node and is a standard node implementation that using a xla.SerializedNode definition
// can be used by most ops (node types).
//
// Almost every new node type implementation will rely on the Node.
type Node struct {
	graph  *Graph
	shape  shapes.Shape
	id     NodeId // id within graph.
	op     *xla.Op
	inputs []*Node

	// logMessage is set if node is marked for logging.
	logMessage string

	// stopGradient is set if no gradient is supposed to pass through.
	stopGradient bool

	// customVJP can be set for a custom reverse gradient definition for the function.
	// Usually, defined for a NoOp operation.
	customVJP VJP

	// constValue is a multi-dimensional Go slice, kept for small values (like scalars), and only used for printing/debugging only.
	// See MinConstValueSizeToKeep to configure.
	constValue any

	trace error // Stack-trace error of where Node was created. Stored if graph.traced is true.
}

// Type identify the operation performed by the node.
func (n *Node) Type() xla.OpType { return n.op.Type }

// Graph that holds this Node.
func (n *Node) Graph() *Graph {
	if n == nil {
		return nil
	}
	return n.graph
}

// Shape of the Node's output. It can be `nil`, for nodes that simply have a side effect, like a "Print" Node.
func (n *Node) Shape() shapes.Shape {
	if n == nil {
		return shapes.Shape{}
	}
	return n.shape
}

// DType returns the DType of the node's shape.
func (n *Node) DType() dtypes.DType {
	return n.shape.DType
}

// Rank returns the rank of the node's shape.
func (n *Node) Rank() int {
	return n.shape.Rank()
}

// IsScalar returns whether the node's shape is a scalar.
func (n *Node) IsScalar() bool {
	return n.shape.IsScalar()
}

// Id is the unique id of this node within the Graph.
func (n *Node) Id() NodeId {
	return n.id
}

// ParameterHandle returns the parameter id in the graph.
// It returns InvalidParameterHandle if node is not a parameter.
func (n *Node) ParameterHandle() ParameterHandle {
	n.AssertValid()
	if n.Type() != xla.ParameterOp {
		return InvalidParameterHandle
	}
	_, idx, _ := xla.DecodeParameter(n.op)
	return ParameterHandle(idx)
}

const NotAParameterStr = "NOT_A_PARAMETER"

// ParameterName returns the parameter name if this node is a parameter. Otherwise, it returns NotAParameterStr
func (n *Node) ParameterName() string {
	n.AssertValid()
	if n.Type() != xla.ParameterOp {
		exceptions.Panicf("trying to get ParameterName of a non-parameter node %q", n.Type())
	}
	name, _, _ := xla.DecodeParameter(n.op)
	return name
}

// Inputs are the other nodes that are direct inputs to the node.
// This doesn't include static inputs for some operations that are not given by other Graph nodes.
func (n *Node) Inputs() []*Node { return n.inputs }

// AssertValid panics if `n` is nil, or if its graph is invalid.
func (n *Node) AssertValid() {
	if n == nil {
		panic(errors.New("Node is nil"))
	}
	n.graph.AssertValid()
}

// SetLogged indicates that a node should be logged by executors.
func (n *Node) SetLogged(message string) {
	n.logMessage = message
}

// IsLogged returns whether node is marked to be logged.
func (n *Node) IsLogged() bool {
	return n.logMessage != ""
}

// LogMessage associated with node, if any.
func (n *Node) LogMessage() string {
	return n.logMessage
}

// Trace returns stack-trace in form of an error, of when the node was created.
// Only available if enabled by `Graph.SetTraced(true)`.
func (n *Node) Trace() error {
	return n.trace
}

// String implements the `fmt.Stringer` interface.
// Logged nodes are marked with (*).
func (n *Node) String() (str string) {
	if n == nil {
		return "Node(nil)"
	}
	if n.graph == nil || n.graph.IsValid() {
		return "Node(graph == nil or invalid)"
	}
	if n.Type() == xla.InvalidOp {
		str = "NoOp"
	} else {
		str = n.Type().String()
	}
	inputIds := make([]NodeId, 0, len(n.inputs))
	for _, inputNode := range n.inputs {
		inputIds = append(inputIds, inputNode.Id())
	}

	var parts []string
	if n.constValue != nil {
		parts = append(parts, fmt.Sprintf("value=%v", n.constValue))
	}
	if n.logMessage != "" {
		parts = append(parts, "[Logged]")
	}
	if n.stopGradient {
		parts = append(parts, "[StopGradient]")
	}
	var partsStr string
	if len(parts) > 0 {
		partsStr = ", " + strings.Join(parts, ", ")
	}

	str = fmt.Sprintf("%s(id=%d, inputs=%v%s) -> %s", str, n.id, inputIds, partsStr, n.shape)
	return
}

// newNode creates a Node based on the given XlaBuilder op, registers it in the Graph.
func newNode(graph *Graph, op *xla.Op) (node *Node) {
	graph.AssertBuilding()
	node = &Node{
		graph: graph,
		op:    op,
	}
	graph.registerNode(node)
	node.setInputs()
	if graph.traced {
		node.trace = errors.New("Stack-trace")
	}
	op.UserPayload = node
	return
}

// setInputs caches the known inputs of the xlabuilder.Op associated with the Node.
// Only the xlabuilder.Op's known the Graph are cached.
func (n *Node) setInputs() {
	if len(n.op.OpInputs) == 0 {
		n.inputs = nil
		return
	}
	n.inputs = make([]*Node, 0, len(n.op.OpInputs))
	g := n.graph
	for _, inputOp := range n.op.OpInputs {
		inputNode, found := g.opToNode[inputOp]
		if found {
			n.inputs = append(n.inputs, inputNode)
		}
	}
}

// StopGradient returns weather node is a StopGradient.
func (n *Node) StopGradient() bool {
	return n.stopGradient
}

// CustomGradient returns a registered custom gradient for the Node. See IdentityWithCustomGradient.
func (n *Node) CustomGradient() VJP {
	return n.customVJP
}
