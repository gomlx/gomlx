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
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	xslices "github.com/gomlx/gomlx/types/xslices"
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
	graph *Graph
	shape shapes.Shape
	id    NodeId // id within graph.
	op    backends.Op

	// inputNodes are the edges of the computation graph.
	// Notice that other static inputs to the node are registered in inputs
	inputNodes []*Node

	// inputs need to be
	inputs NodeInputs

	// logMessage is set if node is marked for logging.
	logMessage string

	// stopGradient is set if no gradient is supposed to pass through.
	stopGradient bool

	// customVJP can be set for a custom reverse gradient definition for the function.
	// Usually, defined for a NoOp operation.
	customVJP VJP

	// constValue is a multidimensional Go slice, kept for small values (like scalars), and only used for printing/debugging only.
	// See MinConstValueSizeToKeep to configure.
	constValue any

	trace error // Stack-trace error of where Node was created. Stored if graph.traced is true.
}

// NodeInputs represents the inputs to node. The common interface is to return the type of the node.
// For the input parameters themselves, the pointer needs to be cast to the corresponding type, usually named
// inputNodes<backend_operation_name>, see generated gen_backend_ops.go
type NodeInputs interface {
	Type() NodeType
}

// Type identify the operation performed by the node.
func (n *Node) Type() NodeType { return n.inputs.Type() }

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

// GetParameterHandle returns the parameter id in the graph.
// It panics if node is not a parameter.
func (n *Node) GetParameterHandle() ParameterHandle {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("node %s is not a Parameter node", n.Type())
	}
	inputs, ok := n.inputs.(*nodeInputsParameter)
	if !ok {
		exceptions.Panicf("Parameter node %s, but doesn't have a configured nodesInputsParamter, instead got a %T",
			n, n.inputs)
	}
	return inputs.handle
}

// GetParameterName returns the parameter name.
// If node is not a parameter, it panics.
func (n *Node) GetParameterName() string {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("trying to get GetParameterName of a non-parameter node %q", n.Type())
	}
	name, _, _ := xla.DecodeParameter(n.op)
	return name
}

// Inputs are the other nodes that are direct inputNodes to the node.
// This doesn't include static inputNodes for some operations that are not given by other Graph nodes.
func (n *Node) Inputs() []*Node { return n.inputNodes }

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
	if n.Type() == NodeTypeInvalid {
		str = "Invalid(?)"
	} else {
		str = n.Type().String()
	}
	inputIds := xslices.Map(n.inputNodes, func(node *Node) NodeId { return node.Id() })

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

	str = fmt.Sprintf("%s(id=%d, inputNodes=%v%s) -> %s", str, n.id, inputIds, partsStr, n.shape)
	return
}

// StopGradient returns weather node is a StopGradient.
func (n *Node) StopGradient() bool {
	return n.stopGradient
}

// CustomGradient returns a registered custom gradient for the Node. See IdentityWithCustomGradient.
func (n *Node) CustomGradient() VJP {
	return n.customVJP
}
