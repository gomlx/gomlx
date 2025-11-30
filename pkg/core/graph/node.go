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
	"strings"

	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	MaxSizeToPrint = 5
)

// Node represents the result of an operation in the computation graph, and can be used as input to further operations.
//
// Internally, it keeps tracks of all parameters used for the computation: this is later used for auto-differentiation
// (see Gradient).
//
// It also stores meta-information: see Node.SetLogged, Node.StopGradient.
//
// Notice some complex methods offered in this package may be implemented with several instances of simpler operations
// and yield several nodes in the graph, that's normal.
//
// Node.String allows for a pretty-printing of node. To see the full graph with all nodes, use Graph.String.
type Node struct {
	graph        *Graph
	id           NodeId // id within graph.
	outputShapes []shapes.Shape
	outputOps    []backends.Op

	// inputNodes are the edges of the computation graph.
	// Notice that other static inputs to the node are registered in inputs
	inputNodes []*Node

	// inputs need to be
	inputs NodeInputs

	// alias is a name by which the Node be referred in the Graph.
	alias string

	// logMessage is set if node is marked for logging.
	logMessage string

	// stopGradient is set if no gradient is supposed to pass through.
	stopGradient bool

	// customVJP can be set for a custom reverse gradient definition for the function.
	// Usually, defined for a NoOp operation.
	customVJP VJP

	trace error // Stack-trace error of where Node was created. Stored if graph.traced is true.
}

// NodeInputs represents the inputs to node. The common interface is to return the type of the node.
// For the input parameters themselves, the pointer needs to be cast to the corresponding type, usually named
// inputNodes<backend_operation_name>, see generated gen_backend_ops.go
type NodeInputs interface {
	Type() NodeType

	// String prints a descriptive representation of the node, using its parameters.
	String() string
}

// Graph that holds this Node.
func (n *Node) Graph() *Graph {
	if n == nil {
		return nil
	}
	return n.graph
}

// Shape of the Node's output. It can be `nil`, for nodes that simply have a side effect, like a "Print" Node.
func (n *Node) Shape() shapes.Shape {
	if n == nil || n.NumOutputs() != 1 {
		return shapes.Shape{}
	}
	return n.outputShapes[0]
}

// DType returns the DType of the node's shapes.
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
	return n.id
}

// GetParameterHandle returns the parameter id in the graph.
// It panics if node is not a parameter.
func (n *Node) GetParameterHandle() ParameterHandle {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("node %s is not a Parameter node", n.Type())
	}
	return n.inputs.(*nodeInputsParameter).handle
}

// GetParameterName returns the parameter name.
// If node is not a parameter, it panics.
func (n *Node) GetParameterName() string {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("trying to get GetParameterName of a non-parameter node %q", n.Type())
	}
	return n.inputs.(*nodeInputsParameter).name
}

// Inputs are the other nodes that are direct inputNodes to the node.
// This doesn't include static inputNodes for some operations that are not given by other Graph nodes.
func (n *Node) Inputs() []*Node { return n.inputNodes }

// NumOutputs returns the number of outputs for a node.
//
// Almost every node will have one output only. But a few (like "RNGBitGenerator") will output various outputs
// that are split before usage. These nodes are marked with an invalid dtype.
//
// Used internally only, all Graph public operations will return nodes with one output only.
func (n *Node) NumOutputs() int {
	return len(n.outputOps)
}

// AssertValid panics if `n` is nil, or if its graph is invalid.
func (n *Node) AssertValid() {
	if n == nil {
		exceptions.Panicf("Node is nil")
	}
	if n.inputs == nil {
		exceptions.Panicf("Node in an invalid state")
	}
	n.graph.AssertValid()
}

// SetLogged indicates that a node should be logged by executors, with the given message.
func (n *Node) SetLogged(message string) {
	n.logMessage = message
}

// SetLoggedf indicates that a node should be logged by executors, with the given formatted message.
// See SetLogged.
func (n *Node) SetLoggedf(format string, args ...any) {
	n.SetLogged(fmt.Sprintf(format, args...))
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
	if n.graph == nil || !n.graph.IsValid() {
		return "Node(invalid graph)"
	}
	if n.alias != "" {
		str = fmt.Sprintf("[%q] ", n.alias)
	}
	if n.Type() == NodeTypeInvalid {
		str += "Invalid(?)"
	} else {
		str += n.inputs.String()
	}

	parts := []string{str}
	if n.logMessage != "" {
		parts = append(parts, "[Logged]")
	}
	if n.stopGradient {
		parts = append(parts, "[StopGradient]")
	}
	if n.customVJP != nil {
		parts = append(parts, "[CustomVJP]")
	}

	memory := xslices.Map(n.outputShapes, func(shape shapes.Shape) string {
		return humanize.Bytes(uint64(shape.Memory()))
	})

	str = fmt.Sprintf("%s -> %s - mem: %v", strings.Join(parts, " "), n.outputShapes, memory)
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
