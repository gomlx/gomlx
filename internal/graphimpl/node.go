// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/exceptions"
)

const MaxSizeToPrint = 5

// NodeImpl is the internal representation of a node in a computation graph.
type NodeImpl struct {
	public any // Back-link to canonical *graph.Node wrapper (for 1-to-1 pointer identity).

	graph        *GraphImpl
	scope        *FunctionImpl
	id           NodeId
	outputShapes []shapes.Shape
	outputOps    []compute.Value

	inputNodes []*NodeImpl
	inputs     NodeInputs

	alias        string
	logMessage   string
	stopGradient bool

	// customVJP stores a graph.VJP (typed as any to avoid importing core/graph).
	customVJP any

	// vjpAlternateOutputs holds decomposed versions of this node's computation built from
	// primitive ops, used during reverse-mode autodiff when no fused VJP exists.
	vjpAlternateOutputs []*NodeImpl

	isCheckpointed         bool
	needsRematerialization bool

	trace error
}

// NodeInputs represents the inputs to a NodeImpl.
type NodeInputs interface {
	Type() NodeType
	String() string
	CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl
}

// Public returns the canonical *graph.Node wrapper (stored as any), or nil.
func (n *NodeImpl) Public() any {
	if n == nil {
		return nil
	}
	return n.public
}

// SetPublic records the canonical *graph.Node wrapper on n.
func (n *NodeImpl) SetPublic(pub any) {
	n.public = pub
}

// Graph returns the GraphImpl that holds this node.
func (n *NodeImpl) Graph() *GraphImpl {
	if n == nil {
		return nil
	}
	return n.graph
}

// Function returns the FunctionImpl scope that holds this node.
func (n *NodeImpl) Function() *FunctionImpl {
	if n == nil {
		return nil
	}
	return n.scope
}

// Shape returns the Shape of the node's output (empty for multi-output nodes).
func (n *NodeImpl) Shape() shapes.Shape {
	if n == nil || n.NumOutputs() != 1 {
		return shapes.Shape{}
	}
	return n.outputShapes[0]
}

// DType returns the DType of the node's shape.
func (n *NodeImpl) DType() dtypes.DType {
	return n.Shape().DType
}

// Rank returns the rank of the node's shape.
func (n *NodeImpl) Rank() int {
	return n.Shape().Rank()
}

// IsScalar returns whether the node's shape is a scalar.
func (n *NodeImpl) IsScalar() bool {
	return n.Shape().IsScalar()
}

// Id returns the unique id of this node within the GraphImpl.
func (n *NodeImpl) Id() NodeId {
	return n.id
}

// GetParameterHandle returns the parameter id in the graph.
func (n *NodeImpl) GetParameterHandle() ParameterHandle {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("node %s is not a Parameter node", n.Type())
	}
	return n.inputs.(*NodeInputsParameter).Handle
}

// GetParameterName returns the parameter name.
func (n *NodeImpl) GetParameterName() string {
	n.AssertValid()
	if n.Type() != NodeTypeParameter {
		exceptions.Panicf("trying to get GetParameterName of a non-parameter node %q", n.Type())
	}
	return n.inputs.(*NodeInputsParameter).Name
}

// Inputs returns the direct input nodes.
func (n *NodeImpl) Inputs() []*NodeImpl { return n.inputNodes }

// NodeInputs returns the NodeInputs descriptor for this node.
func (n *NodeImpl) NodeInputs() NodeInputs { return n.inputs }

// OutputOps returns the underlying backend values produced by this node.
func (n *NodeImpl) OutputOps() []compute.Value { return n.outputOps }

// OutputShapes returns the output shapes produced by this node.
func (n *NodeImpl) OutputShapes() []shapes.Shape { return n.outputShapes }

// SetInputNodes overrides the tracked inputNodes slice on n (e.g. for DynamicPad).
func (n *NodeImpl) SetInputNodes(inputs []*NodeImpl) {
	n.inputNodes = inputs
}

// SetOutputShape overrides the output shape at idx on n (e.g. to preserve dynamic axis names).
func (n *NodeImpl) SetOutputShape(idx int, shape shapes.Shape) {
	n.outputShapes[idx] = shape
}

// NumOutputs returns the number of outputs for this node.
func (n *NodeImpl) NumOutputs() int {
	return len(n.outputOps)
}

// AssertValid panics if n is nil or its graph is invalid.
func (n *NodeImpl) AssertValid() {
	if n == nil {
		exceptions.Panicf("Node is nil")
	}
	if n.inputs == nil {
		exceptions.Panicf("Node in an invalid state")
	}
	n.graph.AssertValid()
}

// SetLogged marks a node to be logged by executors with the given message.
func (n *NodeImpl) SetLogged(message string) {
	n.logMessage = message
}

// SetLoggedf marks a node to be logged with a formatted message.
func (n *NodeImpl) SetLoggedf(format string, args ...any) {
	n.SetLogged(fmt.Sprintf(format, args...))
}

// IsLogged returns whether node is marked to be logged.
func (n *NodeImpl) IsLogged() bool {
	return n.logMessage != ""
}

// LogMessage returns the log message associated with the node, if any.
func (n *NodeImpl) LogMessage() string {
	return n.logMessage
}

// Trace returns the stack-trace error of when the node was created, if Graph.SetTraced(true) was set.
func (n *NodeImpl) Trace() error {
	return n.trace
}

// Type returns the NodeType of this node.
func (n *NodeImpl) Type() NodeType {
	if n == nil || n.inputs == nil {
		return NodeTypeInvalid
	}
	return n.inputs.Type()
}

// ConstantValue returns the tensor stored in a constant node (when size < MinConstValueSizeToKeep).
func (n *NodeImpl) ConstantValue() *tensors.Tensor {
	if n.Type() != NodeTypeConstant {
		return nil
	}
	return n.inputs.(*NodeInputsConstant).Tensor
}

// String implements fmt.Stringer.
func (n *NodeImpl) String() (str string) {
	if n == nil {
		return "Node(nil)"
	}
	if n.graph == nil || !n.graph.IsValid() {
		return fmt.Sprintf("Node #%d(invalid graph)", n.id)
	}
	str = fmt.Sprintf("Node #%d", n.id)
	if n.alias != "" {
		str += fmt.Sprintf(" [%q]", n.alias)
	}
	if n.Type() == NodeTypeInvalid {
		str += " Invalid(?)"
	} else {
		str += " " + n.inputs.String()
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
		if shape.IsDynamic() {
			return "dynamic"
		}
		return humanize.Bytes(uint64(shape.ByteSize()))
	})

	str = fmt.Sprintf("%s -> %s - mem: %v", strings.Join(parts, " "), n.outputShapes, memory)
	return
}

// StopGradient returns whether node is marked with StopGradient.
func (n *NodeImpl) StopGradient() bool {
	return n.stopGradient
}

// SetStopGradient sets the stopGradient flag.
func (n *NodeImpl) SetStopGradient(stop bool) {
	n.stopGradient = stop
}

// CustomVJP returns the registered custom gradient function (stored as any), if any.
func (n *NodeImpl) CustomVJP() any {
	return n.customVJP
}

// SetCustomVJP sets the custom gradient function (stored as any).
func (n *NodeImpl) SetCustomVJP(vjp any) {
	n.customVJP = vjp
}

// VJPAlternateOutputs returns the decomposed alternate outputs for VJP computation.
func (n *NodeImpl) VJPAlternateOutputs() []*NodeImpl {
	return n.vjpAlternateOutputs
}

// SetVJPAlternateOutputs sets the decomposed alternate outputs for VJP computation.
func (n *NodeImpl) SetVJPAlternateOutputs(alts []*NodeImpl) {
	n.vjpAlternateOutputs = alts
}

// IsCheckpointed reports whether this node is a checkpoint boundary.
func (n *NodeImpl) IsCheckpointed() bool {
	return n.isCheckpointed
}

// NeedsRematerialization reports whether this node's inputs must be rematerialized during backward pass.
func (n *NodeImpl) NeedsRematerialization() bool {
	return n.needsRematerialization
}

// SetCheckpointed sets the checkpoint boundary flags on this node.
func (n *NodeImpl) SetCheckpointed(isCheckpointed, needsRematerialization bool) {
	n.isCheckpointed = isCheckpointed
	n.needsRematerialization = needsRematerialization
}

// CloneWithInputs creates a clone of the node using newInputs instead of the original ones.
func (n *NodeImpl) CloneWithInputs(newInputs ...*NodeImpl) *NodeImpl {
	n.AssertValid()
	cloned := n.inputs.CloneWithInputs(n, newInputs...)
	if cloned != nil {
		cloned.stopGradient = n.stopGradient
		cloned.customVJP = n.customVJP
		cloned.alias = n.alias
	}
	return cloned
}

// NewNode constructs and registers a NodeImpl in g with the given outputs and inputs.
func NewNode(g *GraphImpl, outputOps []compute.Value, outputShapes []shapes.Shape, inputs NodeInputs, inputNodes []*NodeImpl) *NodeImpl {
	node := &NodeImpl{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return node
}
