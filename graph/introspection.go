package graph

import "github.com/gomlx/gomlx/types/tensors"

// This file defines methods that allow for introspection of the graph.
//
// The API is limited -- because we want flexibility to change the implementation without concerns on breaking
// compatibility.

// Type identify the operation performed by the node.
// It's an "introspection" method.
func (n *Node) Type() NodeType {
	if n == nil || n.inputs == nil {
		return NodeTypeInvalid
	}
	return n.inputs.Type()
}

// ConstantValue returns the value assigned to a constant node.
// It's an "introspection" method.
// It returns nil if n.Type() != NodeTypeConstant.
func (n *Node) ConstantValue() *tensors.Tensor {
	if n.Type() != NodeTypeConstant {
		return nil
	}
	params := n.inputs.(*nodeInputsConstant)
	return params.tensor
}
