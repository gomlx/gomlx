// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
)

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

// IsConstantExpression returns whether the Node is a Constant or an expression that depends only on constant values.
// It traverses all the node dependencies and checks that all leaf nodes are constants.
func (n *Node) IsConstantExpression() bool {
	return isConstantTraverseSubGraph(n, sets.Make[*Node]())
}

func isConstantTraverseSubGraph(node *Node, visited sets.Set[*Node]) bool {
	if visited.Has(node) {
		return true
	}
	isConstant := node.Type() != NodeTypeParameter
	for _, input := range node.Inputs() {
		isConstant = isConstant && isConstantTraverseSubGraph(input, visited)
	}
	return isConstant
}
