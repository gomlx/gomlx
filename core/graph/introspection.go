// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/sets"
)

// Type identifies the operation performed by the node.
func (n *Node) Type() NodeType {
	if n == nil {
		return NodeTypeInvalid
	}
	return n.impl.Type()
}

// ConstantValue returns the value assigned to a constant node, or nil if n.Type() != NodeTypeConstant.
func (n *Node) ConstantValue() *tensors.Tensor {
	if n == nil {
		return nil
	}
	return n.impl.ConstantValue()
}

// IsConstantExpression returns whether the Node is a Constant or an expression that depends only on constant values.
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
