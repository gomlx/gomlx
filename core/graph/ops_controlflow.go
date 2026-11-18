// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import "github.com/gomlx/gomlx/internal/graphimpl"

// While executes a loop while a condition is true.
func While(cond, body *Function, initialState ...*Node) []*Node {
	return wrapNodes(graphimpl.While(unwrapFunction(cond), unwrapFunction(body), unwrapNodes(initialState)...))
}

// If executes one of two branches based on a boolean predicate.
func If(pred *Node, trueBranch, falseBranch *Function) []*Node {
	return wrapNodes(graphimpl.If(unwrapNode(pred), unwrapFunction(trueBranch), unwrapFunction(falseBranch)))
}
