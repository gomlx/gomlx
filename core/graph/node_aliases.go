// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"iter"

	"github.com/gomlx/gomlx/internal/graphimpl"
)

// AliasScopeSeparator is the string used to join alias scope parts.
const AliasScopeSeparator = graphimpl.AliasScopeSeparator

// PushAliasScope pushes another scope to the current alias scope for new aliases.
func (g *Graph) PushAliasScope(scope string) {
	g.impl.PushAliasScope(scope)
}

// PopAliasScope removes the scope previously pushed with PushAliasScope.
func (g *Graph) PopAliasScope() {
	g.impl.PopAliasScope()
}

// WithAlias sets an alias in the Graph for the node.
func (n *Node) WithAlias(alias string) *Node {
	n.impl.WithAlias(alias)
	return n
}

// GetAlias returns the alias (with the absolute path) of the current node, or "".
func (n *Node) GetAlias() string {
	return n.impl.GetAlias()
}

// GetNodeByAlias returns a node with the given alias or nil if it didn't find it.
func (g *Graph) GetNodeByAlias(alias string) *Node {
	return wrapNode(g.impl.GetNodeByAlias(alias))
}

// IterAliasedNodes provides an iterator over all aliased nodes, sorted by alias.
func (g *Graph) IterAliasedNodes() iter.Seq2[string, *Node] {
	return func(yield func(string, *Node) bool) {
		for alias, impl := range g.impl.IterAliasedNodes() {
			if !yield(alias, wrapNode(impl)) {
				return
			}
		}
	}
}
