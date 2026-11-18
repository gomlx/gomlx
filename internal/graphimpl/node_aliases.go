// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/support/exceptions"
)

// AliasScopeSeparator joins alias scope segments.
const AliasScopeSeparator = "/"

// PushAliasScope pushes another scope to the current alias scope for new aliases.
func (g *GraphImpl) PushAliasScope(scope string) {
	g.aliasScope = append(g.aliasScope, scope)
}

// PopAliasScope removes the scope previously pushed with PushAliasScope.
func (g *GraphImpl) PopAliasScope() {
	if len(g.aliasScope) == 0 {
		exceptions.Panicf("no scopes pushed when calling Graph.PopAliasScope")
	}
	g.aliasScope = g.aliasScope[:len(g.aliasScope)-1]
}

// WithAlias sets an alias in the GraphImpl for the node.
func (n *NodeImpl) WithAlias(alias string) *NodeImpl {
	n.Graph().insertNodeAlias(n, alias)
	return n
}

// GetAlias returns the alias of the current node, or "".
func (n *NodeImpl) GetAlias() string {
	return n.alias
}

func (g *GraphImpl) absoluteAlias(alias string) string {
	if strings.HasPrefix(alias, AliasScopeSeparator) {
		return alias
	}
	if len(g.aliasScope) == 0 {
		return fmt.Sprintf("%s%s", AliasScopeSeparator, alias)
	}
	return fmt.Sprintf("%s%s%s%s",
		AliasScopeSeparator, strings.Join(g.aliasScope, AliasScopeSeparator), AliasScopeSeparator, alias)
}

func (g *GraphImpl) insertNodeAlias(n *NodeImpl, alias string) {
	alias = g.absoluteAlias(alias)
	if _, found := g.aliasToNode[alias]; found {
		exceptions.Panicf("alias already exists in Node.WithAlias(%q) -- they must be unique within the scope (path) they are defined",
			alias)
	}
	n.alias = alias
	g.aliasToNode[alias] = n
}

// GetNodeByAlias returns a node with the given alias or nil if not found.
func (g *GraphImpl) GetNodeByAlias(alias string) *NodeImpl {
	alias = g.absoluteAlias(alias)
	return g.aliasToNode[alias]
}

// IterAliasedNodes yields pairs (alias, node) sorted by alias.
func (g *GraphImpl) IterAliasedNodes() iter.Seq2[string, *NodeImpl] {
	type item struct {
		alias string
		node  *NodeImpl
	}
	items := make([]item, 0, len(g.aliasToNode))
	for alias, node := range g.aliasToNode {
		items = append(items, item{alias, node})
	}
	slices.SortFunc(items, func(a, b item) int {
		return strings.Compare(a.alias, b.alias)
	})
	return func(yield func(string, *NodeImpl) bool) {
		for _, pair := range items {
			if !yield(pair.alias, pair.node) {
				return
			}
		}
	}
}
