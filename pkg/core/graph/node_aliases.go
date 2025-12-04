package graph

import (
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/internal/exceptions"
)

// PushAliasScope pushes another scope to the current alias scope for new aliases.
//
// For instance, for an image model, one may want to push a scope per layer, and create
// an alias "output" to the node with the output of the layer. The different scope helps
// differentiate the different "output" aliases nodes.
//
// Notice this is orthogonal to the context.Context scope used for variables.
// That's because for instance, one may reuse a model multiple times for different inputs (e.g.: triplet loss
// will use the same model for the "anchor", "positive" and "negative" examples, or a style transfer model
// will use the same embedding model for the "source", "style" and "target" images), so the variables context
// scope is the same, but we want a different alias scope, so we can access the outputs per layer of each
// type of the example separately.
//
// Each call to Graph.PushAliasScope should be matched by a call to Graph.PopAliasScope, usually using defer.
func (g *Graph) PushAliasScope(scope string) {
	g.aliasScope = append(g.aliasScope, scope)
}

// PopAliasScope removes the scope previously pushed with PushAliasScope.
//
// It panics if there are no scopes pushed.
func (g *Graph) PopAliasScope() {
	if len(g.aliasScope) == 0 {
		exceptions.Panicf("no scopes pushed when calling Graph.PopAliasScope")
	}
	g.aliasScope = g.aliasScope[:len(g.aliasScope)-1]
}

// WithAlias sets an alias in the Graph for the node.
// It allows it to be retrieved with Graph.GetNodeByAlias.
//
// The alias is automatically prefixed with the Graph current "alias scope", see Graph.PushAliasScope and
// Graph.PopAliasScope. Except if the alias starts with AliasScopeSeparator ("/"), in which case it is
// assumed to be given with an "absolute scope path".
//
// It returns the Node itself, to allow cascading method calling.
//
// It panics if the exact same alias already exists.
func (n *Node) WithAlias(alias string) *Node {
	n.Graph().insertNodeAlias(n, alias)
	return n
}

// GetAlias returns the alias (with the absolute path) of the current node, if one was registered
// with Node.WithAlias, otherwise returns "".
func (n *Node) GetAlias() string {
	return n.alias
}

// AliasScopeSeparator is the string used to join the individual alias scope parts as well as
// the alias itself. So if the scope is currently ["a", "b"] and an alias "output" is created,
// it will be renamed "/a/b/output".
const AliasScopeSeparator = "/"

// absoluteAlias returns an alias with an absolute path, by prepending the current scope.
//
// If alias is absolute (starts with "/") it is not changed.
func (g *Graph) absoluteAlias(alias string) string {
	if strings.HasPrefix(alias, AliasScopeSeparator) {
		return alias
	}
	// Prepend alias with current scope.
	if len(g.aliasScope) == 0 {
		return fmt.Sprintf("%s%s", AliasScopeSeparator, alias)
	}
	return fmt.Sprintf("%s%s%s%s",
		AliasScopeSeparator, strings.Join(g.aliasScope, AliasScopeSeparator), AliasScopeSeparator, alias)
}

// insertNodeAlias will prepend the given alias with the current scope (separated by "/") and
func (g *Graph) insertNodeAlias(n *Node, alias string) {
	alias = g.absoluteAlias(alias)
	if _, found := g.aliasToNode[alias]; found {
		exceptions.Panicf("alias already exists in Node.WithAlias(%q) -- they must be unique within the scope (path) they are defined",
			alias)
	}
	n.alias = alias
	g.aliasToNode[alias] = n
}

// GetNodeByAlias returns a node with the given alias or nil if it didn't find it.
//
// If the search alias has an absolute scope (path), meaning if it starts with AliasScopeSeparator,
// it is searched as is. If not, it is prefixed with the current scope before searching.
//
// See Node.WithAlias to create node aliases, and Graph.PushAliasScope and Graph.PopAliasScope to
// manipulate the scope of the aliases created.
func (g *Graph) GetNodeByAlias(alias string) *Node {
	alias = g.absoluteAlias(alias)
	return g.aliasToNode[alias]
}

// IterAliasedNodes provides an iterator over all aliased nodes. It yields pairs (alias, node).
// The aliases are sorted before iteration.
func (g *Graph) IterAliasedNodes() iter.Seq2[string, *Node] {
	type Item struct {
		Alias string
		Node  *Node
	}
	items := make([]Item, 0, len(g.aliasToNode))
	for alias, node := range g.aliasToNode {
		items = append(items, Item{alias, node})
	}
	slices.SortFunc(items, func(a, b Item) int {
		return strings.Compare(a.Alias, b.Alias)
	})
	return func(yield func(string, *Node) bool) {
		for _, pair := range items {
			next := yield(pair.Alias, pair.Node)
			if !next {
				return
			}
		}
	}
}
