// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/support/sets"

	"github.com/stretchr/testify/assert"
)

func TestNodeAliases(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "Graph With Aliases")

	// Create some nodes
	n1 := ScalarZero(g, dtypes.Float32)
	n2 := ScalarOne(g, dtypes.Float32)

	// Test adding aliases
	n1.WithAlias("n1")
	n2.WithAlias("n2")

	assert.Equal(t, n1, g.GetNodeByAlias("n1"))
	assert.Equal(t, n2, g.GetNodeByAlias("n2"))
	assert.Nil(t, g.GetNodeByAlias("n3"), "Node with alias 'n3' should not exist yet")

	// Test alias scopes
	g.PushAliasScope("scope1")
	n3 := Add(n1, n2)
	n3.WithAlias("n3")
	assert.Equal(t, n3, g.GetNodeByAlias("/scope1/n3"))
	assert.Equal(t, n3, g.GetNodeByAlias("n3"))
	assert.Nil(t, g.GetNodeByAlias("n1"))         // "n1" doesn't exist in this scope.
	assert.True(t, n1 == g.GetNodeByAlias("/n1")) // But it still exists in the global scope.

	// Test IterAliasedNodes:
	aliases := sets.Make[string]()
	for alias, node := range g.IterAliasedNodes() {
		fmt.Printf("\tGraph[%q] = %s\n", alias, node)
		aliases.Insert(alias)
	}
	assert.Len(t, aliases, 3)
	assert.True(t, aliases.Has("/n1"))
	assert.True(t, aliases.Has("/n2"))
	assert.True(t, aliases.Has("/scope1/n3"))

	// Test popping alias scope
	g.PopAliasScope()
	assert.Equal(t, n3, g.GetNodeByAlias("/scope1/n3")) // "/scope1/n3" still exists.
	assert.True(t, g.GetNodeByAlias("n3") == nil)       // But it is now on a different context.
}
