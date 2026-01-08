package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInnermostFunction(t *testing.T) {
	// We use a nil backend and builder because we are only testing the scope logic,
	// which doesn't depend on the backend for this unit test.
	g := &Graph{name: "test_graph"}
	g.mainFunc = &Function{name: "main", graph: g}
	g.currentFunc = g.mainFunc

	// Create a hierarchy of functions
	// main -> sub1 -> sub2
	sub1 := &Function{name: "sub1", parent: g.mainFunc, graph: g}
	sub2 := &Function{name: "sub2", parent: sub1, graph: g}

	// disjoint function
	// main -> other1
	other1 := &Function{name: "other1", parent: g.mainFunc, graph: g}

	createNode := func(scope *Function) *Node {
		if scope == nil {
			scope = g.mainFunc
		}
		return &Node{graph: g, scope: scope}
	}

	t.Run("Empty inputs", func(t *testing.T) {
		scope, err := innermostFunction(nil)
		assert.NoError(t, err)
		assert.Nil(t, scope)
	})

	t.Run("Single input main scope", func(t *testing.T) {
		n1 := createNode(g.mainFunc)
		scope, err := innermostFunction([]*Node{n1})
		assert.NoError(t, err)
		assert.Equal(t, g.mainFunc, scope)
	})

	t.Run("Mixed scopes compatible", func(t *testing.T) {
		nMain := createNode(g.mainFunc)
		nSub1 := createNode(sub1)
		nSub2 := createNode(sub2)

		// inputs: [main, sub1] -> deepest is sub1
		scope, err := innermostFunction([]*Node{nMain, nSub1})
		assert.NoError(t, err)
		assert.Equal(t, sub1, scope)

		// inputs: [sub1, main] -> deepest is sub1 (order shouldn't matter)
		scope, err = innermostFunction([]*Node{nSub1, nMain})
		assert.NoError(t, err)
		assert.Equal(t, sub1, scope)

		// inputs: [main, sub2, sub1] -> deepest is sub2
		scope, err = innermostFunction([]*Node{nMain, nSub2, nSub1})
		assert.NoError(t, err)
		assert.Equal(t, sub2, scope)
	})

	t.Run("Disjoint scopes", func(t *testing.T) {
		nSub1 := createNode(sub1)
		nOther1 := createNode(other1)

		// inputs: [sub1, other1] -> disjoint
		_, err := innermostFunction([]*Node{nSub1, nOther1})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "incompatible scopes")
	})

	t.Run("Nil node or scope", func(t *testing.T) {
		n1 := createNode(g.mainFunc)
		_, err := innermostFunction([]*Node{n1, nil})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "is nil")

		nNoScope := &Node{graph: g, scope: nil}
		_, err = innermostFunction([]*Node{n1, nNoScope})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "nil scope")
	})
}
