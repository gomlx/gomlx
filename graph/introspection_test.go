package graph_test

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestIsConstantExpression(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "")
	a := Const(g, 5.0)
	b := Const(g, 7.0)
	c := Add(a, b)
	d := Parameter(g, "d", a.Shape())
	e := Mul(d, c)

	require.True(t, a.IsConstantExpression())
	require.True(t, c.IsConstantExpression())
	require.False(t, d.IsConstantExpression())
	require.False(t, e.IsConstantExpression())
}
