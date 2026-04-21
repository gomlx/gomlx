// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestIsConstantExpression(t *testing.T) {
	backend := testutil.BuildTestBackend()
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
