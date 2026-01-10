// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/require"
)

func TestAsserts(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	g := NewGraph(backend, "TestAssertGraph")
	node := Parameter(g, "node", shapes.Make(dtypes.Float32, 3, 2))
	scalar := Parameter(g, "scalar", shapes.Make(dtypes.Int64))

	// Check true asserts.
	require.NotPanics(t, func() { node.AssertDims(3, 2) })
	require.NotPanics(t, func() { node.AssertDims(-1, 2) })
	require.NotPanics(t, func() { node.AssertDims(3, -1) })
	require.NotPanics(t, func() { node.AssertDims(-1, -1) })
	require.NotPanics(t, func() { node.AssertRank(2) })
	require.NotPanics(t, func() { scalar.AssertScalar() })
	require.NotPanics(t, func() { scalar.AssertRank(0) })

	// Check false asserts.
	require.Panics(t, func() { node.AssertDims(3) })     // Not enough dimensions
	require.Panics(t, func() { node.AssertDims(-1, 1) }) // One dimension is wrong
	require.Panics(t, func() { node.AssertDims(4, 2) })  // One dimension is wrong
	require.Panics(t, func() { node.AssertRank(3) })     // Wrong rank
	require.Panics(t, func() { node.AssertScalar() })    // Wrong rank
	require.Panics(t, func() { scalar.AssertRank(1) })   // Wrong rank
}
