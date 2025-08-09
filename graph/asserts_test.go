/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
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
