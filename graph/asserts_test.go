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

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/stretchr/testify/require"
)

func TestAsserts(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("TestAssertGraph")
	node := g.Parameter("node", shapes.Make(shapes.F32, 3, 2))
	scalar := g.Parameter("scalar", shapes.Make(shapes.I64))

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
