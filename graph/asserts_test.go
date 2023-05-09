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
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestAsserts(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("TestAssertGraph")
	node := g.Parameter("node", shapes.Make(shapes.F32, 3, 2))
	scalar := g.Parameter("scalar", shapes.Make(shapes.I64))

	// Verify correct checks.
	require.Truef(t, g.Ok(), "Graph.Ok() is false!?: %+v", g.Error())
	require.Truef(t, node.AssertDims(3, 2), "Assert failed: %+v", g.Error())
	require.Truef(t, node.AssertDims(-1, 2), "Assert failed: %+v", g.Error())
	require.Truef(t, node.AssertDims(3, -1), "Assert failed: %+v", g.Error())
	require.Truef(t, node.AssertDims(-1, -1), "Assert failed: %+v", g.Error())
	require.Truef(t, node.AssertRank(2), "Assert failed: %+v", g.Error())
	require.Truef(t, scalar.AssertScalar(), "Assert failed: %+v", g.Error())
	require.Truef(t, scalar.AssertRank(0), "Assert failed: %+v", g.Error())

	// Verify for false asserts.
	require.False(t, node.AssertDims(3)) // Not enough dimensions
	require.Error(t, g.Error())
	g.ResetError()
	require.False(t, node.AssertDims(-1, 1)) // One dimension is wrong
	require.Error(t, g.Error())
	g.ResetError()
	require.False(t, node.AssertDims(4, 2)) // One dimension is wrong
	require.Error(t, g.Error())
	g.ResetError()
	require.False(t, node.AssertRank(3)) // Wrong rank
	require.Error(t, g.Error())
	g.ResetError()
	require.False(t, node.AssertScalar()) // Wrong rank
	require.Error(t, g.Error())
	g.ResetError()
	require.False(t, scalar.AssertRank(1)) // Wrong rank
	require.Error(t, g.Error())
	g.ResetError()
}
