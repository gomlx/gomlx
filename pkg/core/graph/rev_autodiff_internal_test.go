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

package graph

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/shapes"
)

func testState(t *testing.T, rg *reverseGraph, node *Node, selected, included, useful bool) {
	rNode := rg.ReverseNodes[node.Id()]
	if rNode.Selected != selected || rNode.Included != included || rNode.Useful != useful {
		t.Errorf("Node %q has unexpected state (%v, %v, %v), wanted (%v, %v, %v)",
			node, rNode.Selected, rNode.Included, rNode.Useful, selected, included, useful)
	}
}

func TestReverseGraph(t *testing.T) {
	backend := buildTestBackend()

	g := NewGraph(backend, "TestReverseGraph")
	n0 := Parameter(g, "n0", shapes.Scalar[float32]())
	n1 := Parameter(g, "n1", shapes.Scalar[float32]())
	n2 := Parameter(g, "n2", shapes.Scalar[float32]())
	n3 := Add(n0, n1)
	n4 := Mul(n3, n2)
	n5 := Div(n3, n2)
	_ = n5
	rg := newReverseGraph(g, n4, []*Node{n0})

	// Enumerate expectations for results of selected/included/useful.
	want := []struct {
		node                       *Node
		selected, included, useful bool
	}{
		{n0, true, true, true},
		{n1, false, true, false},
		{n2, false, true, false},
		{n3, false, true, true},
		{n4, false, true, true},
		{n5, false, false, false},
	}
	for _, e := range want {
		testState(t, rg, e.node, e.selected, e.included, e.useful)
	}
}
