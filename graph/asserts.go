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
	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/types/shapes"
)

// This file implements various asserts (checks) that can be done on the Node. They are derived
// from the asserts in the shape package. But instead of panic'ing, they just report
// the error to the Graph.

// AssertDims checks whether the shape has the given dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// If the shape is not what was expected, it sets an error in the associated Graph and returns false.
// If the Graph is already in error state, it also returns false.
//
// This often serves as documentation for the code when implementing some complex computational
// graphs. This allows the reader of the code to corroborte what is the expected shape of a node.
//
// Example:
//
// ```
//
//	batch_size := inputs[0].Shape().Dimensions[0]
//	...
//	layer := Concatenate(allEmbeddings, -1)
//	if !layer.AssertDims(batchSize, -1) {  // 2D tensor, with batch size as the leading dimension.
//	    return nil
//	}
//
// ```
func (n *Node) AssertDims(dimensions ...int) bool {
	if !n.Ok() {
		return false
	}
	g := n.Graph()
	if !g.Ok() {
		return false
	}

	err := shapes.CheckDims(n, dimensions...)
	if err != nil {
		g.SetError(errors.WithMessagef(err, "AssertDims(%v)", dimensions))
		return false
	}
	return true
}

// AssertRank checks whether the shape has the given rank.
//
// If the shape is not what was expected, it sets an error in the associated Graph and returns false.
// If the Graph is already in error state, it also returns false.
//
// It can be used in a similar fashion as AssertDims.
func (n *Node) AssertRank(rank int) bool {
	if !n.Ok() {
		return false
	}
	g := n.Graph()
	if !g.Ok() {
		return false
	}

	err := shapes.CheckRank(n, rank)
	if err != nil {
		g.SetError(errors.WithMessagef(err, "AssertRank(%d)", rank))
		return false
	}
	return true
}

// AssertScalar checks whether the shape is a scalar.
//
// If the shape is not what was expected, it sets an error in the associated Graph and returns false.
// If the Graph is already in error state, it also returns false.
//
// It can be used in a similar fashion as AssertDims.
func (n *Node) AssertScalar() bool {
	if !n.Ok() {
		return false
	}
	g := n.Graph()
	if !g.Ok() {
		return false
	}

	err := shapes.CheckScalar(n)
	if err != nil {
		g.SetError(errors.WithMessage(err, "AssertScalar()"))
		return false
	}
	return true
}
