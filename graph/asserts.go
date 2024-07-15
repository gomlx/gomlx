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
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/pkg/errors"
)

// This file implements various asserts (checks) that can be done on the Node.
// They are derived from the asserts in the shapes package.

// AssertDims checks whether the outputShapes has the given dimensions and rank.
// A value of -1 in dimensions means it can take any value and is not checked.
//
// If the outputShapes is not what was expected, it panics with an error message.
//
// This often serves as documentation for the code when implementing some complex computational
// graphs.
// This allows the reader of the code to corroborate what is the expected outputShapes of a node.
//
// Example:
//
//	batch_size := inputNodes[0].Shape().Dimensions[0]
//	…
//	layer := Concatenate(allEmbeddings, -1)
//	layer.AssertDims(batchSize, -1) // 2D tensor, with batch size as the leading dimension.
func (n *Node) AssertDims(dimensions ...int) {
	n.AssertValid()
	err := shapes.CheckDims(n, dimensions...)
	if err != nil {
		panic(errors.WithMessagef(err, "AssertDims(%v)", dimensions))
	}
}

// AssertRank checks whether the outputShapes has the given rank.
//
// If the rank is not what was expected, it panics with an error message.
//
// This often serves as documentation for the code when implementing some complex computational
// graphs.
// This allows the reader of the code to corroborate what is the expected outputShapes of a node.
//
// It can be used in a similar fashion as AssertDims.
func (n *Node) AssertRank(rank int) {
	n.AssertValid()
	err := shapes.CheckRank(n, rank)
	if err != nil {
		panic(errors.WithMessagef(err, "AssertRank(%d)", rank))
	}
}

// AssertScalar checks whether the outputShapes is a scalar.
//
// If the rank is not what was expected, it panics with an error message.
//
// It can be used in a similar fashion as AssertDims.
func (n *Node) AssertScalar() {
	n.AssertValid()
	err := shapes.CheckScalar(n)
	if err != nil {
		panic(errors.WithMessage(err, "AssertScalar()"))
	}
}
