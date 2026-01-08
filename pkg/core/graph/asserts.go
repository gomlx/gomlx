// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// This file implements various asserts (checks) that can be done on the Node.
// They are derived from the asserts in the shapes package.

// AssertDims checks whether the shape has the given dimensions and rank.
// A value of -1 in dimensions means it can take any value and is not checked.
//
// If the shape is not what was expected, it panics with an error message.
//
// This often serves as documentation for the code when implementing some complex computational
// graphs.
// This allows the reader of the code to corroborate what is the expected shape of a node.
//
// Example:
//
//	batch_size := inputNodes[0].Shape().Dimensions[0]
//	…
//	layer := Concatenate(allEmbeddings, -1)
//	layer.AssertDims(batchSize, -1) // 2D tensor, with batch size as the leading dimension.
func (n *Node) AssertDims(dimensions ...int) {
	n.AssertValid()
	if n.NumOutputs() != 1 {
		exceptions.Panicf("node has %d outputs, cannot AssertDims(%v)", n.NumOutputs(), dimensions)
	}
	err := shapes.CheckDims(n, dimensions...)
	if err != nil {
		panic(errors.WithMessagef(err, "AssertDims(%v)", dimensions))
	}
}

// AssertRank checks whether the shape has the given rank.
//
// If the rank is not what was expected, it panics with an error message.
//
// This often serves as documentation for the code when implementing some complex computational
// graphs.
// This allows the reader of the code to corroborate what is the expected shape of a node.
//
// It can be used in a similar fashion as AssertDims.
func (n *Node) AssertRank(rank int) {
	n.AssertValid()
	err := shapes.CheckRank(n, rank)
	if err != nil {
		panic(errors.WithMessagef(err, "AssertRank(%d)", rank))
	}
}

// AssertScalar checks whether the shape is a scalar.
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
