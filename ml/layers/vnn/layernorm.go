package vnn

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
)

// LayerNormalization perform an equivariant (SO(3)) layer normalization on vectors.
//
// The value epsilon is used in the denominator, and a value of 1e-3 is a good default.
func LayerNormalization(x *Node, epsilon float64) *Node {
	if x.Rank() < 2 {
		exceptions.Panicf("Relu requires at least two inputs, got x.shape=%s", x.Shape())
	}
	if x.Shape().Dim(-1) != 3 {
		exceptions.Panicf("Relu requires that the last axis of x has dimension 3, got x.shape=%s", x.Shape())
	}
	if x.Shape().Size() < 2*3 {
		exceptions.Panicf("Relu requires at least two vectors as input, got x.shape=%s", x.Shape())
	}

	// Normalize X to rank-3: [batchSize, inputChannels, 3 (vector size)]
	originalShape := x.Shape()
	x = Reshape(x, -1, x.Shape().Dim(-2), x.Shape().Dim(-1))

	// Find mean vectors per example.
	xMean := ReduceAndKeep(x, ReduceMean, 1)
	variance := ReduceAndKeep(L2NormSquare(Sub(x, xMean), -1), ReduceMean, 1)

	// Shift x such that the new mean is the origin.
	x = Sub(x, xMean)
	x = Div(x, Sqrt(AddScalar(variance, epsilon)))

	// Denormalize X shape back to the original shape.
	x = Reshape(x, originalShape.Dimensions...)
	return x
}
