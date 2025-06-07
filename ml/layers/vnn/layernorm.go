package vnn

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
)

// LayerNormalization perform an equivariant (SO(3)) layer normalization on vectors.
//
// The value epsilon is used in the denominator, and a value of 1e-5 is a good default.
func LayerNormalization(operand *Node, epsilon float64) *Node {
	if epsilon <= 0 {
		exceptions.Panicf("vnn.LayerNormalization: epsilon must be > 0, got epsilon=%.2g", epsilon)
	}
	if operand.Rank() < 2 {
		exceptions.Panicf("Relu requires at least two inputs, got operand.shape=%s", operand.Shape())
	}
	if operand.Shape().Dim(-1) != 3 {
		exceptions.Panicf("Relu requires that the last axis of operand has dimension 3, got operand.shape=%s", operand.Shape())
	}
	if operand.Shape().Size() < 2*3 {
		exceptions.Panicf("Relu requires at least two vectors as operand, got operand.shape=%s", operand.Shape())
	}

	// Normalize X to rank-3: [batchSize, inputChannels, 3 (vector size)]
	originalShape := operand.Shape()
	vecDim := operand.Shape().Dim(-1) // 3
	numChannels := operand.Shape().Dim(-2)
	operand = Reshape(operand, -1, numChannels, vecDim)

	// Find mean and variance of vectors over the channels.
	xMean := ReduceAndKeep(operand, ReduceMean, 1)
	variance := L2NormSquare(Sub(operand, xMean), -1)
	variance = ReduceAndKeep(variance, ReduceMean, 1)
	//variance.SetLoggedf("vnn.LayerNorm: variance")

	// Shift operand such that the new mean is the origin.
	operand = Sub(operand, xMean)
	denominator := Sqrt(AddScalar(variance, epsilon))
	operand = Div(operand, denominator)

	// Denormalize X shape back to the original shape.
	operand = Reshape(operand, originalShape.Dimensions...)
	return operand
}
