package vnn

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
)

// Relu perform an equivariant to rotation (SO(3)) relu. It involves learning a
// projection (also equivariant), hence it needs the context to store the variable.
//
// The Relu forces the initializer of the variable to be random: a zero value here
// would lead to NaN.
func Relu(ctx *context.Context, x *Node) *Node {
	if x.Rank() < 2 {
		exceptions.Panicf("Relu requires at least two inputs, got x.shape=%s", x.Shape())
	}
	if x.Shape().Dim(-1) != 3 {
		exceptions.Panicf("Relu requires that the last axis of x has dimension 3, got x.shape=%s", x.Shape())
	}
	if x.Shape().Size() < 2*3 {
		exceptions.Panicf("Relu requires at least two vectors as input, got x.shape=%s", x.Shape())
	}

	// We force a random initializer: we don't want a Zero initializer here.
	g := x.Graph()
	dtype := x.DType()
	numChannels := x.Shape().Dim(-2)
	ctx = ctx.In("relu").WithInitializer(initializers.RandomNormalFn(ctx, 0.1))

	// Normalize X to rank-3: [batchSize, inputChannels, 3 (vector size)]
	originalShape := x.Shape()
	x = Reshape(x, -1, x.Shape().Dim(-2), x.Shape().Dim(-1))

	// Calculate k: the projected direction on which we are doing the Relu.
	projection := ctx.VariableWithShape("projection", shapes.Make(dtype, numChannels)).ValueGraph(g)
	nonZeroProjection := Where(
		ReduceLogicalAnd(Equal(projection, ZerosLike(projection))),
		MulScalar(OnesLike(projection), 1.0/float64(numChannels)), // Converts a (0,... 0) to (1/C, ... 1/C) to avoid NaN.
		projection)
	k := Einsum("bcv,c->bv", x, nonZeroProjection)
	k = ExpandAxes(k, -2) // Re-introduce the channels' axis, with dim=1 -> [batchSize, 1, 3]
	kIsZero := ReduceLogicalAnd(Equal(k, ZerosLike(k)))
	nonZeroK := Where(
		kIsZero,
		OnesLike(k), // Converts a (0, 0, 0) to (1, 1, 1) to avoid NaN.
		k)
	// unitK is K rescaled to a unit vector.
	unitK := Div(nonZeroK, L2Norm(nonZeroK, -1))

	// Dot product of x and k will tell us the inputs that will have to be adjusted.
	dotXK := ReduceAndKeep(Mul(x, unitK), ReduceSum, -1)
	xMinusK := Sub(x, Mul(dotXK, unitK))
	adjustMask := GreaterOrEqual(dotXK, ScalarZero(g, dtype))
	adjustMask = BroadcastToShape(adjustMask, x.Shape())
	xMinusK = Where(adjustMask, x, xMinusK)

	// Safety: if k is zero everywhere, we don't do anything.
	x = Where(kIsZero, x, xMinusK)

	// Denormalize X shape back to the original shape.
	x = Reshape(x, originalShape.Dimensions...)
	return x
}
