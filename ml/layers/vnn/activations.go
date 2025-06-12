package vnn

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
)

const (
	// ParamReluShareNonLinearity hyperparameter provides the default value for ReluConfig.ShareNonLinearity().
	// It defaults to false.
	ParamReluShareNonLinearity = "vnn_relu_share_non_linearity"

	// ParamReluNegativeSlope hyperparameter provides the default value for the Relu negative slope (the "leak").
	// It defaults to 0.2.
	ParamReluNegativeSlope = "vnn_relu_negative_slope"
)

type ReluConfig struct {
	ctx               *context.Context
	operand           *Node
	shareNonLinearity bool
	negativeSlope     float64
}

// Relu perform an equivariant to rotation (SO(3)) relu. It involves learning a
// projection (also equivariant), hence it needs the context to store the variable.
//
// The Relu forces the initializer of the variable to be random: a zero value here
// would lead to NaN.
//
// This returns a configuration object. Call ReluConfig.Done when finished and it
// will compute the Relu of the operand as specified.
func Relu(ctx *context.Context, operand *Node) *ReluConfig {
	if operand.Rank() < 2 {
		exceptions.Panicf("Relu requires at least two axes, got operand.shape=%s", operand.Shape())
	}
	if operand.Shape().Dim(-1) != 3 {
		exceptions.Panicf("Relu requires that the last axis of operand has dimension 3, got operand.shape=%s", operand.Shape())
	}
	if operand.Shape().Dim(-2) < 2 {
		exceptions.Panicf("Relu requires at least two vectors (channels > 2) as operand, got operand.shape=%s", operand.Shape())
	}
	return &ReluConfig{
		ctx:               ctx,
		operand:           operand,
		shareNonLinearity: context.GetParamOr(ctx, ParamReluShareNonLinearity, false),
		negativeSlope:     context.GetParamOr(ctx, ParamReluNegativeSlope, 0.2),
	}
}

// ShareNonLinearity makes the relu to share the direction of the non-linearity to all features.
func (c *ReluConfig) ShareNonLinearity(enabled bool) *ReluConfig {
	c.shareNonLinearity = enabled
	return c
}

// NegativeSlope sets the amount of "leakage" to allow on the negative side of the Relu non-linearity.
// Any value <= 0 implies no leakage.
//
// Values should be < 1, the default is 0.2.
func (c *ReluConfig) NegativeSlope(negativeSlope float64) *ReluConfig {
	c.negativeSlope = negativeSlope
	return c
}

// Done applies the Relu layer as configured.
func (c *ReluConfig) Done() *Node {
	// We force a random initializer: we don't want a Zero initializer here.
	ctx := c.ctx
	operand := c.operand
	g := operand.Graph()
	dtype := operand.DType()
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)

	numChannels := operand.Shape().Dim(-2)
	vecDim := operand.Shape().Dim(-1) // 3
	ctx = ctx.In("relu").WithInitializer(initializers.RandomNormalFn(ctx, 0.1))

	// Normalize the operand to rank-3: [batchSize, inputChannels, 3 (each vector dimension)]
	originalShape := operand.Shape()
	operand = Reshape(operand, -1, numChannels, vecDim)

	// Calculate k: the projected direction on which we are doing the Relu.
	numProjections := numChannels
	if c.shareNonLinearity {
		numProjections = 1
	}
	projection := ctx.VariableWithShape("projection", shapes.Make(dtype, numProjections, numChannels)).ValueGraph(g)
	k := Einsum("bcv,pc->bpv", operand, projection)
	kL2NormSq := L2NormSquare(k, -1)
	kL2NormSq = Where(Equal(kL2NormSq, zero), one, kL2NormSq)

	// Dot product of operand and k will tell us the inputs that will have to be adjusted.
	dotXK := ReduceAndKeep(Mul(operand, k), ReduceSum, -1)
	adjustMask := GreaterOrEqual(dotXK, ScalarZero(g, dtype))
	adjustMask = BroadcastToShape(adjustMask, operand.Shape())
	operandMinusK := Sub(operand, Div(Mul(dotXK, k), kL2NormSq))
	if c.negativeSlope > 0 {
		operandMinusK = Add(
			MulScalar(operand, c.negativeSlope),
			MulScalar(operandMinusK, 1-c.negativeSlope))
	}

	// Apply the non-linearity where operand and k don't align.
	operand = Where(adjustMask, operand, operandMinusK)

	// Denormalize X shape back to the original shape.
	operand = Reshape(operand, originalShape.Dimensions...)
	return operand
}

// ReluFromContext applies Relu with the default parameters used from the context.
func ReluFromContext(ctx *context.Context, operand *Node) *Node {
	return Relu(ctx, operand).Done()
}
