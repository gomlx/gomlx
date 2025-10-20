package vnn

import (
	"slices"

	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// DropoutNormalize randomly replace the operand with zeros if ctx.IsTraining() is true. Otherwise,
// it's a no op (it returns operand). If normalize is set, it scales the output by 1/(1-dropoutRate)
// to preserve the mean of the operand values.
//
// It expects the operand to have shape [..., inputFeatures, 3], and the dropout happens on the inputFeatures.
func DropoutNormalize(ctx *context.Context, operand *Node, dropoutRate *Node, normalize bool) *Node {
	g := operand.Graph()
	if operand.Shape().Dim(-1) != 3 {
		exceptions.Panicf("vnn: the operand last dimensions must be 3 -- it works with 3D vectors only for now")
	}
	if operand.Rank() < 2 {
		exceptions.Panicf("vnn: operand must be rank at least 2 for Dropout, got operand.shape=%s", operand.Shape())
	}
	if !dropoutRate.IsScalar() {
		exceptions.Panicf("vnn: dropoutRate must be a scalar, got dropoutRate.shape=%s", dropoutRate.Shape())
	}
	if !ctx.IsTraining(g) {
		return operand
	}

	// Disable (by multiplying by 0) random entries.
	dtype := dropoutRate.DType()
	dims := slices.Clone(operand.Shape().Dimensions)
	dims[len(dims)-1] = 1
	rnd := ctx.RandomUniform(g, shapes.Make(dtype, dims...))
	result := Where(LessOrEqual(rnd, dropoutRate), ZerosLike(operand), operand)
	if normalize {
		// Normalize operand values, so mean value remains constant.
		keepRate := ConvertDType(OneMinus(dropoutRate), operand.DType())
		result = Div(result, keepRate)
	}
	return result
}

// RotateOnOrigin 3D elements arbitrarily around the 3 axes.
//
// It requires that the last axis of x to be of dimension 3.
// Typically, x will be shaped [batchSize, N, 3].
//
// The rotation angles roll, pitch and yaw (rotations on the axes x, y and z respectively) must either be nil
// (no rotation on that axis) or a scalar.
//
// TODO: allow the angles to be shaped like x, and have one rotation per 3d coordinates.
func RotateOnOrigin(x, roll, pitch, yaw *Node) *Node {
	g := x.Graph()
	dtype := x.DType()
	one := ScalarOne(g, dtype)
	zero := ScalarZero(g, dtype)
	if x.Shape().Dim(-1) != 3 {
		exceptions.Panicf("Rotate requires that the last axis of x has dimension 3, got x.shape=%s", x.Shape())
	}
	if roll != nil && !roll.IsScalar() {
		exceptions.Panicf("Rotate requires roll (the x-axis rotation angle) be a scalar, got roll.shape=%s", roll.Shape())
	}
	if pitch != nil && !pitch.IsScalar() {
		exceptions.Panicf("Rotate requires pitch (the y-axis rotation angle) be a scalar, got pitch.shape=%s", pitch.Shape())
	}
	if yaw != nil && !yaw.IsScalar() {
		exceptions.Panicf("Rotate requires yaw (the z-axis rotation angle) be a scalar, got yaw.shape=%s", yaw.Shape())
	}
	if roll == nil && pitch == nil && yaw == nil {
		// No rotation, return x.
		return x
	}

	// Normalize X: reshape it [N, 3]
	normX := Reshape(x, -1, 3)

	// Calculate rotation matrices for each axis.
	var R *Node
	if roll != nil {
		rX := Stack(
			[]*Node{one, zero, zero,
				zero, Cos(roll), Neg(Sin(roll)),
				zero, Sin(roll), Cos(roll),
			}, 0)
		rX = Reshape(rX, 3, 3)
		R = rX
	}
	if pitch != nil {
		rY := Stack(
			[]*Node{Cos(pitch), zero, Sin(pitch),
				zero, one, zero,
				Neg(Sin(pitch)), zero, Cos(pitch),
			}, 0)
		rY = Reshape(rY, 3, 3)
		if R == nil {
			R = rY
		} else {
			R = Dot(R, rY)
		}
	}
	if yaw != nil {
		rZ := Stack(
			[]*Node{Cos(yaw), Neg(Sin(yaw)), zero,
				Sin(yaw), Cos(yaw), zero,
				zero, zero, one,
			}, 0)
		rZ = Reshape(rZ, 3, 3)
		if R == nil {
			R = rZ
		} else {
			R = Dot(R, rZ)
		}
	}
	rotated := Dot(normX, R)
	return Reshape(rotated, x.Shape().Dimensions...)
}
