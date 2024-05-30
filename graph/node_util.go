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
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

// This file contains derived practical calculations that often used.

// Scalar returns a constant scalar with the given value.
func Scalar(g *Graph, dtype shapes.DType, value float64) *Node {
	return g.getScalarConst(dtype, value)
}

// FillScalar creates a Node with a value with the given shape, filled with the given value.
// It's implemented indirectly using other nodes.
func FillScalar(g *Graph, shape shapes.Shape, value float64) *Node {
	return BroadcastPrefix(Scalar(g, shape.DType, value), shape.Dimensions)
}

// ScalarZero returns a scalar constant 0 for the given DType.
func ScalarZero(g *Graph, dtype shapes.DType) *Node {
	return Scalar(g, dtype, 0)
}

// ScalarOne returns a scalar constant 1 for the given DType.
func ScalarOne(g *Graph, dtype shapes.DType) *Node {
	return Scalar(g, dtype, 1)
}

// MulScalar converts scalar to a constant with x's DType and returns `x * scalar`
// with proper broadcasting.
func MulScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Mul(x, Scalar(g, x.DType(), scalar))
}

// DivScalar converts scalar to a constant with x's DType and returns `x / scalar`
// with proper broadcasting.
//
// For float DType's, DivScalar instead uses MulScalar(x, 1/scalar).
func DivScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	if scalar == 0 {
		Panicf("division by zero in DivScalar")
	}
	if x.DType().IsFloat() {
		// Multiply by 1/scalar instead:
		return MulScalar(x, 1.0/float64(scalar))
	}
	return Div(x, Scalar(g, x.DType(), scalar))
}

// PowScalar converts scalar to a constant with x's DType and returns `Pow(x, scalar)` (or `x ** scalar`)
// with proper broadcasting.
func PowScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Pow(x, Scalar(g, x.DType(), scalar))
}

// Square returns x^2 point-wise. Same as `Mul(x, x)`.
func Square(x *Node) *Node {
	return Mul(x, x)
}

// AddScalar converts scalar to a constant with x's DType and returns `x + scalar`
// with proper broadcasting.
func AddScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Add(x, Scalar(g, x.DType(), scalar))
}

// ModScalar converts scalar to a constant with x's DType and returns `x % scalar`
// with proper broadcasting.
func ModScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Mod(x, Scalar(g, x.DType(), scalar))
}

// MaxScalar converts scalar to a constant with x's DType and returns element-wise `Max(x, scalar)`.
func MaxScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Max(x, Scalar(g, x.DType(), scalar))
}

// MinScalar converts scalar to a constant with x's DType and returns element-wise `Min(x, scalar)`.
func MinScalar(x *Node, scalar float64) *Node {
	g := x.Graph()
	return Min(x, Scalar(g, x.DType(), scalar))
}

func lowestForDType(g *Graph, dtype shapes.DType) *Node {
	return Const(g, shapes.LowestValueForDType(dtype))
}

// OnesLike returns a tensor with the same shape of x, filled with 1's.
func OnesLike(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Ones(g, x.Shape())
}

// Ones creates a computation with the same shape as the input, but with the value 1.
// It's implemented indirectly using other nodes.
func Ones(g *Graph, shape shapes.Shape) *Node {
	g.AssertValid()
	scalar := ScalarOne(g, shape.DType)
	if scalar == nil {
		return nil
	}
	return BroadcastPrefix(scalar, shape.Dimensions)
}

// ZerosLike returns a tensor with the same shape of x, filled with 0's.
func ZerosLike(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Zeros(g, x.Shape())
}

// Zeros creates a computation with the same shape as the input, but with the value 0.
// It's implemented indirectly using other nodes.
func Zeros(g *Graph, shape shapes.Shape) *Node {
	g.AssertValid()
	return BroadcastPrefix(ScalarZero(g, shape.DType), shape.Dimensions)
}

// OneMinus returns (1-x).
func OneMinus(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Sub(ScalarOne(g, x.DType()), x)
}

// MinusOne returns (x-1).
func MinusOne(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Sub(x, ScalarOne(g, x.DType()))
}

// OnePlus returns (1+x).
func OnePlus(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Add(ScalarOne(g, x.DType()), x)
}

// Inverse returns (1/x), the multiplicative inverse. Also known as the reciprocal.
func Inverse(x *Node) *Node {
	g := validateGraphFromInputs(x)
	return Div(ScalarOne(g, x.DType()), x)
}

// SignPlusOrMinus return +1 or -1 whether x >= 0 or x < 0. It's similar to Sign, but
// where 0s are considered positive.
func SignPlusOrMinus(x *Node) *Node {
	g := validateGraphFromInputs(x)
	half := Scalar(g, x.DType(), 0.5)
	return Sign(Add(Sign(x), half))
}

// PositiveIndicator returns 1 where x >= 0, 0 otherwise. See also StrictlyPositiveIndicator.
// E.g: PositiveIndicator({1.0, 0.0001, 0, -0.2, -3.0}) -> [1, 1, 1, 0, 0], with the same shape/dtype as x.
func PositiveIndicator(x *Node) *Node {
	g := validateGraphFromInputs(x)
	one := ScalarOne(g, x.DType())
	return Sign(Add(Sign(x), one))
}

// StrictlyPositiveIndicator returns 1 where x > 0, 0 otherwise.
// E.g: StrictlyPositiveIndicator({1.0, 0.0001, 0, -0.2, -3.0}) -> [1, 1, 0, 0, 0], with the same shape/dtype as x.
func StrictlyPositiveIndicator(x *Node) *Node {
	g := validateGraphFromInputs(x)
	one := ScalarOne(g, x.DType())
	return Add(Sign(Sub(Sign(x), one)), one)
}

// Clip is a shortcut to `Min(max, Max(x, min))`, which returns the values of x clipped between
// min and max.
func Clip(x, min, max *Node) *Node {
	return Min(max, Max(x, min))
}

// ClipScalar is a shortcut to `Min(max, Max(x, min))`, which returns the values of x clipped between
// min and max. The values min and max are given as scalar values -- the float64 is converted to the
// `DType` of x.
func ClipScalar(x *Node, min, max float64) *Node {
	return MinScalar(MaxScalar(x, min), max)
}

// OneHot converts an integer numbers representing indices to it's "one-hot" representation, that is an expanded
// tensor with the indices position set to 1, and the other positions set to 0. The returned tensor has one extra
// dimension at the end.
// For example `OneHot([][]INT64{1, 0, 3}, 4, types.Float32)` returns  `[][]F32{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}}`
// TODO: implement with Select once it's implemented, since it's likely going to be faster (TensorFlow uses that).
func OneHot(indices *Node, depth int, dtype shapes.DType) *Node {
	g := indices.Graph()
	if !indices.shape.DType.IsInt() {
		Panicf("invalid indices dtype (%s), it must be integer", indices.shape.DType)
	}

	// Add an expanded dimension at the end, which will contain the one-hot representation.
	indices = ExpandDims(indices, -1)

	// The target shape will expand the indices dimension (last/innermost one) to depth.
	targetShape := indices.shape.Copy()
	targetShape.Dimensions[targetShape.Rank()-1] = depth
	targetShape.DType = dtype

	// scatterIndices must create the full indices (for all dimensions, not only for the last being set to 1).
	// * Create one sub-id per leading dimension on indices.
	// * ConcatenateDimensions them, along with the indices themselves (the last dimension).
	// * Flatten them.
	parts := make([]*Node, 0, indices.shape.Rank())
	for dimIdx := 0; dimIdx < indices.shape.Rank()-1; dimIdx++ {
		parts = append(parts, Iota(g, indices.shape, dimIdx))
	}
	parts = append(parts, indices)
	scatterIndices := Concatenate(parts, -1)
	scatterIndices = Reshape(scatterIndices, indices.shape.Size(), indices.shape.Rank())
	ones := Ones(g, shapes.Make(dtype, indices.shape.Size()))
	return Scatter(scatterIndices, ones, targetShape)

	//// ones will have the same shape as the sindices, but with the dtype set to the desired type.
	//onesShape := indices.shape.Copy()
	//onesShape.DType = dtype
	//ones := Ones(g, onesShape)
	//return Scatter(indices, ones, targetShape)
}

// ReduceAndKeep applies the given reduction function but regenerate the reduced dimensions with size 1.
func ReduceAndKeep(x *Node, reduceFn func(x *Node, reduceAxes ...int) *Node, reduceAxes ...int) *Node {
	_ = validateGraphFromInputs(x)
	rank := x.Rank()
	reduceAxes = convertNegativeAxesAndSort(rank, reduceAxes)
	reduced := reduceFn(x, reduceAxes...)
	shapeWithRecoveredDims := x.Shape().Copy()
	for ii := 0; ii < rank && len(reduceAxes) > 0; ii++ {
		if ii == reduceAxes[0] {
			shapeWithRecoveredDims.Dimensions[ii] = 1
			reduceAxes = reduceAxes[1:]
		}
	}
	return Reshape(reduced, shapeWithRecoveredDims.Dimensions...)
}

// ReduceAndKeepMasked applies the given masked reduction function but regenerates the reduced
// dimensions with size 1.
func ReduceAndKeepMasked(x, mask *Node, reduceFn func(x, mask *Node, reduceAxes ...int) *Node, reduceAxes ...int) *Node {
	_ = validateGraphFromInputs(x)
	rank := x.Rank()
	reduceAxes = convertNegativeAxesAndSort(rank, reduceAxes)
	reduced := reduceFn(x, mask, reduceAxes...)
	shapeWithRecoveredDims := x.Shape().Copy()
	for ii := 0; ii < rank && len(reduceAxes) > 0; ii++ {
		if ii == reduceAxes[0] {
			shapeWithRecoveredDims.Dimensions[ii] = 1
			reduceAxes = reduceAxes[1:]
		}
	}
	return Reshape(reduced, shapeWithRecoveredDims.Dimensions...)
}

// Softmax computes softmax activations. It's the equivalent to
// ```
//
//	Exp(logits) / ExpandDims(ReduceSum(Exp(logits), -1), -1)
//
// ```
//
// But implemented in a numerical stable way.
//
// The list axes defines which axes is it supposed to run the softmax over
// (the axes that will be summed over). If no axes are given, it is assumed to
// be [-1], meaning, the last axes.
func Softmax(logits *Node, axes ...int) *Node {
	_ = validateGraphFromInputs(logits)
	if !logits.shape.DType.IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.shape.DType)
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	max := StopGradient(ReduceAndKeep(logits, ReduceMax, axes...))
	normalizedLogits := Sub(logits, max)
	numerator := Exp(normalizedLogits)
	denominator := ReduceAndKeep(numerator, ReduceSum, axes...)
	return Div(numerator, denominator)
}

// MaskedSoftmax computes softmax activations. It's the equivalent to
// ```
//
//	Exp(logits) / ExpandDims(ReduceSum(Exp(logits), -1), -1)
//
// ```
//
// But implemented in a numerical stable way.
//
// The list axes defines which axes is it supposed to run the softmax over
// (the axes that will be summed over). If no axes are given, it is assumed to
// be [-1], meaning, the last axes.
//
// It ignores values for which the corresponding mask is false, and will return 0 for
// those fields. mask and logits must have the same shape.
func MaskedSoftmax(logits, mask *Node, axes ...int) *Node {
	_ = validateGraphFromInputs(logits)
	if !logits.shape.DType.IsFloat() {
		Panicf("invalid logits dtype (%s), it must be float", logits.shape.DType)
	}
	if len(axes) == 0 {
		axes = []int{-1}
	}
	max := StopGradient(ReduceAndKeepMasked(logits, mask, ReduceMaskedMax, axes...))
	zeros := ZerosLike(logits)
	normalizedLogits := Sub(logits, max)
	normalizedLogits = Where(mask, normalizedLogits, zeros)
	numerator := Exp(normalizedLogits)
	numerator = Where(mask, numerator, zeros)
	// Apply mask on numerator, setting softmax to zero where masked.
	denominator := ReduceAndKeep(numerator, ReduceSum, axes...)
	result := Div(numerator, denominator)
	result = Where(mask, result, zeros)
	return result
}

// L1Norm returns the L1 norm (same as Manhattan length) of the last axis of x.
// The returned value has the same rank, but the last axes will have dimension 1.
//
// If no axes are given, it returns a scalar.
// Otherwise, the returned value has the same rank as `x`, but the reduce axes will have dimension 1.
func L1Norm(x *Node, reduceAxes ...int) *Node {
	if len(reduceAxes) == 0 {
		return ReduceAllSum(Abs(x))
	}
	return ReduceAndKeep(Abs(x), ReduceSum, -1)
}

// L2NormSquare returns the L2 norm square (same as square of the Euclidean length) over the given axes
// of x (defaults to all).
// Same as `\Sum_{reduceAxes}{x_i^2}`.
//
// If no axes are given, it returns a scalar.
// Otherwise, the returned value has the same rank as `x`, but the reduce axes will have dimension 1.
func L2NormSquare(x *Node, reduceAxes ...int) *Node {
	if len(reduceAxes) == 0 {
		return ReduceAllSum(Square(x))
	}
	return ReduceAndKeep(Square(x), ReduceSum, reduceAxes...)
}

// L2Norm returns the L2 norm (same as Euclidean length) over the given axes of x (defaults to all), given by Sqrt(\Sum{x_i^2}).
//
// If no axes are given, it returns a scalar.
// Otherwise, the returned value has the same rank as `x`, but the reduce axes will have dimension 1.
func L2Norm(x *Node, reduceAxes ...int) *Node {
	return Sqrt(L2NormSquare(x, reduceAxes...))
}

// L2Normalize returns `x/L2Norm(x)` on the given reduce axes, making the last axis a unit-length vector.
//
// It will return `inf` for values of x that are zero-length.
// See L2NormalizeWithEpsilon for a version that adds an epsilon to the denominator to avoid that.
func L2Normalize(x *Node, reduceAxis int, moreReduceAxes ...int) *Node {
	reduceAxes := make([]int, 1, 1+len(moreReduceAxes))
	reduceAxes[0] = reduceAxis
	reduceAxes = append(reduceAxes, moreReduceAxes...)
	return Div(x, L2Norm(x, reduceAxes...))
}

// L2NormalizeWithEpsilon returns `x/(L2Norm(x)+epsilon)` on the last axis, making the last axis a unit-length vector.
func L2NormalizeWithEpsilon(x *Node, epsilon float64, reduceAxis int, moreReduceAxes ...int) *Node {
	reduceAxes := make([]int, 1, 1+len(moreReduceAxes))
	reduceAxes[0] = reduceAxis
	reduceAxes = append(reduceAxes, moreReduceAxes...)
	return Div(x, AddScalar(L2Norm(x, reduceAxes...), epsilon))
}

// LowerTriangular returns a lower-triangular boolean square matrix of shape `[dim, dim]`.
//
// This can be combined with `Where` to select values of any arbitrary other matrix.
func LowerTriangular(g *Graph, dim int) *Node {
	shapeInt := shapes.Make(shapes.I64, dim, dim)
	rows := Iota(g, shapeInt, 0)
	cols := Iota(g, shapeInt, 1)
	return LessOrEqual(cols, rows)
}

// UpperTriangular returns a upper-triangular boolean square matrix of shape `[dim, dim]`.
//
// This can be combined with `Where` to select values of any arbitrary other matrix.
func UpperTriangular(g *Graph, dim int) *Node {
	shapeInt := shapes.Make(shapes.I64, dim, dim)
	rows := Iota(g, shapeInt, 0)
	cols := Iota(g, shapeInt, 1)
	return GreaterOrEqual(cols, rows)
}

// Diagonal returns a diagonal boolean square matrix of shape `[dim, dim]`.
//
// This can be combined with `Where` to select values of any arbitrary other matrix.
func Diagonal(g *Graph, dim int) *Node {
	shapeInt := shapes.Make(shapes.I64, dim, dim)
	rows := Iota(g, shapeInt, 0)
	cols := Iota(g, shapeInt, 1)
	return Equal(cols, rows)
}

// DiagonalWithValue returns a diagonal matrix of shape `[dim, dim]` with
// scalar in the diagonal and zero elsewhere. Set scalar to `ScalarOne()`
// and you get an identity matrix.
func DiagonalWithValue(scalar *Node, dim int) *Node {
	g := scalar.Graph()
	matrix := BroadcastPrefix(scalar, []int{dim, dim})
	return Where(Diagonal(g, dim), matrix, ZerosLike(matrix))
}
