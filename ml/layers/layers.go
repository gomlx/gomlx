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

// Package layers holds a collection of common modeling layers. It includes dense layer, convolutions (TODO),
// activation functions, dropout (TODO), etc.
//
// A small convention on naming: typically layers are nouns (like "Convolution", "Dense" (layer), "MultiHeadAttention"),
// while computations are usually verbs ("Convolve", "Reduce..", "Multiply (Mul)", etc.).
package layers

import (
	"cmp"
	"fmt"

	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	// ParamL2Regularization context hyperparameter defines the L2 regularization of kernels.
	// Each layer may decide independently to implement it or not.
	//
	// This is an alias to regularizers.ParamL2
	// Dense, DenseWithBias, FNN, kan and Convolution kernels look at this hyperparameter.
	// The value should be a float64.
	// The default is `0.0`.
	//
	// Deprecated: use regularizers.ParamL2
	ParamL2Regularization = "l2_regularization"

	// ParamDropoutRate context hyperparameter defines the amount of dropout applied when DropoutFromContext is used.
	// Should be a value from `0.0` to `1.0`, where 0 means no dropout, and 1 would drop everything out.
	//
	// It is only applied if `Context.IsTraining() == true`, that is, during evaluation/inference it is
	// ignored.
	//
	// The default is `0.0`, which means no dropout.
	ParamDropoutRate = "dropout_rate"

	// ParamDropPathProbability provides the probability of DropPathFromContext to drop paths.
	//
	// This is only applied if the model actually calls DropPathFromContext.
	//
	// Default is `0.0`, which means never do any DropPath.
	ParamDropPathProbability = "droppath_prob"
)

// DenseWithBias adds a single dense linear layer, a learnable linear transformation plus a bias term.
//
// It the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// See also FNN for a more configurable (including hidden layers) version.
func DenseWithBias(ctx *context.Context, input *Node, outputDimensions ...int) *Node {
	return Dense(ctx, input, true, outputDimensions...)
}

// Dense adds a single dense linear layer, a learnable linear transformation.
// Optionally, it can include a bias term.
//
// It automatically adds regularization to the weights (not to biases) configured in hyperparameters -- see regularizers.FromContext.
//
// It the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// See also FNN for a more configurable (including hidden layers) version.
func Dense(ctx *context.Context, input *Node, useBias bool, outputDimensions ...int) *Node {
	g := input.Graph()
	ctx = ctx.In("dense")
	regularizer := regularizers.FromContext(ctx)

	inputShape := input.Shape()
	inputRank := inputShape.Rank()
	if inputRank == 0 {
		Panicf("input for layers.Dense needs to have rank >= 1, got %s", input.Shape())
	}
	if len(outputDimensions) == 0 {
		Panicf("at least one outputDimension must be given for layers.Dense, got 0 -- use outputDims=[1] for a scalar output")
	}
	inputLastDimension := inputShape.Dimensions[inputShape.Rank()-1]

	// Linear transformation.
	weightsDims := make([]int, 1+len(outputDimensions))
	weightsDims[0] = inputLastDimension
	copy(weightsDims[1:], outputDimensions)
	weightsVar := ctx.VariableWithShape("weights", shapes.Make(inputShape.DType, weightsDims...))
	if regularizer != nil {
		// Only for the weights, not for the bias.
		regularizer(ctx, g, weightsVar)
	}
	weights := weightsVar.ValueGraph(g)
	var output *Node
	if inputRank <= 2 && len(outputDimensions) == 1 {
		// Vanilla version: input = [batch_size, feature_size], output = [batch_size, output_dim].
		output = Dot(input, weights)
	} else {
		// Einsum all batch dimensions:
		axis := 'a'
		var equationPrefix string
		for ii := 0; ii < inputRank-1; ii++ {
			equationPrefix += string(axis)
			axis += 1
		}
		featureAxis := axis
		axis += 1
		var outputSuffix string
		for ii := 0; ii < len(outputDimensions); ii++ {
			outputSuffix += string(axis)
			axis += 1
		}

		equationPrefix = fmt.Sprintf("%s%c,%c%s->%s%s", equationPrefix, featureAxis, featureAxis, outputSuffix, equationPrefix, outputSuffix)
		output = Einsum(equationPrefix, input, weights)
	}

	// Add bias: it takes no regularizer by default.
	if useBias {
		biasVar := ctx.VariableWithShape("biases", shapes.Make(inputShape.DType, outputDimensions...))
		bias := biasVar.ValueGraph(g)
		expandedBiasShape := output.Shape().Clone()
		for ii := range expandedBiasShape.Dimensions[:output.Rank()-len(outputDimensions)] {
			expandedBiasShape.Dimensions[ii] = 1
		}
		expandedBias := ReshapeWithShape(bias, expandedBiasShape)
		output = Add(output, expandedBias)
	}
	return output
}

// Embedding creates an embedding table with vocabSize elements (typically a vocabulary size)
// each of dimension values -- so a [vocabSize, dimension] variable table.
//
// It then converts each integer value of the input to an embedding of the given dimension size.
// The input must have an integer dtype, and the last dimension must be of size 1. If it's not
// of size one, an extra dimension is added to the end. All values of the input
// must smaller than vocabSize, otherwise it will fail -- no checking is explicitly made.
//
// indicesAreSorted should be set to true only if the input is guaranteed to be sorted (ascending).
// This allows some optimizations in some backends. The default is false (for compatibility),
// in future versions this argument will be made obligatory.
//
// The output has rank one larger than the input, with the last dimension the same as
// the embedding dimension.
func Embedding(ctx *context.Context, input *Node, dtype dtypes.DType, vocabSize, dimension int, indicesAreSorted ...bool) *Node {
	inputShape := input.Shape()
	if !inputShape.DType.IsInt() {
		Panicf("can only use Embedding on integer inputs, passed %s instead", input.Shape())
	}
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		// Add a last dimension of size 1, since we are pointing to a table that needs
		// and index of size 1.
		input = InsertAxes(input, -1)
	}
	embeddingTable := ctx.VariableWithShape("embeddings", shapes.Make(dtype, vocabSize, dimension))
	return Gather(embeddingTable.ValueGraph(input.Graph()), input, indicesAreSorted...)
}

// AssertQuantilesForPWLCalibrationValid validates that raw values for quantiles are ok to be used for
// PieceWiseLinearCalibration. It checks for:
//   - Enough data points.
//   - Monotonicity of data points: quantiles should always be increasing.
//
// Errors are reported back with `panic`.
func AssertQuantilesForPWLCalibrationValid[T cmp.Ordered](values []T) {
	if len(values) < 2 {
		Panicf("PieceWiseLinearCalibration requires at least 2 quantile values")
	}
	current := values[0]
	for ii, value := range values[1:] {
		if value <= current {
			Panicf("quantile %d (out of %d), valued %v, for PieceWiseLinearCalibration is out of order or repeated",
				ii, len(values), value)
		}
		current = value
	}
}

// PieceWiseLinearCalibration creates a piece-wise linear function from the input, splitting
// it in the given keypoints with outputs initialized
// with values from 0 to 1.
//
// The keypoints are typically quantiles of the input feature, starting with the minimum value
// and ending on the maximum. It must have rank-1 and be of the same DType as input.
// Its values must be ordered, and cannot be repeated (this may lead to NaNs). Consider using
// AssertQuantilesForPWLCalibrationValid on the quantiles.
//
// If outputTrainable is set to true, the outputs mapped to the keypoints are made trainable, and
// may change to values outside the range [0, 1].
//
// In any case, if the input is beyond the first or last keypoint, the output of the function
// will flatten, preventing any extrapolations (often they are bad in NN).
//
// This is a simpler version to the one described here:
// https://www.tensorflow.org/lattice/api_docs/python/tfl/layers/PWLCalibration
func PieceWiseLinearCalibration(ctx *context.Context, input, keypoints *Node, outputTrainable bool) *Node {
	g := input.Graph()
	ctx = ctx.In("piece_wise_linear")
	if !input.DType().IsFloat() {
		Panicf("PieceWiseLinearCalibration only accepts float inputs, but got %s", input.Shape())
	}
	if keypoints.Rank() != 1 || keypoints.Shape().Dimensions[0] < 2 {
		Panicf("PieceWiseLinearCalibration keypoints shape %q invalid, it must be rank-1 and at list size 2", keypoints.Shape())
	}
	if keypoints.DType() != input.DType() {
		Panicf("PieceWiseLinearCalibration keypoints DType %s != input's DType %s",
			keypoints.DType(), input.DType())
	}

	inputShape := input.Shape()
	dtype := inputShape.DType
	numKeypoints := keypoints.Shape().Dimensions[0]

	// Normalize input to rank-2: [batch_size, 1]
	numInputs := inputShape.Size()
	input2D := Reshape(input, numInputs, 1)
	_ = input2D

	// Initialize outputKeypoints uniformly.
	outputKeypoints := make([]float64, numKeypoints)
	for ii := range outputKeypoints {
		outputKeypoints[ii] = float64(ii) / float64(numKeypoints-1)
	}
	outKPValue := shapes.CastAsDType(outputKeypoints, dtype)
	var outputKeypointsNode *Node
	if outputTrainable {
		outputKeypointsNode = ctx.VariableWithValue("output_keypoints", outKPValue).ValueGraph(g)
	} else {
		outputKeypointsNode = Const(g, outKPValue)
	}
	_ = outputKeypointsNode

	// Calculate lengths of each linear piece: all in shape [1, numKeypoints-1]
	kpStarts := InsertAxes(Slice(keypoints, AxisRange(0, numKeypoints-1)), 0)
	kpEnds := InsertAxes(Slice(keypoints, AxisRange(1, numKeypoints)), 0)
	lengths := Sub(kpEnds, kpStarts)

	//
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)
	trimLeft := NonNegativeIndicator(Sub(input2D, kpStarts))
	trimRight := PositiveIndicator(Sub(kpEnds, input2D))
	trimRegions := Mul(trimLeft, trimRight)

	// Calculate "left" weights on each keypoint: left weight is the weight
	leftWeights := Div(Sub(input2D, kpStarts), lengths)
	leftWeights = Min(Max(leftWeights, zero), one)
	rightWeights := OneMinus(leftWeights)

	leftWeights = Mul(trimRegions, leftWeights)
	rightWeights = Mul(trimRegions, rightWeights)

	// left and right weights are shifted by one from one another. And we need to add the weights
	// on the edge keypoints.
	//leftEdge := InsertAxes(SliceLists(keypoints, []int{0}, []int{1}), 0)
	leftEdge := InsertAxes(Slice(keypoints, AxisRangeFromStart(1)), 0)
	leftEdge = NonNegativeIndicator(Sub(leftEdge, input2D))
	leftWeights = Concatenate([]*Node{leftEdge, leftWeights}, -1)

	rightEdge := InsertAxes(Slice(keypoints, AxisRangeToEnd(-1)), 0)
	rightEdge = NonNegativeIndicator(Sub(input2D, rightEdge))
	rightWeights = Concatenate([]*Node{rightWeights, rightEdge}, -1)

	weights := Add(leftWeights, rightWeights)

	// The calibrated value is the weighted sum of the output keypoints.
	calibrated := ReduceSum(Mul(weights, InsertAxes(outputKeypointsNode, 0)), 1)
	return ReshapeWithShape(calibrated, inputShape)
}

// PieceWiseLinearCalibrationCascaded is a similar implementation for PieceWiseLinearCalibration that
// is equally powerful (express the same functions) simpler (fewer ops) and faster, but is parametrizing
// differently (cascaded linear functions), and may have different learning characteristics when
// doing gradient descent.
func PieceWiseLinearCalibrationCascaded(ctx *context.Context, input, keypoints *Node, outputTrainable bool) *Node {
	g := input.Graph()
	ctx = ctx.In("piece_wise_linear")
	if !input.DType().IsFloat() {
		Panicf("PieceWiseLinearCalibration only accepts float inputs, but got %s", input.Shape())
	}
	if keypoints.Rank() != 1 || keypoints.Shape().Dimensions[0] < 2 {
		Panicf("PieceWiseLinearCalibration keypoints shape %q invalid, it must be rank-1 and at list size 2", keypoints.Shape())
	}
	if keypoints.DType() != input.DType() {
		Panicf("PieceWiseLinearCalibration keypoints DType %s != input's DType %s",
			keypoints.DType(), input.DType())
	}

	inputShape := input.Shape()
	dtype := inputShape.DType
	numKeypoints := keypoints.Shape().Dimensions[0]

	// Normalize input to rank-2: [batch_size, 1]
	numInputs := inputShape.Size()
	input2D := Reshape(input, numInputs, 1)
	_ = input2D

	// Initialize outputKeypoints uniformly.
	outputKeypoints := make([]float64, numKeypoints)
	cumulative := 0.0
	for ii := range outputKeypoints {
		target := float64(ii) / float64(numKeypoints-1)
		outputKeypoints[ii] = target - cumulative
		cumulative = target
	}
	outKPValue := shapes.CastAsDType(outputKeypoints, dtype)
	var outputKeypointsNode *Node
	if outputTrainable {
		outputKeypointsNode = ctx.VariableWithValue("output_keypoints", outKPValue).ValueGraph(g)
	} else {
		outputKeypointsNode = Const(g, outKPValue)
	}

	// Calculate lengths of each linear piece: all in shape [1, numKeypoints-1]
	kpStarts := InsertAxes(Slice(keypoints, AxisRangeFromStart(-1)), 0)
	//kpStarts := InsertAxes(SliceLists(keypoints, []int{0}, []int{numKeypoints - 1}), 0)
	kpEnds := InsertAxes(Slice(keypoints, AxisRangeToEnd(1)), 0)
	lengths := Sub(kpEnds, kpStarts)

	// weights so far applies to outputKeypoints[1:]: it is shaped [numInputs, numKeypoints-1]
	weights := Div(Sub(input2D, kpStarts), lengths)
	weights = Min(Max(weights, ZerosLike(weights)), OnesLike(weights))

	//  We need to concatenate a weight of 1 for keypoint[0] as bias.
	weights = Concatenate([]*Node{
		Ones(g, shapes.Make(weights.DType(), numInputs, 1)),
		weights}, -1) // Now weights has shape [numInputs, numKeypoints]

	// The calibrated value is the weighted sum of the output keypoints.
	calibrated := ReduceSum(Mul(weights, InsertAxes(outputKeypointsNode, 0)), 1)
	return ReshapeWithShape(calibrated, inputShape)
}

// Dropout randomly replace the input with zeros if ctx.IsTraining() is true. Otherwise,
// it's a no op (it returns input).
// If the input is float, it scales the output by 1/(1-dropoutRate) to preserve the mean of the values of the input.
func Dropout(ctx *context.Context, input *Node, dropoutRate *Node) *Node {
	return DropoutNormalize(ctx, input, dropoutRate, input.DType().IsFloat())
}

// DropoutStatic is the same as Dropout, but it takes the `dropoutRate` as a static value, given as a float64.
// If `dropoutRate <= 0` or it's not training, this is a no-op.
func DropoutStatic(ctx *context.Context, input *Node, dropoutRate float64) *Node {
	if dropoutRate <= 0 {
		return input
	}
	g := input.Graph()
	return Dropout(ctx, input, Scalar(g, dtypes.Float32, dropoutRate))
}

// DropoutNormalize randomly replace the input with zeros if ctx.IsTraining() is true. Otherwise,
// it's a no op (it returns input). If normalize is set, it scales the output by 1/(1-dropoutRate)
// to preserve the mean of the input values.
func DropoutNormalize(ctx *context.Context, input *Node, dropoutRate *Node, normalize bool) *Node {
	g := input.Graph()
	if !ctx.IsTraining(g) {
		return input
	}

	// Disable (by multiplying by 0) random entries.
	dtype := dropoutRate.DType()
	dims := input.Shape().Dimensions
	rnd := ctx.RandomUniform(g, shapes.Make(dtype, dims...))
	broadcastRate := BroadcastToDims(dropoutRate, dims...)
	result := Where(LessOrEqual(rnd, broadcastRate), ZerosLike(input), input)
	if normalize {
		// Normalize input values, so mean value remains constant.
		keepRate := ConvertDType(OneMinus(dropoutRate), input.DType())
		result = Div(result, keepRate)
	}
	return result
}

// DropoutFromContext applies a dropout configured in the context parameters keyed by [ParamDropoutRate].
//
// If it is 0.0 this is a no-op.
// If `Context.IsTraining() == false` this is also a no-op, so it doesn't impact evaluation or inference.
func DropoutFromContext(ctx *context.Context, x *Node) *Node {
	dropoutRate := context.GetParamOr(ctx, ParamDropoutRate, 0.0)
	if dropoutRate > 0 {
		// We apply edge dropout to the mask.
		g := x.Graph()
		normalize := x.DType().IsFloat()
		x = DropoutNormalize(ctx, x, Scalar(g, x.DType(), dropoutRate), normalize)
	}
	return x
}

// DropPath drops with a certain probability whole examples (paths). It assumes x is shaped [batchSize, ...],
// and each batchSize example is independently either fully "dropped" (set to zero) or not.
//
// This is commonly applied on residual models, with updates like `x = Add(residual, x)`, which can
// be replaced with `x = Add(residual, DropPath(x, dropProbability))`.
// Another usage example is dropping whole positional embedding.
//
// If it is not training or dropProbability is nil, it is a no-op.
//
// See also DropPathFromContext.
func DropPath(ctx *context.Context, x, dropProbability *Node) *Node {
	g := x.Graph()
	if !ctx.IsTraining(g) || dropProbability == nil {
		return x
	}
	maskShape := x.Shape().Clone()
	for ii := 1; ii < maskShape.Rank(); ii++ {
		maskShape.Dimensions[ii] = 1
	}
	return Mul(x, ctx.RandomBernoulli(OneMinus(dropProbability), maskShape))
}

// DropPathFromContext will execute DropPath if the hyperparameter ParamDropPathProb is set to a value > 0.
// If ParamDropPathProb is not set or if not training, this is a no-op.
func DropPathFromContext(ctx *context.Context, x *Node) *Node {
	g := x.Graph()
	if !ctx.IsTraining(g) {
		return x
	}
	dropPathProb := context.GetParamOr(ctx, ParamDropPathProbability, 0.0)
	if dropPathProb > 0 {
		// We apply edge dropout to the mask.
		g := x.Graph()
		x = DropPath(ctx, x, Scalar(g, x.DType(), dropPathProb))
	}
	return x
}

// AddL2RegularizationStatic is like AddL2Regularization, but takes the `amount` as a static Go float64 value.
//
// Deprecated: use package regularizers instead.
func AddL2RegularizationStatic(ctx *context.Context, amount float64, values ...*Node) {
	if len(values) == 0 {
		Panicf("no values given to AddL2RegularizationAsFloat")
	}
	g := values[0].Graph()
	amountNode := Scalar(g, values[0].DType(), amount)
	AddL2Regularization(ctx, amountNode, values...)
}

// AddL2Regularization calculates the L2 of the given values (typically variable nodes returned
// by context.Variable.ValueGraph()), scale by the given amount (typically a constant) and then
// train.AddLoss the resulting value, having the effect of regularizing the weights (variables).
//
// Deprecated: use package regularizers instead.
func AddL2Regularization(ctx *context.Context, amount *Node, values ...*Node) {
	if len(values) == 0 {
		Panicf("no values given to AddL2Regularization")
	}
	var loss *Node
	for _, v := range values {
		l2 := ReduceAllSum(Square(v))
		if loss == nil {
			loss = l2
		} else {
			loss = Add(loss, l2)
		}
	}
	loss = Mul(loss, amount)
	train.AddLoss(ctx, loss)
}

// Normalize shifts and scales the input such that the mean becomes zero and the variance one.
// It calculates `(x - mean(x)) / (sigma(x))`, where sigma is the standard deviation.
//
// The parameter `independentAxes` list axes that should not be normalized together.
// A typical value is -1, the feature axis (last axis), so that each feature gets its own normalization.
func Normalize(x *Node, independentAxes ...int) *Node {
	if len(independentAxes) >= x.Rank() {
		return x
	}

	mapIndependentAxes := make([]bool, x.Rank())
	for _, axis := range independentAxes {
		mapIndependentAxes[axis] = true
	}
	reduceAxes := make([]int, x.Rank()-len(independentAxes))
	for axis, independent := range mapIndependentAxes {
		if !independent {
			reduceAxes = append(reduceAxes, axis)
		}
	}

	mean := ReduceAndKeep(x, ReduceMean, reduceAxes...)
	normalized := Sub(x, mean)
	variance := ReduceAndKeep(Square(normalized), ReduceSum, reduceAxes...)
	stdDev := Sqrt(variance)
	normalized = Div(normalized, stdDev)
	return normalized
}
