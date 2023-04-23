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
	"fmt"
	"golang.org/x/exp/constraints"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
)

const (
	// L2RegularizationKey is the key to a context.Context.Params that defines the default L2 regularization
	// of kernels. Each layer may decide independently to implement it or not. DenseWithBias and Convolution kernels
	// look at this hyperparameter. The value should be a float64.
	L2RegularizationKey = "l2_regularization"
)

// DenseWithBias adds a dense linear layer, a learnable linear transformation plus a bias term.
//
// It the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
func DenseWithBias(ctx *context.Context, input *Node, outputDimensions ...int) *Node {
	return Dense(ctx, input, true, outputDimensions...)
}

// Dense adds a dense linear layer, a learnable linear transformation. Optionally it
// can include a bias term.
//
// It the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
func Dense(ctx *context.Context, input *Node, useBias bool, outputDimensions ...int) *Node {
	g := input.Graph()
	if !g.Ok() || !input.Ok() {
		return g.InvalidNode()
	}
	if !ctx.Ok() {
		g.SetErrorf("context passed to layers.Dense has errors: %w", ctx.Error())
		return g.InvalidNode()
	}
	ctx = ctx.In("dense")
	inputShape := input.Shape()
	inputRank := inputShape.Rank()
	if inputRank == 0 {
		g.SetErrorf("input for layers.Dense needs to have rank >= 1, got %s", input.Shape())
		return g.InvalidNode()
	}
	if len(outputDimensions) == 0 {
		g.SetErrorf("at least one outputDimension must be given for layers.Dense, got 0 -- use outputDims=[1] for a scalar output")
		return g.InvalidNode()
	}
	inputLastDimension := inputShape.Dimensions[inputShape.Rank()-1]

	// Linear transformation.
	weightsDims := make([]int, 1+len(outputDimensions))
	weightsDims[0] = inputLastDimension
	copy(weightsDims[1:], outputDimensions)
	weightsVar := ctx.VariableWithShape("weights", shapes.Make(inputShape.DType, weightsDims...))
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

	// Add bias.
	if useBias {
		biasVar := ctx.VariableWithShape("biases", shapes.Make(inputShape.DType, outputDimensions...))
		bias := biasVar.ValueGraph(g)
		if !output.Ok() {
			return g.InvalidNode()
		}
		expandedBiasShape := output.Shape().Copy()
		for ii := range expandedBiasShape.Dimensions[:output.Rank()-len(outputDimensions)] {
			expandedBiasShape.Dimensions[ii] = 1
		}
		expandedBias := ReshapeWithShape(bias, expandedBiasShape)
		output = Add(output, expandedBias)
	}

	// Add regularization -- notice not for the bias term.
	if l2any, found := ctx.GetParam(L2RegularizationKey); found {
		l2 := l2any.(float64)
		if l2 > 0 {
			l2Node := Const(g, shapes.CastAsDType(l2, inputShape.DType))
			AddL2Regularization(ctx, l2Node, weights)
		}
	}

	return output
}

// Relu returns Max(x, 0), and is commonly used as an activation function in neural networks.
func Relu(x *Node) *Node {
	if !x.Ok() {
		return nil
	}
	g := x.Graph()
	if !g.Ok() {
		return nil
	}
	return Max(x, ZerosLike(x))
}

// Embedding creates an embedding table with vocabSize elements (typically a vocabulary size)
// each of dimension values -- so a [vocabSize, dimension] variable table.
//
// It then converts each integer value of the input to an embedding of the given dimension size.
// The input must have an integer dtype, and the last dimension must be of size 1. If it's not
// of size one, an extra dimension is added to the end. All values of the input
// must smaller than vocabSize, otherwise it will fail -- no checking is explicitly made.
//
// The output has rank one larger than the input, with the last dimension the same as
// the embedding dimension.
func Embedding(ctx *context.Context, input *Node, dtype shapes.DType, vocabSize, dimension int) *Node {
	g := input.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	inputShape := input.Shape()
	if !inputShape.DType.IsInt() {
		g.SetErrorf("can only use Embedding on integer inputs, passed %s instead", input.Shape())
		return g.InvalidNode()
	}
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		// Add a last dimension of size 1, since we are pointing to a table that needs
		// and index of size 1.
		input = ExpandDims(input, -1)
	}
	embeddingTable := ctx.VariableWithShape("embeddings", shapes.Make(dtype, vocabSize, dimension))
	return Gather(embeddingTable.ValueGraph(input.Graph()), input)
}

// ValidateQuantilesForPWLCalibration validate that raw values for quantiles are ok to be used for
// PieceWiseLinearCalibration. It checks for:
//   - Enough data points.
//   - Monotonicity of data points: quantiles should always be increasing.
func ValidateQuantilesForPWLCalibration[T constraints.Ordered](values []T) error {
	if len(values) < 2 {
		return fmt.Errorf("PieceWiseLinearCalibration requires at least 2 quantile values")
	}
	current := values[0]
	for ii, value := range values[1:] {
		if value <= current {
			return fmt.Errorf("quantile %d (out of %d), valued %v, for PieceWiseLinearCalibration is out of order or repeated",
				ii, len(values), value)
		}
		current = value
	}
	return nil
}

// PieceWiseLinearCalibration creates a piece-wise linear function from the input, splitting
// it in the given keypoints with outputs initialized
// with values from 0 to 1.
//
// The keypoints are typically quantiles of the input feature, starting with the minimum value
// and ending on the maximum. It must have rank-1 and be of the same DType as input.
// Its values must be ordered, and cannot be repeated (this may lead to NaNs). Consider using
// ValidateQuantilesForPWLCalibration on the quantiles.
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
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !ctx.Ok() {
		g.SetErrorf("context passed to PieceWiseLinearCalibration() has errors: %w", ctx.Error())
		return g.InvalidNode()
	}
	ctx = ctx.In("piece_wise_linear")
	if !input.Ok() {
		g.SetErrorf("input Node is not ok: %s", input)
		return g.InvalidNode()
	}
	if !input.DType().IsFloat() {
		g.SetErrorf("PieceWiseLinearCalibration only accepts float inputs, but got %s", input.Shape())
		return g.InvalidNode()
	}
	if keypoints.Rank() != 1 || keypoints.Shape().Dimensions[0] < 2 {
		g.SetErrorf("PieceWiseLinearCalibration keypoints shape %q invalid, it must be rank-1 and at list size 2", keypoints.Shape())
		return g.InvalidNode()
	}
	if keypoints.DType() != input.DType() {
		g.SetErrorf("PieceWiseLinearCalibration keypoints DType %s != input's DType %s",
			keypoints.DType(), input.DType())
		return g.InvalidNode()
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
	kpStarts := ExpandDims(Slice(keypoints, AxisRange(0, numKeypoints-1)), 0)
	kpEnds := ExpandDims(Slice(keypoints, AxisRange(1, numKeypoints)), 0)
	lengths := Sub(kpEnds, kpStarts)

	//
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)
	trimLeft := PositiveIndicator(Sub(input2D, kpStarts))
	trimRight := StrictlyPositiveIndicator(Sub(kpEnds, input2D))
	trimRegions := Mul(trimLeft, trimRight)

	// Calculate "left" weights on each keypoint: left weight is the weight
	leftWeights := Div(Sub(input2D, kpStarts), lengths)
	leftWeights = Min(Max(leftWeights, zero), one)
	rightWeights := OneMinus(leftWeights)

	leftWeights = Mul(trimRegions, leftWeights)
	rightWeights = Mul(trimRegions, rightWeights)

	// left and right weights are shifted by one from one another. And we need to add the weights
	// on the edge keypoints.
	leftEdge := ExpandDims(SliceXLA(keypoints, []int{0}, []int{1}), 0)
	leftEdge = PositiveIndicator(Sub(leftEdge, input2D))
	leftWeights = Concatenate([]*Node{leftEdge, leftWeights}, -1)

	rightEdge := ExpandDims(SliceXLA(keypoints, []int{numKeypoints - 1}, []int{numKeypoints}), 0)
	rightEdge = PositiveIndicator(Sub(input2D, rightEdge))
	rightWeights = Concatenate([]*Node{rightWeights, rightEdge}, -1)

	weights := Add(leftWeights, rightWeights)

	// The calibrated value is the weighted sum of the output keypoints.
	calibrated := ReduceSum(Mul(weights, ExpandDims(outputKeypointsNode, 0)), 1)
	return ReshapeWithShape(calibrated, inputShape)
}

// PieceWiseLinearCalibrationCascaded is a similar implementation for PieceWiseLinearCalibration that
// is equally powerful (express the same functions) simpler (fewer ops) and faster, but is parametrizing
// differently (cascaded linear functions), and may have different learning characteristics when
// doing gradient descent.
func PieceWiseLinearCalibrationCascaded(ctx *context.Context, input, keypoints *Node, outputTrainable bool) *Node {
	g := input.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !ctx.Ok() {
		g.SetErrorf("context passed to PieceWiseLinearCalibration() has errors: %w", ctx.Error())
		return g.InvalidNode()
	}
	ctx = ctx.In("piece_wise_linear")
	if !input.Ok() {
		g.SetErrorf("input Node is not ok: %s", input)
		return g.InvalidNode()
	}
	if !input.DType().IsFloat() {
		g.SetErrorf("PieceWiseLinearCalibration only accepts float inputs, but got %s", input.Shape())
		return g.InvalidNode()
	}
	if keypoints.Rank() != 1 || keypoints.Shape().Dimensions[0] < 2 {
		g.SetErrorf("PieceWiseLinearCalibration keypoints shape %q invalid, it must be rank-1 and at list size 2", keypoints.Shape())
		return g.InvalidNode()
	}
	if keypoints.DType() != input.DType() {
		g.SetErrorf("PieceWiseLinearCalibration keypoints DType %s != input's DType %s",
			keypoints.DType(), input.DType())
		return g.InvalidNode()
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
	kpStarts := ExpandDims(SliceXLA(keypoints, []int{0}, []int{numKeypoints - 1}), 0)
	kpEnds := ExpandDims(SliceXLA(keypoints, []int{1}, []int{numKeypoints}), 0)
	lengths := Sub(kpEnds, kpStarts)

	// weights so far applies to outputKeypoints[1:]: it is shaped [numInputs, numKeypoints-1]
	weights := Div(Sub(input2D, kpStarts), lengths)
	weights = Min(Max(weights, ZerosLike(weights)), OnesLike(weights))

	//  We need to concatenate a weight of 1 for keypoint[0] as bias.
	weights = Concatenate([]*Node{
		Ones(g, shapes.Make(weights.DType(), numInputs, 1)),
		weights}, -1) // Now weights has shape [numInputs, numKeypoints]

	// The calibrated value is the weighted sum of the output keypoints.
	calibrated := ReduceSum(Mul(weights, ExpandDims(outputKeypointsNode, 0)), 1)
	return ReshapeWithShape(calibrated, inputShape)
}

// Dropout randomly replace the input with zeros if ctx.IsTraining() is true. Otherwise,
// it's a no op (it returns input). It scales the output by 1/(1-dropoutRate)
// to preserve the mean of the values of the input.
func Dropout(ctx *context.Context, input *Node, dropoutRate *Node) *Node {
	return DropoutNormalize(ctx, input, dropoutRate, true)
}

// DropoutNormalize randomly replace the input with zeros if ctx.IsTraining() is true. Otherwise,
// it's a no op (it returns input). If normalize is set, it scales the output by 1/(1-dropoutRate)
// to preserve the mean of the values of the input.
func DropoutNormalize(ctx *context.Context, input *Node, dropoutRate *Node, normalize bool) *Node {
	g := input.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !ctx.Ok() {
		g.SetErrorf("context has errors: %w", ctx.Error())
		return g.InvalidNode()
	}
	if !input.Ok() {
		g.SetErrorf("input Node is not ok: %s", input)
		return g.InvalidNode()
	}
	if !ctx.IsTraining(g) {
		return input
	}

	// Disable (by multiplying by 0) random entries.
	dtype := dropoutRate.DType()
	dims := input.Shape().Dimensions
	rnd := RngUniform(ScalarZero(g, dtype), ScalarOne(g, dtype), shapes.Make(dtype, dims...))
	broadcastRate := BroadcastToDims(dropoutRate, dims...)
	result := Where(LessOrEqual(rnd, broadcastRate), ZerosLike(input), input)
	if normalize {
		// Normalize input values, so mean value remains constant.
		keepRate := ConvertType(OneMinus(dropoutRate), input.DType())
		result = Div(result, keepRate)
	}
	return result
}

// AddL2Regularization calculates the L2 of the given values (typically variable nodes returned
// by context.Variable.ValueGraph()), scale by the given amount (typically a constant) and then
// train.AddLoss the resulting value, having the effect of regularizing the weights (variables).
func AddL2Regularization(ctx *context.Context, amount *Node, values ...*Node) {
	graph := amount.Graph()
	if !ctx.Ok() || !graph.Ok() {
		return
	}
	if len(values) == 0 {
		graph.SetErrorf("no values given to AddL2Regularization")
		return
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
