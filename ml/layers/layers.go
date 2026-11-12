// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package layers holds a collection of common modeling layers. It includes dense layer, convolutions (TODO),
// activation functions, dropout (TODO), etc.
//
// A small convention on naming: typically layers are nouns (like "Convolution", "Dense" (layer), "MultiHeadAttention"),
// while computations are usually verbs ("Convolve", "Reduce..", "Multiply (Mul)", etc.).
package layers

import (
	"cmp"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/ml/train"
	. "github.com/gomlx/gomlx/support/exceptions"
)

const (
	// ParamL2Regularization scope hyperparameter defines the L2 regularization of kernels.
	// Each layer may decide independently to implement it or not.
	//
	// This is an alias to regularizers.ParamL2
	// Dense, DenseWithBias, FNN, kan and Convolution kernels look at this hyperparameter.
	// The value should be a float64.
	// The default is `0.0`.
	//
	// Deprecated: use regularizers.ParamL2
	ParamL2Regularization = "l2_regularization"

	// ParamDropoutRate scope hyperparameter defines the amount of dropout applied when DropoutFromScope is used.
	// Should be a value from `0.0` to `1.0`, where 0 means no dropout, and 1 would drop everything out.
	//
	// It is only applied if `Scope.IsTraining() == true`, that is, during evaluation/inference it is
	// ignored.
	//
	// The default is `0.0`, which means no dropout.
	ParamDropoutRate = "dropout_rate"

	// ParamDropPathProbability provides the probability of DropPathFromScope to drop paths.
	//
	// This is only applied if the model actually calls DropPathFromScope.
	//
	// Default is `0.0`, which means never do any DropPath.
	ParamDropPathProbability = "droppath_prob"

	// ParamQuantizedMatmul scope hyperparameter controls whether quantized dense layers
	// quantize floating-point activations on-the-fly for integer dot products (true, the default)
	// or dequantize weights to floating-point (false). Can also be overridden per layer via
	// [DenseBuilder.WithQuantizedMatmul].
	ParamQuantizedMatmul = "quantized_matmul"
)

// DenseWithBias adds a single dense linear layer, a learnable linear transformation plus a bias term.
//
// It creates a "dense" sub-scope under `scope`.
//
// If the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// Deprecated: Use [NewDense] to configure layout, fused activation, regularization, or scope behavior,
// and `fnn.New` for multi-layer feedforward networks.
func DenseWithBias(scope *model.Scope, input *Node, outputDimensions ...int) *Node {
	return NewDense(scope, input, outputDimensions...).WithBias(true).Done()
}

// Dense adds a single dense linear layer, a learnable linear transformation.
// Optionally, it can include a bias term. Weight layout is [compute.DenseLayoutInputOutputs].
//
// It creates a "dense" sub-scope under `scope`.
//
// It automatically adds regularization to the weights (not to biases) configured in hyperparameters -- see [regularizer.FromScope].
//
// If the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// Deprecated: Use [NewDense] to configure layout, fused activation, regularization, or scope behavior,
// and `fnn.New` for multi-layer feedforward networks.
func Dense(scope *model.Scope, input *Node, useBias bool, outputDimensions ...int) *Node {
	return NewDense(scope, input, outputDimensions...).WithBias(useBias).Done()
}

// DenseWithLayout adds a single dense linear layer with a specific weight layout.
//
// It creates a "dense" sub-scope under `scope`.
//
// If layout is [compute.DenseLayoutInputOutputs], weights have shape `[inputLastDimension, prod(outputDimensions)]`.
// If layout is [compute.DenseLayoutOutputsInput], weights have shape `[prod(outputDimensions), inputLastDimension]`.
//
// Optionally, it can include a bias term.
// It automatically adds regularization to the weights (not to biases) configured in hyperparameters -- see [regularizer.FromScope].
//
// If the input has shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// Deprecated: Use [NewDense] for additional options such as fused activations.
func DenseWithLayout(scope *model.Scope, input *Node, layout compute.DenseLayout, useBias bool, outputDimensions ...int) *Node {
	return NewDense(scope, input, outputDimensions...).WithBias(useBias).WithLayout(layout).Done()
}

// DenseBuilder configures and builds a single dense (linear) layer.
// Create it with [NewDense], chain optional configuration methods, and call [DenseBuilder.Done].
type DenseBuilder struct {
	scope              *model.Scope
	input              *Node
	outputDimensions   []int
	useBias            bool
	layout             compute.DenseLayout
	activation         activation.Type
	regularizer        regularizer.Regularizer
	regularizerSet     bool
	quantizedMatmul    bool
	quantizedMatmulSet bool
	newScope           bool
}

// NewDense prepares a single dense (linear) projection layer on `input` producing `<outputDimensions...>`.
//
// Defaults:
//   - Sub-scope: creates `"dense"` under `scope` (call [DenseBuilder.CurrentScope] to place `"weights"`/`"biases"` directly in `scope`).
//   - Bias: disabled (`false`; enable with [DenseBuilder.WithBias]).
//   - Weight layout: [compute.DenseLayoutInputOutputs] (override with [DenseBuilder.WithLayout]).
//   - Activation: [activation.TypeNone] (override with [DenseBuilder.WithActivation]).
//   - Regularization: [regularizer.FromScope] on the layer scope (override with [DenseBuilder.WithRegularizer]).
//   - Quantized matmul: reads [ParamQuantizedMatmul] from the layer scope, defaulting to `true`
//     (override per layer with [DenseBuilder.WithQuantizedMatmul]).
//
// Quantized weights do not need explicit builder configuration: if the `"weights"` variable in the
// target scope already exists with quantization metadata (see [model.Variable.WithQuantization]),
// [DenseBuilder.Done] automatically dispatches to [nn.QuantizedDense] without materializing
// dequantized floating-point weights.
func NewDense(scope *model.Scope, input *Node, outputDimensions ...int) *DenseBuilder {
	if input.Rank() == 0 {
		Panicf("input for layers.NewDense needs to have rank >= 1, got %s", input.Shape())
	}
	if len(outputDimensions) == 0 {
		Panicf("at least one outputDimension must be given for layers.NewDense, got 0 -- use outputDims=[1] for a scalar output")
	}
	for _, dim := range outputDimensions {
		if dim <= 0 {
			Panicf("dimensions for outputDimensions in layers.NewDense must be > 0, got %v", outputDimensions)
		}
	}
	return &DenseBuilder{
		scope:            scope,
		input:            input,
		outputDimensions: outputDimensions,
		layout:           compute.DenseLayoutInputOutputs,
		activation:       activation.TypeNone,
		newScope:         true,
	}
}

// WithBias configures whether to include an additive bias term (`"biases"` variable).
// Default is false.
func (b *DenseBuilder) WithBias(useBias bool) *DenseBuilder {
	b.useBias = useBias
	return b
}

// WithLayout configures the memory layout of the `"weights"` variable:
//   - [compute.DenseLayoutInputOutputs] (default): `[inputFeatures, outFeatures]`
//   - [compute.DenseLayoutOutputsInput]: `[outFeatures, inputFeatures]` (used by PyTorch `nn.Linear` and GGUF/GGML)
func (b *DenseBuilder) WithLayout(layout compute.DenseLayout) *DenseBuilder {
	b.layout = layout
	return b
}

// WithActivation configures an activation function applied after the linear projection (and after
// any registered post-linear hooks). When possible, the activation is fused directly into
// [nn.Dense] or [nn.QuantizedDense].
//
// Note: For gating activations that halve the last dimension (such as [activation.TypeSwiGLU]),
// `outputDimensions` specifies the pre-activation linear output dimension (e.g. `dim * act.HiddenDimMultiplier()`).
func (b *DenseBuilder) WithActivation(act activation.Type) *DenseBuilder {
	b.activation = act
	return b
}

// WithRegularizer sets the regularizer applied to `"weights"` (never to `"biases"`), overriding
// [regularizer.FromScope]. Pass `nil` to explicitly disable regularization.
func (b *DenseBuilder) WithRegularizer(reg regularizer.Regularizer) *DenseBuilder {
	b.regularizer = reg
	b.regularizerSet = true
	return b
}

// WithQuantizedMatmul configures whether pre-loaded quantized `"weights"` dynamically quantize
// floating-point input activations on-the-fly for integer dot products (`true`) or dequantize
// weights to floating-point (`false`), overriding [ParamQuantizedMatmul] in `scope`.
//
// If `"weights"` is not quantized, this setting has no effect.
func (b *DenseBuilder) WithQuantizedMatmul(enabled bool) *DenseBuilder {
	b.quantizedMatmul = enabled
	b.quantizedMatmulSet = true
	return b
}

// CurrentScope configures the layer not to create a `"dense"` sub-scope, placing `"weights"` and
// `"biases"` directly in the scope passed to [NewDense].
func (b *DenseBuilder) CurrentScope() *DenseBuilder {
	b.newScope = false
	return b
}

// Done builds the dense layer graph operations and returns the output node.
func (b *DenseBuilder) Done() *Node {
	scope := b.scope
	if b.newScope {
		scope = scope.In("dense")
	}
	g := b.input.Graph()
	reg := b.regularizer
	if !b.regularizerSet {
		reg = regularizer.FromScope(scope)
	}

	inputShape := b.input.Shape()
	inputRank := inputShape.Rank()
	inputLastDimension := inputShape.Dimensions[inputRank-1]

	outDim := 1
	for _, d := range b.outputDimensions {
		outDim *= d
	}

	var weightsDims []int
	switch b.layout {
	case compute.DenseLayoutInputOutputs:
		weightsDims = []int{inputLastDimension, outDim}
	case compute.DenseLayoutOutputsInput:
		weightsDims = []int{outDim, inputLastDimension}
	default:
		Panicf("layers.NewDense: unknown layout %v", b.layout)
	}

	var biasNode *Node
	if b.useBias {
		if existingBias := scope.GetVariable("biases"); existingBias != nil {
			biasNode = existingBias.NodeValue(g)
			if biasNode.DType().IsFloat() && biasNode.DType() != inputShape.DType {
				biasNode = ConvertDType(biasNode, inputShape.DType)
			}
		} else {
			biasVar := scope.VariableWithShape("biases", shapes.Make(inputShape.DType, outDim))
			biasNode = biasVar.NodeValue(g)
		}
	}

	canFuseAct := b.activation != activation.TypeNone &&
		b.activation != activation.TypeSwiGLU &&
		len(b.outputDimensions) == 1 &&
		!hasPostLinearHooks()
	fusedAct := activation.TypeNone
	if canFuseAct {
		fusedAct = b.activation
	}

	var res *Node
	existingWeights := scope.GetVariable("weights")
	switch {
	case existingWeights != nil && existingWeights.Quantization() != nil:
		weights := existingWeights.NodeValue(g)
		quantMatmul := b.quantizedMatmul
		if !b.quantizedMatmulSet {
			quantMatmul = model.GetParamOr(scope, ParamQuantizedMatmul, true)
		}
		qCfg := QuantizedDenseConfig{
			WeightQuantization: existingWeights.Quantization().ToGraph(g),
			WeightLayout:       b.layout,
			QuantizedMatmul:    quantMatmul,
		}
		res = nn.QuantizedDense(b.input, weights, biasNode, qCfg, fusedAct)

	case existingWeights != nil && existingWeights.DType().IsFloat() && existingWeights.DType() != inputShape.DType:
		if reg != nil {
			reg(g, existingWeights)
		}
		weights := ConvertDType(existingWeights.NodeValue(g), inputShape.DType)
		res = nn.Dense(b.input, weights, biasNode, b.layout, fusedAct)

	default:
		weightsVar := scope.VariableWithShape("weights", shapes.Make(inputShape.DType, weightsDims...))
		if reg != nil {
			// Only for the weights, not for the bias.
			reg(g, weightsVar)
		}
		weights := weightsVar.NodeValue(g)
		res = nn.Dense(b.input, weights, biasNode, b.layout, fusedAct)
	}

	res = ApplyPostLinearHooks(scope, b.input, res)
	if len(b.outputDimensions) > 1 {
		if inputShape.IsDynamic() {
			outAxisSpecs := make([]DimensionSpec, inputRank-1+len(b.outputDimensions))
			for i := range inputRank - 1 {
				outAxisSpecs[i] = DimensionSpecFor(b.input, i)
			}
			for j, d := range b.outputDimensions {
				outAxisSpecs[inputRank-1+j] = StaticDim(d)
			}
			res = DynamicReshape(res, outAxisSpecs...)
		} else {
			newDims := make([]int, inputRank-1+len(b.outputDimensions))
			copy(newDims, inputShape.Dimensions[:inputRank-1])
			copy(newDims[inputRank-1:], b.outputDimensions)
			res = Reshape(res, newDims...)
		}
	}
	if b.activation != activation.TypeNone && !canFuseAct {
		res = activation.Apply(b.activation, res)
	}
	return res
}

// Embedding creates an embedding table with vocabSize elements (typically a vocabulary size)
// each of dimension values -- so a [vocabSize, dimension] variable table.
//
// If the "embeddings" variable in `scope` already exists with quantization metadata
// (see [model.Variable.WithQuantization]), it automatically dispatches to [nn.QuantizedGather]
// to dequantize only the selected token rows on-the-fly.
//
// It then converts each integer value of the input to an embedding of the given dimension size.
// The input must have an integer dtype, and the last dimension must be of size 1. If it's not
// of size one, an extra dimension is added to the end. All values of the input
// must smaller than vocabSize, otherwise it will fail -- no checking is explicitly made.
//
// indicesAreSorted should be set to true only if the input is guaranteed to be sorted (ascending).
// This allows some optimizations in some compute. The default is false (for compatibility),
// in future versions this argument will be made obligatory.
//
// The output has rank one larger than the input, with the last dimension the same as
// the embedding dimension.
func Embedding(scope *model.Scope, input *Node, dtype dtypes.DType, vocabSize, dimension int, indicesAreSorted ...bool) *Node {
	inputShape := input.Shape()
	if !inputShape.DType.IsInt() {
		Panicf("can only use Embedding on integer inputs, passed %s instead", input.Shape())
	}
	if inputShape.IsScalar() || inputShape.Dimensions[inputShape.Rank()-1] != 1 {
		// Add a last dimension of size 1, since we are pointing to a table that needs
		// and index of size 1.
		input = InsertAxes(input, -1)
	}
	g := input.Graph()
	if existing := scope.GetVariable("embeddings"); existing != nil {
		if existing.Quantization() != nil {
			res := nn.QuantizedGather(existing.NodeValue(g), input, existing.Quantization().ToGraph(g))
			if res.DType() != dtype {
				res = ConvertDType(res, dtype)
			}
			return res
		}
		if existing.DType() != dtype {
			res := Gather(existing.NodeValue(g), input, indicesAreSorted...)
			return ConvertDType(res, dtype)
		}
	}
	embeddingTable := scope.VariableWithShape("embeddings", shapes.Make(dtype, vocabSize, dimension))
	return Gather(embeddingTable.NodeValue(g), input, indicesAreSorted...)
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
func PieceWiseLinearCalibration(scope *model.Scope, input, keypoints *Node, outputTrainable bool) *Node {
	g := input.Graph()
	scope = scope.In("piece_wise_linear")
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

	// Normalize input to rank-2: [batch_size, 1] -- it works with any rank input and both static and dynamic shapes.
	input2D := DynamicReshape(input, NamedInferredDim(g.UniqueName("pwl_calibration_size")), StaticDim(1))

	// Initialize outputKeypoints uniformly.
	outputKeypoints := make([]float64, numKeypoints)
	for ii := range outputKeypoints {
		outputKeypoints[ii] = float64(ii) / float64(numKeypoints-1)
	}
	outKPValue := shapes.CastAsDType(outputKeypoints, dtype)
	var outputKeypointsNode *Node
	if outputTrainable {
		outputKeypointsNode = scope.VariableWithValue("output_keypoints", outKPValue).NodeValue(g)
	} else {
		outputKeypointsNode = Const(g, outKPValue)
	}
	_ = outputKeypointsNode

	// Calculate lengths of each linear piece: all in shape [1, numKeypoints-1]
	kpStarts := InsertAxes(Slice(keypoints, AxisRange(0, numKeypoints-1)), 0)
	kpEnds := InsertAxes(Slice(keypoints, AxisRange(1, numKeypoints)), 0)
	lengths := Sub(kpEnds, kpStarts)

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
	return DynamicReshapeLike(calibrated, input)
}

// PieceWiseLinearCalibrationCascaded is a similar implementation for PieceWiseLinearCalibration that
// is equally powerful (express the same functions) simpler (fewer ops) and faster, but is parametrizing
// differently (cascaded linear functions), and may have different learning characteristics when
// doing gradient descent.
func PieceWiseLinearCalibrationCascaded(scope *model.Scope, input, keypoints *Node, outputTrainable bool) *Node {
	g := input.Graph()
	scope = scope.In("piece_wise_linear")
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
		outputKeypointsNode = scope.VariableWithValue("output_keypoints", outKPValue).NodeValue(g)
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

// Dropout randomly replace the input with zeros if scope.IsTraining() is true. Otherwise,
// it's a no op (it returns input).
// If the input is float, it scales the output by 1/(1-dropoutRate) to preserve the mean of the values of the input.
func Dropout(scope *model.Scope, input *Node, dropoutRate *Node) *Node {
	return DropoutNormalize(scope, input, dropoutRate, input.DType().IsFloat())
}

// DropoutStatic is the same as Dropout, but it takes the `dropoutRate` as a static value, given as a float64.
// If `dropoutRate <= 0` or it's not training, this is a no-op.
func DropoutStatic(scope *model.Scope, input *Node, dropoutRate float64) *Node {
	if dropoutRate <= 0 {
		return input
	}
	g := input.Graph()
	return Dropout(scope, input, Scalar(g, dtypes.Float32, dropoutRate))
}

// IsDropoutActive returns true if dropout is active (i.e. if it's training).
// If using DropoutStatic, also check that dropoutRate > 0.
func IsDropoutActive(scope *model.Scope, g *Graph) bool {
	return scope != nil && scope.IsTraining(g)
}

// DropoutNormalize randomly replace the input with zeros if scope.IsTraining() is true. Otherwise,
// it's a no op (it returns input). If normalize is set, it scales the output by 1/(1-dropoutRate)
// to preserve the mean of the input values.
func DropoutNormalize(scope *model.Scope, input *Node, dropoutRate *Node, normalize bool) *Node {
	g := input.Graph()
	if !IsDropoutActive(scope, g) {
		return input
	}

	// Disable (by multiplying by 0) random entries.
	dtype := dropoutRate.DType()
	dims := input.Shape().Dimensions
	rnd := scope.RandomUniform(g, shapes.Make(dtype, dims...))
	broadcastRate := BroadcastToDims(dropoutRate, dims...)
	result := Where(LessOrEqual(rnd, broadcastRate), ZerosLike(input), input)
	if normalize {
		// Normalize input values, so mean value remains constant.
		keepRate := ConvertDType(OneMinus(dropoutRate), input.DType())
		result = Div(result, keepRate)
	}
	return result
}

// DropoutFromScope applies a dropout configured in the scope parameters keyed by [ParamDropoutRate].
//
// If it is 0.0 this is a no-op.
// If `Scope.IsTraining() == false` this is also a no-op, so it doesn't impact evaluation or inference.
func DropoutFromScope(scope *model.Scope, x *Node) *Node {
	dropoutRate := model.GetParamOr(scope, ParamDropoutRate, 0.0)
	if dropoutRate > 0 {
		// We apply edge dropout to the mask.
		g := x.Graph()
		normalize := x.DType().IsFloat()
		x = DropoutNormalize(scope, x, Scalar(g, x.DType(), dropoutRate), normalize)
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
// See also DropPathFromScope.
func DropPath(scope *model.Scope, x, dropProbability *Node) *Node {
	g := x.Graph()
	if !scope.IsTraining(g) || dropProbability == nil {
		return x
	}
	maskShape := x.Shape().Clone()
	for ii := 1; ii < maskShape.Rank(); ii++ {
		maskShape.Dimensions[ii] = 1
	}
	return Mul(x, scope.RandomBernoulli(OneMinus(dropProbability), maskShape))
}

// DropPathFromScope will execute DropPath if the hyperparameter ParamDropPathProb is set to a value > 0.
// If ParamDropPathProb is not set or if not training, this is a no-op.
func DropPathFromScope(scope *model.Scope, x *Node) *Node {
	g := x.Graph()
	if !scope.IsTraining(g) {
		return x
	}
	dropPathProb := model.GetParamOr(scope, ParamDropPathProbability, 0.0)
	if dropPathProb > 0 {
		// We apply edge dropout to the mask.
		g := x.Graph()
		x = DropPath(scope, x, Scalar(g, x.DType(), dropPathProb))
	}
	return x
}

// AddL2RegularizationStatic is like AddL2Regularization, but takes the `amount` as a static Go float64 value.
//
// Deprecated: use package regularizers instead.
func AddL2RegularizationStatic(scope *model.Scope, amount float64, values ...*Node) {
	if len(values) == 0 {
		Panicf("no values given to AddL2RegularizationAsFloat")
	}
	g := values[0].Graph()
	amountNode := Scalar(g, values[0].DType(), amount)
	AddL2Regularization(scope, amountNode, values...)
}

// AddL2Regularization calculates the L2 of the given values (typically variable nodes returned
// by model.Variable.NodeValue()), scale by the given amount (typically a constant) and then
// train.AddLoss the resulting value, having the effect of regularizing the weights (variables).
//
// Deprecated: use package regularizers instead.
func AddL2Regularization(scope *model.Scope, amount *Node, values ...*Node) {
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
	train.AddRegularizationLoss(loss)
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
