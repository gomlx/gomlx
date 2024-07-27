package kan

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"k8s.io/klog/v2"
	"math"
)

// Discrete-KAN is a variation of KAN that uses a piecewise-constant function (PCF for short) as
// its univariate function family -- as opposed to the original spline functions.
//
// These PCFs (piecewise-constant functions) have "split points", where the function change values,
// and "control points" which are the outputs of the function between the corresponding split points.
// For N split points, there are N+1 split points.
// The first and last regions extend to -/+ infinity, so it extrapolates as a constant.
//
// The goal is to train models that for inference require no multiplications, only additions and a max(x, 0) for
// the relu activations, and can be executed efficiently in any cheap hardware -- not requiring specialized
// accelerators.
//
// in order to make the split points of the PCFs trainable, and regularize, during training we can
// "soften" the PCFs, by implicitly converting the input of a PCF (let's call it x) to a probability distribution
// (aka. "input perturbation"). Then integrate over the distribution over x to calculate the result of the
// softened PCF. See details for this in [1].
//
// [1] Learning Representations for Axis-Aligned Decision Forests through Input Perturbation -
// Sebastian Bruch, Jan Pfeifer, Mathieu Guillame-Bert - https://arxiv.org/pdf/2007.14761

var (
	// ParamDiscrete indicates whether to use Discrete-KAN as the univariate function to learn.
	ParamDiscrete = "kan_discrete"

	// ParamDiscreteSoftness indicates whether to soften the PCF (piecewise contant functions) during training,
	// and by how much.
	ParamDiscreteSoftness = "kan_discrete_softness"
)

// Discrete configures the KAN to use a "piecewise-constant" functions (as opposed to splines) to model \phi(x),
// the univariate function used in the pape, and set the number of control points to use for the function.
//
// The numControlPoints must be greater or equal to 2, and it defaults to 20 and can also be set by using the
// hyperparameter ParamNumControlPoints ("kan_num_points").
func (c *Config) Discrete(numControlPoints int) *Config {
	c.useDiscrete = true
	c.discreteControlPoints = numControlPoints
	if c.discreteControlPoints < 2 {
		exceptions.Panicf("kan: discrete version requires at least 2 control points, %d given", c.discreteControlPoints)
	}
	return c
}

// DiscreteSoftness sets how much softness to use during training.
// If set to 0 softness is disabled.
//
// The default is 0.1, and it can be set with the hyperparameter ParamDiscreteSoftness ("kan_discrete_softness").
func (c *Config) DiscreteSoftness(softness float64) *Config {
	c.discreteSoftness = softness
	return c
}

// Layer implements one KAN bsplineLayer. x is expected to be rank-2.
func (c *Config) discreteLayer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	g := x.Graph()
	dtype := x.DType()
	residual := x
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	batchSize := x.Shape().Dimensions[0]

	if klog.V(2).Enabled() {
		klog.Infof("kan discreteLayer (%s): (%d+2) x %d x %d = %d weights\n",
			ctx.Scope(), c.discreteControlPoints, numInputNodes, numOutputNodes,
			(c.discreteControlPoints)*numInputNodes*numOutputNodes)
	}

	// Apply piecewise-constant function (PCF)
	controlPointsVar := ctx.WithInitializer(initializers.RandomNormalFn(0, 0.01)).
		VariableWithShape("discrete_control_points", shapes.Make(dtype, numOutputNodes, numInputNodes, c.discreteControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)
	// splitPoints are fixed, and invariant to the nodes: they are always the same.
	splitPoints := Iota(g, shapes.Make(dtype, 1, 1, c.discreteControlPoints-1), -1)
	if c.discreteControlPoints > 2 {
		splitPoints = DivScalar(splitPoints, float64(c.discreteControlPoints-2))
	} else {
		// For 2 control points, splitPoints is 0.5
		splitPoints = AddScalar(splitPoints, 0.5)
	}
	splitPoints = AddScalar(MulScalar(splitPoints, 2), -1)

	var output *Node
	if c.discreteSoftness <= 0 {
		output = PiecewiseConstantFunction(x, controlPoints, splitPoints)
	} else {
		softnessConst := Scalar(g, dtype, c.discreteSoftness)
		output = PiecewiseConstantFunctionWithInputPerturbation(x, controlPoints, splitPoints, softnessConst)
	}
	output.AssertDims(batchSize, numOutputNodes, numInputNodes) // Shape=[batch, outputs, inputs]

	// Per section "2.2 Kan Architecture"
	if c.bsplineResidual {
		residual = activations.Apply(c.activation, residual)
		residual = ExpandDims(residual, 1)
		residual.AssertDims(batchSize, 1, numInputNodes)
		output = Add(output, residual)
	}

	// ReduceMean the inputs to get the outputs: we use the mean (and not sum) because if the number of inputs is large
	// with ReduceSum we would get large numbers (and gradients) that are harder for the gradient descent to learn.
	// In particular for multiple hidden-layers: there is a geometric growth of the values per number of layers.
	output = ReduceMean(output, -1)
	output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]
	return output
}

// PiecewiseConstantFunction (PCF) generates a PCF output for a cross of numInputNodes x numOutputNodes, defined
// as the dimensions of its inputs, as follows:
//
//   - input to be transformed, shaped [batchSize, numInputNodes]
//   - controlPoints are the output values of the PCFs, and should be shaped [numOutputNodes, numInputNodes, NumControlPoints]
//   - splitPoints are the splitting values of the inputs, shaped [numOutputNodes or 1, numInputNodes or 1, NumControlPoints-1],
//     but if the first or the second axes are set to 1, they are broadcast accordingly.
//
// The output will be shaped [batchSize, numOutputPoints, numInputNodes].
// Presumably, the caller will graph.ReduceSum on the last axis (after residual value is added) for a shape [batchSize, numOutputPoints].
func PiecewiseConstantFunction(input, controlPoints, splitPoints *Node) *Node {
	g := input.Graph()

	// Expand missing dimensions for scalar evaluation.
	inputRank := input.Rank()
	if inputRank == 0 {
		input = Reshape(input, 1, 1) // batchSize=1, numInputNodes=1
	} else if inputRank == 1 {
		input = Reshape(input, -1, 1) // batchSize=?, numInputNodes=1
	}
	if controlPoints.Rank() == 1 {
		controlPoints = ExpandDims(controlPoints, 0, 0)
	}
	if splitPoints.Rank() == 1 {
		splitPoints = ExpandDims(splitPoints, 0, 0)
	}

	// Various dimensions and assertions of correctness of shapes.
	batchSize := input.Shape().Dimensions[0]
	numInputNodes := input.Shape().Dimensions[1]
	numOutputNodes := controlPoints.Shape().Dimensions[0]
	numControlPoints := controlPoints.Shape().Dimensions[2]
	input.AssertDims(batchSize, numInputNodes)
	controlPoints.AssertDims(numOutputNodes, numInputNodes, numControlPoints)

	// Standard PCF, no softening.
	expandedInputs := ExpandDims(input, -2, -1)
	expandedSplitPoints := ExpandDims(splitPoints, 0) // Shape [1, numOutputNodes, numInputNodes, NumControlPoints-1]
	toTheLeft := LessThanTotalOrder(expandedInputs,
		GrowRight(expandedSplitPoints, -1, 1, math.Inf(1)))
	toTheRight := GreaterOrEqualTotalOrder(expandedInputs,
		GrowLeft(expandedSplitPoints, -1, 1, math.Inf(-1)))
	controlPicks := ArgMax(And(toTheLeft, toTheRight), -1, dtypes.Int16)
	//controlPicks.AssertDims(batchSize, numOutputNodes, numInputNodes)
	controlPicks = ExpandDims(controlPicks, -1)
	controlPicks = Concatenate([]*Node{
		Iota(g, controlPicks.Shape(), 1), // output node index.
		Iota(g, controlPicks.Shape(), 2), // input node index.
		controlPicks,
	}, -1)
	controlPicks = StopGradient(controlPicks) // No gradient with respect to the picks.
	// Makes sure we broadcast on the output dimension, if needed.
	controlPicks = BroadcastToDims(controlPicks, batchSize, numOutputNodes, numInputNodes, 3)

	output := Gather(controlPoints, controlPicks)
	output.AssertDims(batchSize, numOutputNodes, numInputNodes)
	if inputRank == 0 {
		output = Reshape(output) // Back to scalar.
	} else if inputRank == 1 {
		output = Reshape(output, -1) // Back to scalar.
	}
	return output
}

// PiecewiseConstantFunctionWithInputPerturbation works similarly to PiecewiseConstantFunction, but
// adds a "perturbation" of the inputs by a triangular distribution of the value, controlled by smoothness.
//
// The shapes and inputs are the same as PiecewiseConstantFunction, with the added smoothness parameter
// that should be a scalar with suggested values from 0 to 1.0.
//
// The smoothness softens the function by perturbing the input using a triangular distribution,
// whose base is given by softness * 2 * (splitPoint[-1] - splitPoint[0]).
// If softness is 0, the function is back to being piece-wise constant.
//
// The softening makes it differentiable with respect to the splitPoints, and hence can be used for training.
// One can control the softness as a form of annealing on the split points. As it reaches 0, the split points are
// no longer changed (only the control points).
//
// The output will be shaped [batchSize, numOutputPoints, numInputNodes].
// Presumably, the caller will graph.ReduceSum on the last axis (after residual value is added) for a shape [batchSize, numOutputPoints].
func PiecewiseConstantFunctionWithInputPerturbation(input, controlPoints, splitPoints, softness *Node) *Node {
	// Expand missing dimensions for scalar evaluation.
	inputRank := input.Rank()
	if inputRank == 0 {
		input = Reshape(input, 1, 1) // batchSize=1, numInputNodes=1
	} else if inputRank == 1 {
		input = Reshape(input, -1, 1) // batchSize=?, numInputNodes=1
	}
	if controlPoints.Rank() == 1 {
		controlPoints = ExpandDims(controlPoints, 0, 0)
	}
	if splitPoints.Rank() == 1 {
		splitPoints = ExpandDims(splitPoints, 0, 0)
	}

	// Various dimensions and assertions of correctness of shapes.
	batchSize := input.Shape().Dimensions[0]
	numInputNodes := input.Shape().Dimensions[1]
	numOutputNodes := controlPoints.Shape().Dimensions[0]
	numControlPoints := controlPoints.Shape().Dimensions[2]
	input.AssertDims(batchSize, numInputNodes)
	controlPoints.AssertDims(numOutputNodes, numInputNodes, numControlPoints)
	if softness != nil {
		softness.AssertScalar()
	}

	// Calculate the half-base of the triangle distribution base.
	triangleHalfBase := Sub(
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(numControlPoints-2)),
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(0)),
	)
	triangleHalfBase = Mul(triangleHalfBase, softness)

	// Calculate cumulative distribution function for all split nodes.
	expandedInputs := ExpandDims(input, -2, -1)       // Shape [batchSize, numInputNodes, 1, 1]
	expandedSplitPoints := ExpandDims(splitPoints, 0) // Shape [1, numOutputNodes, numInputNodes, NumControlPoints-1]
	cdfs := triangleDistributionCDF(Sub(expandedSplitPoints, expandedInputs), ExpandDims(triangleHalfBase, 0))
	//cdfs.SetLogged("cdfs")

	// The sum of the integral parts is control points multiplied by the difference between the split points (left
	// and right)
	leftCDF := GrowLeft(cdfs, -1, 1, 0)                   // Prepend the CDF for -inf to the left == 0
	rightCDF := GrowRight(cdfs, -1, 1, 1)                 // Append the CDF for +inf to the right == 1
	expandedControlPoints := ExpandDims(controlPoints, 0) // Shape [1, numOutputNodes, numInputNodes, NumControlPoints]
	weights := Sub(rightCDF, leftCDF)
	//weights.SetLogged("weights")
	termsOfSum := Mul(weights, expandedControlPoints)
	//termsOfSum.SetLogged("termsOfSum")

	// Reduce control points dimensions. Output is [batchSize, numOutputNodes, numInputsNodes].
	// Presumably, it will also be reduced by the caller on the numInputNodes axis.
	output := ReduceSum(termsOfSum, -1)
	output.AssertDims(batchSize, numOutputNodes, numInputNodes)
	if inputRank == 0 {
		output = Reshape(output) // Back to scalar.
	} else if inputRank == 1 {
		output = Reshape(output, -1) // Back to scalar.
	}
	return output
}

// triangleDistributionCDF calculates the cumulative distribution points for a triangle distribution with
// halfBaseLength (the implicit height is 1/halfBaseLength).
//
// The function is univariate, it is calculated for each element of x, and the returned value has the same shape as
// x.
//
// See math derivation in the notebook saved along the discretekan package.
func triangleDistributionCDF(x, halfBase *Node) *Node {
	triangleHeight := Inverse(halfBase) // That will make the area under the triangle = 1.
	leftSide := MulScalar(Square(OnePlus(Mul(x, triangleHeight))), 0.5)
	rightSide := OneMinus(MulScalar(Square(OneMinus(Mul(x, triangleHeight))), 0.5))
	return Where(LessThan(x, Neg(halfBase)),
		ZerosLike(x),
		Where(LessThan(x, ZerosLike(x)),
			leftSide,
			Where(LessThan(x, halfBase), rightSide, OnesLike(x)),
		),
	)
}
