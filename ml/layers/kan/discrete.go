package kan

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
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

	// ParamDiscretePerturbation selects the type of input perturbation used to make the Discrete-KAN differentiable
	// during training.
	//
	// It can take 2 values: "triangular", "normal". The default is "triangular".
	ParamDiscretePerturbation = "kan_discrete_perturbation"

	// ParamDiscreteSoftness indicates how much to soften the PCF (piecewise constant functions) during training,
	// and by how much.
	//
	// For triangular perturbation, this is a factor that multiplied by the distance from the first and last
	// split points, defines the base of the triangle.
	//
	// For normal perturbation, this is multiplied by the distance from the first and last split points to
	// define the standard deviation.
	//
	// The default value is 0.1.
	ParamDiscreteSoftness = "kan_discrete_softness"

	// ParamDiscreteNumControlPoints is the number of points to use (and learn) in the piecewise-constant function
	// for DiscreteKAN. Default is 6.
	ParamDiscreteNumControlPoints = "kan_discrete_num_points"

	// ParamDiscreteSplitPointsTrainable indicates whether the split points are trainable and can move around.
	// Default is true.
	ParamDiscreteSplitPointsTrainable = "kan_discrete_splits_trainable"

	// ParamDiscreteSplitsMargin is the minimum distance between consecutive split points.
	// Only applies if the split points are trainable, in which case they are always projected to
	// monotonicity with this minimum margin -- it can be set to 0.0, in which case split points can merge.
	// Default is 0.01.
	ParamDiscreteSplitsMargin = "kan_discrete_splits_margin"
)

// Discrete configures the KAN to use a "piecewise-constant" functions (as opposed to splines) to model \phi(x),
// the univariate function used in the pape, and set the number of control points to use for the function.
func (c *Config) Discrete() *Config {
	c.useDiscrete = true
	return c
}

// DiscreteNumPoints sets the number of control points to use for the piecewise constant function.
// numControlPoints must be greater or equal to 2, and it defaults to 6 and can also be set by using the
// hyperparameter ParamDiscreteNumControlPoints ("kan_discrete_num_points").
func (c *Config) DiscreteNumPoints(numControlPoints int) *Config {
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

// DiscreteSplitsTrainable defines whether the split points are trainable.
// Default is true.
func (c *Config) DiscreteSplitsTrainable(trainable bool) *Config {
	c.discreteSplitPointsTrainable = trainable
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
		klog.Infof("kan discreteLayer (%s): (%d+2) x %d x %d = %d weights, splits_trainable=%v\n",
			ctx.Scope(), c.discreteControlPoints, numInputNodes, numOutputNodes,
			(c.discreteControlPoints)*numInputNodes*numOutputNodes, c.discreteSplitPointsTrainable)
	}

	// Apply piecewise-constant function (PCF)
	initialSeed := context.GetParamOr(ctx, initializers.ParamInitialSeed, initializers.NoSeed)
	controlPointsInitializer := func(graph *Graph, shape shapes.Shape) *Node {
		// Values initialized from -1.0 to 1.0 linearly.
		v := Iota(graph, shape, -1)
		v = MulScalar(v, 2.0/float64(shape.Dim(-1)-1))
		v = AddScalar(v, -1.0)
		// Apply a random constant.
		var slope *Node
		slopeShape := shape.Clone()
		slopeShape.Dimensions[slopeShape.Rank()-1] = 1 // Same multiplier for all control points:
		initializers.UseRngState(graph, initialSeed, func(rngState *Node) (newRngState *Node) {
			newRngState, slope = RandomNormal(rngState, shape)
			return newRngState
		})
		slope = MulScalar(slope, 0.1)
		v = Mul(v, slope)
		return v
	}
	controlPointsVar := ctx.WithInitializer(controlPointsInitializer).
		VariableWithShape("discrete_control_points", shapes.Make(dtype, numOutputNodes, numInputNodes, c.discreteControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)

	// splitPoints: start from values distributed between -1 and 1, and let them be trained.
	var splitPoints *Node
	keys := make([]float64, c.discreteControlPoints-1)
	if c.discreteControlPoints > 2 {
		// Initialize split points uniformly from -1.0 to 1.0
		for ii := range keys {
			keys[ii] = 2.0*float64(ii)/float64(c.discreteControlPoints-2) - 1.0
		}
	}
	keysT := tensors.FromValue(keys)
	if c.discreteSplitPointsTrainable {
		// Trainable split points:
		splitPointsVar := ctx.WithInitializer(initializers.BroadcastTensorToShape(keysT)).
			VariableWithShape("split_points", shapes.Make(dtype, 1, numInputNodes, c.discreteControlPoints-1))
		splitPoints = splitPointsVar.ValueGraph(g)

		// At the end of each training step, project splitPoints back to monotonically increasing values, so they
		// don't overlap.
		train.AddPerStepUpdateGraphFn(ctx.In("split_points_projection"), g, func(ctx *context.Context, g *Graph) {
			splitPoints := splitPointsVar.ValueGraph(g)
			margin := Scalar(g, splitPoints.DType(), context.GetParamOr(ctx, ParamDiscreteSplitsMargin, 0.01))
			splitPoints = optimizers.MonotonicProjection(splitPoints, margin, -1)
			splitPointsVar.SetValueGraph(splitPoints)
		})

	} else {
		// Fixed split points.
		splitPoints = ConstCachedTensor(g, keysT)
		splitPoints = ConvertDType(splitPoints, dtype)
	}

	var output *Node
	if c.discreteSoftness <= 0 || !ctx.IsTraining(g) {
		output = PiecewiseConstantFunction(x, controlPoints, splitPoints)
		// The version with perturbation is faster than the original PCF !?
		//output = PiecewiseConstantFunctionWithInputPerturbation(x, controlPoints, splitPoints, PerturbationTriangular, ScalarZero(g, dtype))
	} else {
		softnessConst := Scalar(g, dtype, c.discreteSoftness)
		output = PiecewiseConstantFunctionWithInputPerturbation(x, controlPoints, splitPoints, c.discretePerturbation, softnessConst)
	}
	output.AssertDims(batchSize, numOutputNodes, numInputNodes) // Shape=[batch, outputs, inputs]

	// ReduceMean the inputs to get the outputs: we use the mean (and not sum) because if the number of inputs is large
	// with ReduceSum we would get large numbers (and gradients) that are harder for the gradient descent to learn.
	// In particular for multiple hidden-layers: there is a geometric growth of the values per number of layers.
	output = ReduceMean(output, -1)
	output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]

	if c.useResidual && numInputNodes == numOutputNodes {
		output = Add(output, residual)
	}
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
// Presumably, the caller will graph.ReduceMean (or ReduceSum) on the last axis (after residual value is added) for a shape [batchSize, numOutputPoints].
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
	// Makes sure we broadcast on the output dimension, if needed.
	controlPicks = BroadcastToDims(controlPicks, batchSize, numOutputNodes, numInputNodes)
	controlPicks = ExpandDims(controlPicks, -1)
	controlPicks = Concatenate([]*Node{
		Iota(g, controlPicks.Shape(), 1), // output node index.
		Iota(g, controlPicks.Shape(), 2), // input node index.
		controlPicks,
	}, -1)
	controlPicks = StopGradient(controlPicks) // No gradient with respect to the picks.

	output := Gather(controlPoints, controlPicks)
	output.AssertDims(batchSize, numOutputNodes, numInputNodes)
	if inputRank == 0 {
		output = Reshape(output) // Back to scalar.
	} else if inputRank == 1 {
		output = Reshape(output, -1) // Back to scalar.
	}
	return output
}

// PerturbationType used by PiecewiseConstantFunctionWithInputPerturbation
type PerturbationType int

const (
	PerturbationTriangular PerturbationType = iota
	PerturbationNormal
)

//go:generate enumer -type=PerturbationType -trimprefix=Perturbation -transform=snake -values -text -json -yaml discrete.go

// PiecewiseConstantFunctionWithInputPerturbation works similarly to PiecewiseConstantFunction, but
// adds a "perturbation" of the inputs by a triangular distribution of the value, controlled by smoothness.
//
// The shapes and inputs are the same as PiecewiseConstantFunction, with the added smoothness parameter
// that should be a scalar with suggested values from 0 to 1.0.
//
// The smoothness softens the function by perturbing the input using a triangular (or normal) distribution,
// whose base is given by softness * 2 * (splitPoint[-1] - splitPoint[0]).
// If softness is 0, the function is back to being piece-wise constant.
//
// The softening makes it differentiable with respect to the splitPoints, and hence can be used for training.
// One can control the softness as a form of annealing on the split points. As it reaches 0, the split points are
// no longer changed (only the control points).
//
// The output will be shaped [batchSize, numOutputPoints, numInputNodes].
// Presumably, the caller will graph.ReduceMean (or ReduceSum) on the last axis (after residual value is added) for a shape [batchSize, numOutputPoints].
func PiecewiseConstantFunctionWithInputPerturbation(input, controlPoints, splitPoints *Node, perturbation PerturbationType, softness *Node) *Node {
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

	// Calculate the distribution base
	distributionBase := Sub(
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(numControlPoints-2)),
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(0)),
	)

	// Calculate cumulative distribution function for all split nodes.
	expandedInputs := ExpandDims(input, -2, -1)       // Shape [batchSize, numInputNodes, 1, 1]
	expandedSplitPoints := ExpandDims(splitPoints, 0) // Shape [1, numOutputNodes, numInputNodes, NumControlPoints-1]
	cdfsPoints := Sub(expandedSplitPoints, expandedInputs)
	var cdfs *Node
	switch perturbation {
	case PerturbationTriangular:
		triangleHalfBase := Mul(distributionBase, softness)
		cdfs = triangleDistributionCDF(cdfsPoints, ExpandDims(triangleHalfBase, 0))
	case PerturbationNormal:
		stddev := Mul(distributionBase, softness)
		cdfs = NormalDistributionCDF(cdfsPoints, ExpandDims(stddev, 0))
	}

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

// NormalDistributionCDF calculates the CDF for a normal distribution centered in zero and with the given
// standard deviation.
func NormalDistributionCDF(x, stddev *Node) *Node {
	x = Div(x, stddev)
	cdf := Erf(MulScalar(x, 1.0/math.Sqrt(2)))
	cdf = DivScalar(OnePlus(cdf), 2)
	return cdf
}
