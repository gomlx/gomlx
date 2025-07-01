package kan

import (
	"math"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"k8s.io/klog/v2"
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

	// ParamDiscreteSoftnessSchedule allows one to have a dynamic softness schedule during training.
	//
	// Current valid values are "none", "cosine", "linear", "exponential". Default is "none".
	ParamDiscreteSoftnessSchedule = "kan_discrete_softness_schedule"

	// ParamDiscreteSoftnessScheduleMin sets the minimum softness for the softness schedule.
	//
	// Default is 1e-5
	ParamDiscreteSoftnessScheduleMin = "kan_discrete_min_softness"

	// ParamDiscreteSplitPointsTrainable is a boolean that indicates whether the split points are trainable and can move around.
	//
	// Notice that this is unstable (NaN) for small values of softness -- or at the later stages or training if
	// using a softness schedule -- because the gradients go to infinity close to the split points.
	// The recommend way of using it is training in multiple stages, with a fixed softness and trainable splits
	// first, and later freezing the split points (see ParamDiscreteSplitPointsFrozen).
	//
	// Default is false.
	ParamDiscreteSplitPointsTrainable = "kan_discrete_splits_trainable"

	// ParamDiscreteSplitPointsFrozen is a boolean that indicates whether the trained split-points should be frozen.
	//
	// This only has an effect if the split points are set to trainable (see ParamDiscreteSplitPointsTrainable).
	// This is useful if after training the split points, one is going to use a softness schedule, and the
	// softness is going to get very small (which leads to NaNs in the split points).
	ParamDiscreteSplitPointsFrozen = "kan_discrete_splits_frozen"

	// ParamDiscreteSplitsMargin is the minimum distance between consecutive split points.
	// Only applies if the split points are trainable, in which case they are always projected to
	// monotonicity with this minimum margin -- it can be set to 0.0, in which case split points can merge.
	// Default is 0.01.
	ParamDiscreteSplitsMargin = "kan_discrete_splits_margin"
)

// discreteConfig holds the configuration exclusive for Discrete-KANs.
type discreteConfig struct {
	perturbation                            PerturbationType
	softness                                float64
	softnessSchedule                        SoftnessScheduleType
	minSoftness                             float64
	splitPointsTrainable, splitPointsFrozen bool
	rangeMin, rangeMax                      float64
	splitPointInitialValue                  *tensors.Tensor
}

// initDiscrete initializes the default values for Discrete-KANs based on context.
func (c *Config) initDiscrete(ctx *context.Context) {
	c.discrete.softness = context.GetParamOr(ctx, ParamDiscreteSoftness, 0.1)
	c.discrete.minSoftness = context.GetParamOr(ctx, ParamDiscreteSoftnessScheduleMin, 1e-5)
	c.discrete.splitPointsTrainable = context.GetParamOr(ctx, ParamDiscreteSplitPointsTrainable, false)
	c.discrete.splitPointsFrozen = context.GetParamOr(ctx, ParamDiscreteSplitPointsFrozen, false)
	c.DiscreteInputRange(-1, 1)

	perturbationStr := context.GetParamOr(ctx, ParamDiscretePerturbation, "triangular")
	switch perturbationStr {
	case "", "triangular":
		c.discrete.perturbation = PerturbationTriangular
	case "normal":
		c.discrete.perturbation = PerturbationNormal
	default:
		exceptions.Panicf("Invalid Discrete-KAN perturbation given by context[%q]: %q -- valid values are "+
			"\"triangular\", \"normal\"", ParamDiscretePerturbation, perturbationStr)
	}

	softnessScheduleStr := context.GetParamOr(ctx, ParamDiscreteSoftnessSchedule, SoftnessScheduleNone.String())
	var err error
	c.discrete.softnessSchedule, err = SoftnessScheduleTypeString(softnessScheduleStr)
	if err != nil {
		values := xslices.Iota(SoftnessScheduleNone, int(SoftnessScheduleLast))
		valuesStr := xslices.Map(values, func(v SoftnessScheduleType) string { return v.String() })
		exceptions.Panicf("Invalid Discrete-KAN hyperparameter %q: %q -- valid values are %q",
			ParamDiscreteSoftnessSchedule, softnessScheduleStr, valuesStr)
	}

}

// Discrete configures the KAN to use a "piecewise-constant" functions (as opposed to splines) to model \phi(x),
// the univariate function used in the pape, and set the number of control points to use for the function.
//
// Discrete-KAN is a variation of KAN that uses a piecewise-constant function (PCF for short) as
// its univariate function family -- as opposed to the original spline functions.
//
// The number of control points used can be set up with Config.Para
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
// Notice that the regularizer set with Config.Regularizer applies only to the "control points".
//
// [1] Learning Representations for Axis-Aligned Decision Forests through Input Perturbation -
// Sebastian Bruch, Jan Pfeifer, Mathieu Guillame-Bert - https://arxiv.org/pdf/2007.14761
func (c *Config) Discrete() *Config {
	c.useDiscrete = true
	return c
}

// DiscreteSoftness sets how much softness to use during training.
// If set to 0 softness is disabled.
//
// The default is 0.1, and it can be set with the hyperparameter ParamDiscreteSoftness ("kan_discrete_softness").
func (c *Config) DiscreteSoftness(softness float64) *Config {
	c.discrete.softness = softness
	return c
}

// SoftnessScheduleType describes the type of schedule of the softness for Discrete-KAN.
type SoftnessScheduleType int

const (
	// SoftnessScheduleNone indicates no schedule, that is a constant softness during training.
	SoftnessScheduleNone SoftnessScheduleType = iota

	// SoftnessScheduleCosine uses softness*(Cos(π.progress)+1.0)/2.0 + epsilon).
	SoftnessScheduleCosine

	// SoftnessScheduleLinear uses softness*((1.0-progress) + epsilon).
	SoftnessScheduleLinear

	// SoftnessScheduleExponential uses softness*e^(-lambda*progress), where lambda = log(100), so it is 1/100 of the
	// original softness at the end of the training.
	SoftnessScheduleExponential

	// SoftnessScheduleLast indicates the last valid enum value -> keep this last.
	SoftnessScheduleLast
)

//go:generate enumer -type=SoftnessScheduleType -trimprefix=SoftnessSchedule -transform=snake -values -text -json -yaml discrete.go

// DiscreteSoftnessScheduleType configures the type of schedule of the softness for Discrete-KAN.
//
// It defaults to SoftnessScheduleNone (0)
func (c *Config) DiscreteSoftnessScheduleType(schedule SoftnessScheduleType) *Config {
	c.discrete.softnessSchedule = schedule
	return c
}

// DiscreteSplitsTrainable defines whether the split points are trainable.
//
// Notice that this is unstable (NaN) for small values of softness -- or at the later stages or training if
// using a softness schedule -- because the gradients go to infinity close to the split points.
// The recommend way of using it is training in multiple stages, with a fixed softness and trainable splits
// first, and later freezing the split points (see ParamDiscreteSplitPointsFrozen).
//
// Default is false, and can be set with the context hyperparameter ParamDiscreteSplitPointsTrainable.
func (c *Config) DiscreteSplitsTrainable(trainable bool) *Config {
	c.discrete.splitPointsTrainable = trainable
	return c
}

// DiscreteSplitsFrozen is a boolean that indicates whether the trained split-points should be frozen.
//
// This only has an effect if the split points are set to trainable (see ParamDiscreteSplitPointsTrainable).
// This is useful if after training the split points, one is going to use a softness schedule, and the
// softness is going to get very small (which leads to NaNs in the split points).
//
// Default is false and can be set with the context hyperparameter ParamDiscreteSplitPointsFrozen.
func (c *Config) DiscreteSplitsFrozen(frozen bool) *Config {
	c.discrete.splitPointsFrozen = frozen
	return c
}

// DiscreteInputRange defines how to initialize the split-points: they are uniformly taken from the given
// range, with the first split point at rangeMin, and the last split point at rangeMax.
// It requires rangeMax > rangeMin.
//
// The default range is [-1, 1.0]. See also DiscreteInitialSplitPoints.
//
// Enable Discrete-KAN with Discrete(true).
func (c *Config) DiscreteInputRange(rangeMin, rangeMax float64) *Config {
	if rangeMin >= rangeMax {
		exceptions.Panicf("invalid range [%g, %g] given to DiscreteInputRange: it must observe rangeMax > rangeMin",
			rangeMin, rangeMax)
	}
	c.discrete.rangeMin, c.discrete.rangeMax = rangeMin, rangeMax
	return c
}

// DiscreteInitialSplitPoints sets the initialValues to the split-points. initialValues should be shaped [n-1],
// where n is the number of control points.
// This also sets the number of control points to n.
//
// The default is to initialize the split points as a uniform range. See also DiscreteInputRange.
//
// Enable Discrete-KAN with Discrete(true).
func (c *Config) DiscreteInitialSplitPoints(initialValues *tensors.Tensor) *Config {
	if initialValues.Rank() != 1 {
		exceptions.Panicf("Split points for Discrete-KAN must be rank-1, got %s instead", initialValues.Shape())
	}
	c.discrete.splitPointInitialValue = initialValues
	c.NumControlPoints(initialValues.Shape().Dim(0) + 1)
	return c
}

// Layer implements one Discrete-KAN layer. x is expected to be shaped [batchSize, numInputNodes].
func (c *Config) discreteLayer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	g := x.Graph()
	dtype := x.DType()
	residual := x
	batchSize := x.Shape().Dimensions[0]
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	numInputGroups := numInputNodes
	if c.inputGroupSize > 1 {
		if numInputNodes%c.inputGroupSize != 0 {
			exceptions.Panicf("KAN configured with input group size %d, but input (shape %s) last dimension %d is not divisible by %d",
				c.inputGroupSize, c.input.Shape(), numInputNodes, c.inputGroupSize)
		}
		numInputGroups = numInputNodes / c.inputGroupSize
		x = Reshape(x, -1, numInputGroups, c.inputGroupSize)
	}

	if klog.V(2).Enabled() {
		klog.Infof("kan discreteLayer (%s): ~ %d x %d x %d = %d weights, splits_trainable=%v\n",
			ctx.Scope(), c.numControlPoints, numInputGroups, numOutputNodes,
			c.numControlPoints*numInputGroups*numOutputNodes, c.discrete.splitPointsTrainable)
	}

	// Create control points for piecewise-constant function (PCF)
	controlPointsInitializer := func(graph *Graph, shape shapes.Shape) *Node {
		// Values initialized from -1.0 to 1.0 linearly.
		v := Iota(graph, shape, -1)
		v = MulScalar(v, 2.0/float64(shape.Dim(-1)-1))
		v = AddScalar(v, -1.0)
		// Apply a random constant.
		slopeShape := shape.Clone()
		slopeShape.Dimensions[slopeShape.Rank()-1] = 1 // The same multiplier for all control points, so it's linear.
		slope := ctx.RandomNormal(graph, slopeShape)
		slope = MulScalar(slope, 0.1)
		v = Mul(v, slope)
		return v
	}
	controlPointsVar := ctx.WithInitializer(controlPointsInitializer).
		VariableWithShape("kan_discrete_control_points", shapes.Make(dtype, numOutputNodes, numInputGroups, c.numControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)

	// splitPoints: start from values distributed between -1 and 1, and let them be trained.
	var splitPoints *Node
	initialSplitPointsT := c.discrete.splitPointInitialValue
	if initialSplitPointsT == nil {
		keys := make([]float64, c.numControlPoints-1)
		if c.numControlPoints > 2 {
			// Initialize split points uniformly from rangeMin (-1.0) to rangeMax (1.0)
			rangeMin, rangeMax := c.discrete.rangeMin, c.discrete.rangeMax
			rangeLen := rangeMax - rangeMin
			for ii := range keys {
				keys[ii] = rangeLen*float64(ii)/float64(c.numControlPoints-2) + rangeMin
			}
		}
		initialSplitPointsT = tensors.FromValue(keys)
	}
	if c.discrete.splitPointsTrainable {

		// Trainable split points: one per input.
		// * We could also make it learn one per output ... at the cost of more parameters.
		splitPointsVar := ctx.WithInitializer(initializers.BroadcastTensorToShape(initialSplitPointsT)).
			VariableWithShape("kan_discrete_split_points", shapes.Make(dtype, 1, numInputGroups, c.numControlPoints-1))
		if c.discrete.splitPointsFrozen {
			splitPointsVar.Trainable = false
		}
		splitPoints = splitPointsVar.ValueGraph(g)

		// At the end of each training step, project splitPoints back to monotonically increasing values, so they
		// don't overlap.
		train.AddPerStepUpdateGraphFn(ctx.In("kan_discrete_split_points_projection"), g, func(ctx *context.Context, g *Graph) {
			splitPoints := splitPointsVar.ValueGraph(g)
			margin := Scalar(g, splitPoints.DType(), context.GetParamOr(ctx, ParamDiscreteSplitsMargin, 0.01))
			splitPoints = optimizers.MonotonicProjection(splitPoints, margin, -1)
			splitPointsVar.SetValueGraph(splitPoints)
		})

	} else {
		// Fixed split points.
		splitPoints = ConstCachedTensor(g, initialSplitPointsT)
		splitPoints = ConvertDType(splitPoints, dtype)
	}

	var output *Node
	if c.discrete.softness <= 0 || !ctx.IsTraining(g) {
		output = PiecewiseConstantFunction(x, controlPoints, splitPoints)
	} else {
		softness := c.scheduledSoftness(ctx, Scalar(g, dtype, c.discrete.softness))
		output = PiecewiseConstantFunctionWithInputPerturbation(x, controlPoints, splitPoints, c.discrete.perturbation, softness)
	}
	output.AssertDims(batchSize, numOutputNodes, numInputNodes) // Shape=[batch, outputs, inputs]

	// Reduce the inputs to get the outputs: we prefer the mean (and not sum) because if the number of inputs is large
	// with ReduceSum we would get large numbers (and gradients) that are harder for the gradient descent to learn.
	// In particular for multiple hidden-layers: there is a geometric growth of the values per number of layers.
	if c.useMean {
		output = ReduceMean(output, -1)
	} else {
		output = ReduceSum(output, -1)
	}
	output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]

	if c.useResidual && numInputNodes == numOutputNodes {
		output = Add(output, residual)
	}
	return output
}

// scheduledSoftness adjust the base softness according to the configured schedule.
func (c *Config) scheduledSoftness(ctx *context.Context, base *Node) *Node {
	if c.discrete.softnessSchedule == SoftnessScheduleNone {
		return base
	}

	g := base.Graph()
	dtype := base.DType()
	rootCtx := ctx.InAbsPath(context.RootScope)

	// Calculate scheduleTime: from 0.0 to 1.0 depending on training progress.
	globalStep := ConvertDType(optimizers.GetGlobalStepVar(rootCtx).ValueGraph(g), dtypes.Float32)
	lastStep := ConvertDType(train.GetTrainLastStepVar(rootCtx).ValueGraph(g), dtypes.Float32)
	// scheduleTime will be at most 1.0, if for some reason globalStep >
	scheduleTime := MinScalar(Div(globalStep, MaxScalar(lastStep, 1.0)), 1.0)
	zero := ZerosLike(lastStep)
	scheduleTime = Where(LessThan(lastStep, zero), zero, scheduleTime)
	epsilon := c.discrete.minSoftness

	switch c.discrete.softnessSchedule {
	case SoftnessScheduleLinear:
		schedule := Mul(base, OneMinus(scheduleTime))
		schedule = ConvertDType(schedule, dtype)
		schedule = MaxScalar(schedule, epsilon)
		return schedule

	case SoftnessScheduleExponential:
		var lambda = math.Log(100) // That means at the end of the scheduleTime (==1) the schedule will be 1/100 of the original value.
		schedule := Exp(MulScalar(Neg(scheduleTime), lambda))
		return MaxScalar(Mul(base, ConvertDType(schedule, dtype)), epsilon)

	case SoftnessScheduleCosine:
		// cosineSchedule = Cos(π.scheduleTime)+1.0)/2.0 + epsilon
		cosineSchedule := Cos(MulScalar(scheduleTime, math.Pi))
		cosineSchedule = DivScalar(OnePlus(cosineSchedule), 2.0)
		cosineSchedule = AddScalar(cosineSchedule, epsilon)
		return Mul(base, ConvertDType(cosineSchedule, dtype))
	default:
		exceptions.Panicf("invalid Discrete-KAN softness schedule: %s", c.discrete.softnessSchedule)
	}
	return nil
}

// PiecewiseConstantFunction (PCF) generates a PCF output for a cross of numInputNodes x numOutputNodes, defined
// as the dimensions of its inputs, as follows:
//
//   - input to be transformed, shaped [batchSize, numInputNodes]. If using input grouping, the input should be shaped
//     [batchSize, numInputGroups, inputGroupSize], otherwise assume numInputGroups == numInputNodes.
//   - controlPoints are the output values of the PCFs, and should be shaped [numOutputNodes, numInputGroups, NumControlPoints]
//   - splitPoints are the splitting values of the inputs, shaped [numOutputNodes or 1, numInputGroups or 1, NumControlPoints-1],
//     but if the first or the second axes are set to 1, they are broadcast accordingly.
//
// The output will be shaped [batchSize, numOutputPoints, numInputNodes].
// Presumably, the caller will graph.ReduceMean (or ReduceSum) on the last axis (after residual value is added) for a
// shape [batchSize, numOutputPoints].
func PiecewiseConstantFunction(input, controlPoints, splitPoints *Node) *Node {
	g := input.Graph()

	// Expand missing dimensions for scalar evaluation.
	inputRank := input.Rank()
	if inputRank == 0 {
		input = Reshape(input, 1, 1, 1) // batchSize=1, numInputNodes=1
	} else if inputRank == 1 {
		input = Reshape(input, -1, 1) // batchSize=?, numInputNodes=1
	} else if inputRank == 2 {
		input = InsertAxes(input, -1) // Reshape it to [batchSize, numInputGroups, inputGroupSize].
	} else if inputRank != 3 {
		exceptions.Panicf("invalid input shape %s for PiecewiseConstantFunction, expected rank <= 3", input.Shape())
	}

	// Check that control points and split points are compatible.
	if controlPoints.Rank() == 1 {
		controlPoints = InsertAxes(controlPoints, 0, 0)
	}
	if splitPoints.Rank() == 1 {
		splitPoints = InsertAxes(splitPoints, 0, 0)
	}
	numControlPoints := controlPoints.Shape().Dimensions[2]
	if splitPoints.Shape().Dim(-1) != numControlPoints-1 {
		exceptions.Panicf("number of split points (last dimension) must be equal to number of control points - 1, got "+
			"splitPoints shape %s and controlPoints shape %s", splitPoints.Shape(), controlPoints.Shape())
	}

	// Define the various dimensions.
	batchSize := input.Shape().Dimensions[0]
	numInputGroups := input.Shape().Dimensions[1]
	inputGroupSize := input.Shape().Dimensions[2]
	numInputNodes := numInputGroups * inputGroupSize
	numOutputNodes := controlPoints.Shape().Dimensions[0]

	// Standard PCF, no softening.
	// Expand shapes to [batchSize, numOutputs, numInputGroups, inputGroupSize, numPoints], each can be set to one
	// if not present in the expanded tensors.
	expandedInputs := InsertAxes(input, 1, -1)            // Add output dimension and num points axes.
	expandedSplitPoints := InsertAxes(splitPoints, 0, -2) // Add batch and inputGroupSize axes.

	// Find the control point to use, by comparing the inputs and split points.
	toTheLeft := LessThanTotalOrder(expandedInputs,
		GrowRight(expandedSplitPoints, -1, 1, math.Inf(1)))
	toTheRight := GreaterOrEqualTotalOrder(expandedInputs,
		GrowLeft(expandedSplitPoints, -1, 1, math.Inf(-1)))
	controlPicks := ArgMax(LogicalAnd(toTheLeft, toTheRight), -1, dtypes.Int16)
	controlPicks.AssertDims(batchSize, -1 /* numOutputNodes */, numInputGroups, inputGroupSize)

	// Makes sure we broadcast on the output dimension, if needed.
	controlPicks = BroadcastToDims(controlPicks, batchSize, numOutputNodes, numInputGroups, inputGroupSize)
	controlPicks = InsertAxes(controlPicks, -1)
	controlPicks = Concatenate([]*Node{
		Iota(g, controlPicks.Shape(), 1), // output node index.
		Iota(g, controlPicks.Shape(), 2), // input group index.
		controlPicks,
	}, -1)
	controlPicks = StopGradient(controlPicks) // No gradient with respect to the picks.

	// controlPoints is shaped: [numOutputNodes, numInputGroups, NumControlPoints]
	output := Gather(controlPoints, controlPicks)
	output.AssertDims(batchSize, numOutputNodes, numInputGroups, inputGroupSize)

	// Join the input groups back into numInputNodes.
	output = Reshape(output, batchSize, numOutputNodes, numInputNodes)

	// Final touch: special case when input was a scalar or rank-1.
	if inputRank == 0 && numOutputNodes == 1 {
		output = Reshape(output) // Back to scalar.
	} else if inputRank == 1 && numOutputNodes == 1 {
		output = Reshape(output, -1) // Back to a vector.
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
// adds a "perturbation" of the inputs by a noise distribution of the value, controlled by smoothness.
//
// The shapes and inputs are the same as PiecewiseConstantFunction, with the added perturbation type and smoothness parameters.
// The smoothness should be a scalar with suggested values from 0 to 1.0.
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
	softness.AssertScalar()

	// Expand missing dimensions for scalar evaluation.
	inputRank := input.Rank()
	if inputRank == 0 {
		input = Reshape(input, 1, 1, 1) // batchSize=1, numInputNodes=1
	} else if inputRank == 1 {
		input = Reshape(input, -1, 1) // batchSize=?, numInputNodes=1
	} else if inputRank == 2 {
		input = InsertAxes(input, -1) // Reshape it to [batchSize, numInputGroups, inputGroupSize].
	} else if inputRank != 3 {
		exceptions.Panicf("invalid input shape %s for PiecewiseConstantFunction, expected rank <= 3", input.Shape())
	}

	// Check that control points and split points are compatible.
	if controlPoints.Rank() == 1 {
		controlPoints = InsertAxes(controlPoints, 0, 0)
	}
	if splitPoints.Rank() == 1 {
		splitPoints = InsertAxes(splitPoints, 0, 0)
	}
	numControlPoints := controlPoints.Shape().Dimensions[2]
	if splitPoints.Shape().Dim(-1) != numControlPoints-1 {
		exceptions.Panicf("number of split points (last dimension) must be equal to number of control points - 1, got "+
			"splitPoints shape %s and controlPoints shape %s", splitPoints.Shape(), controlPoints.Shape())
	}

	// Define the various dimensions.
	batchSize := input.Shape().Dimensions[0]
	numInputGroups := input.Shape().Dimensions[1]
	inputGroupSize := input.Shape().Dimensions[2]
	numInputNodes := numInputGroups * inputGroupSize
	numOutputNodes := controlPoints.Shape().Dimensions[0]

	// Calculate the distribution base
	distributionBase := Sub(
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(numControlPoints-2)),
		Slice(splitPoints, AxisRange(), AxisRange(), AxisElem(0)),
	)

	// Expand shapes to [batchSize, numOutputs, numInputGroups, inputGroupSize, numPoints], each can be set to one
	// if not present in the expanded tensors.
	expandedInputs := InsertAxes(input, 1, -1)            // Add output dimension and num points axes.
	expandedSplitPoints := InsertAxes(splitPoints, 0, -2) // Add batch and inputGroupSize axes.

	// Calculate cumulative distribution function for all split nodes.
	cdfsPoints := Sub(expandedSplitPoints, expandedInputs)
	var cdfs *Node
	switch perturbation {
	case PerturbationTriangular:
		triangleHalfBase := Mul(distributionBase, softness)
		triangleHalfBase = InsertAxes(triangleHalfBase, 0, -2) // [batchSize=1, numOutputs, numInputGroups, inputGroupSize=1, numPoints]
		cdfs = triangleDistributionCDF(cdfsPoints, triangleHalfBase)
	case PerturbationNormal:
		stddev := Mul(distributionBase, softness)
		stddev = InsertAxes(stddev, 0, -2) // [batchSize=1, numOutputs, numInputGroups, inputGroupSize=1, numPoints]
		cdfs = NormalDistributionCDF(cdfsPoints, stddev)
	}

	// The sum of the integral parts is control points multiplied by the difference between the split points (left
	// and right)
	leftCDF := GrowLeft(cdfs, -1, 1, 0)                       // Prepend the CDF for -inf to the left == 0
	rightCDF := GrowRight(cdfs, -1, 1, 1)                     // Append the CDF for +inf to the right == 1
	expandedControlPoints := InsertAxes(controlPoints, 0, -2) // Shape [1 (batch), numOutputNodes, numInputGroups, 1(inputGroup), NumControlPoints]
	weights := Sub(rightCDF, leftCDF)
	//weights.SetLogged("weights")
	termsOfSum := Mul(weights, expandedControlPoints)
	//termsOfSum.SetLogged("termsOfSum")

	// Reduce control points dimensions. Output is [batchSize, numOutputNodes, numInputGroups, inputGroupSize].
	output := ReduceSum(termsOfSum, -1)
	output.AssertDims(batchSize, numOutputNodes, numInputGroups, inputGroupSize)

	// Join the input groups back into numInputNodes.
	output = Reshape(output, batchSize, numOutputNodes, numInputNodes)

	// Final touch: special case when input was a scalar or rank-1.
	if inputRank == 0 && numOutputNodes == 1 {
		output = Reshape(output) // Back to scalar.
	} else if inputRank == 1 && numOutputNodes == 1 {
		output = Reshape(output, -1) // Back to a vector.
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
