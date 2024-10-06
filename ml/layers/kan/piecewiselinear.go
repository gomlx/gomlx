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
)

// PWL-KAN (Piecewise Linear KAN) uses learnable piecewise-linear functions as univariate functions.

var (
	// ParamPiecewiseLinear indicates whether to use PWL-KAN as the univariate function to learn.
	ParamPiecewiseLinear = "kan_pwl"

	// ParamPWLExtrapolate indicates whether to extrapolate linearly outside the range of the split points.
	// If set to false, the function becomes a constant at the extremes.
	ParamPWLExtrapolate = "kan_pwl_extrapolate"

	// ParamPWLSplitPointsTrainable is a boolean that indicates whether the split points are trainable and can move around.
	//
	// Default is false.
	ParamPWLSplitPointsTrainable = "kan_pwl_splits_trainable"

	// ParamPWLSplitPointsFrozen is a boolean that indicates whether the trained split-points should be frozen.
	//
	// This only has an effect if the split points are set to trainable (see ParamPWLSplitPointsTrainable).
	// This is useful if after training the split points, one is going to use a softness schedule, and the
	// softness is going to get very small (which leads to NaNs in the split points).
	ParamPWLSplitPointsFrozen = "kan_pwl_splits_frozen"

	// ParamPWLSplitsMargin is the minimum distance between consecutive split points.
	// Only applies if the split points are trainable, in which case they are always projected to
	// monotonicity with this minimum margin -- it can be set to 0.0, in which case split points can merge.
	// Default is 0.01.
	ParamPWLSplitsMargin = "kan_pwl_splits_margin"
)

// pwlConfig holds PWL-KAN specific configuration.
type pwlConfig struct {
	linearExtrapolation                     bool
	splitPointsTrainable, splitPointsFrozen bool
	rangeMin, rangeMax                      float64
	splitPointInitialValue                  *tensors.Tensor
	splitPointsMargin                       float64
}

// initPiecewiseLinear initializes the default values for PWL-KANs based on context.
func (c *Config) initPiecewiseLinear(ctx *context.Context) {
	c.pwl.linearExtrapolation = context.GetParamOr(ctx, ParamPWLExtrapolate, false)
	c.pwl.splitPointsTrainable = context.GetParamOr(ctx, ParamPWLSplitPointsTrainable, false)
	c.pwl.splitPointsFrozen = context.GetParamOr(ctx, ParamPWLSplitPointsFrozen, false)
	c.PWLInputRange(-1, 1)
	c.pwl.splitPointsMargin = context.GetParamOr(ctx, ParamPWLSplitsMargin, 0.01)
}

// PiecewiseLinear configures for a PWL-KAN (PieceWise-Linear), as opposed to the default BSpline.
//
// This can also be set using the hyperparameter ParamPiecewiseLinear.
func (c *Config) PiecewiseLinear() *Config {
	c.usePWL = true
	return c
}

// PWLInputRange defines how to initialize the split-points: they are uniformly taken from the given
// range, with the first split point at rangeMin, and the last split point at rangeMax.
// It requires rangeMax > rangeMin.
//
// The default range is [-1, 1.0]. See also PWLInitialSplitPoints.
//
// Enable PWL-KAN with PWL(true).
func (c *Config) PWLInputRange(rangeMin, rangeMax float64) *Config {
	if rangeMin >= rangeMax {
		exceptions.Panicf("invalid range [%g, %g] given to PWLInputRange: it must observe rangeMax > rangeMin",
			rangeMin, rangeMax)
	}
	c.pwl.rangeMin, c.pwl.rangeMax = rangeMin, rangeMax
	return c
}

// PWLInitialSplitPoints sets the initialValues shaped [n] to the split-points.
// This also sets the number of control points to n, the dimension of the tensor.
//
// The default is to initialize the split points as a uniform range. See also PWLInputRange.
//
// Enable PWL-KAN with Config.Piecewise(true).
func (c *Config) PWLInitialSplitPoints(initialValues *tensors.Tensor) *Config {
	if initialValues.Rank() != 1 {
		exceptions.Panicf("Split points for PWL-KAN must be rank-1, got %s instead", initialValues.Shape())
	}
	c.pwl.splitPointInitialValue = initialValues
	c.NumControlPoints(initialValues.Shape().Dim(0))
	return c
}

// PWLExtrapolate configures whether the PWL-KAN should extrapolate linearly outside the split-points.
// If useLinear is set to false, it extrapolates as a constant value.
//
// Default is false (constant extrapolation), and it is set by the hyperparameter ParamPWLExtrapolate.
func (c *Config) PWLExtrapolate(useLinear bool) *Config {
	c.pwl.linearExtrapolation = useLinear
	return c
}

// Layer implements one PWL-KAN layer. x is expected to be shaped [batchSize, numInputNodes].
func (c *Config) pwlLayer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	g := x.Graph()
	dtype := x.DType()
	residual := x
	batchSize := x.Shape().Dimensions[0]
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	numInputGroups := numInputNodes
	inputGroupSize := 1
	if c.inputGroupSize > 1 {
		if numInputNodes%c.inputGroupSize != 0 {
			exceptions.Panicf("KAN configured with input group size %d, but input (shape %s) last dimension %d is not divisible by %d",
				c.inputGroupSize, c.input.Shape(), numInputNodes, c.inputGroupSize)
		}
		numInputGroups = numInputNodes / c.inputGroupSize
		inputGroupSize = c.inputGroupSize
	}
	x = Reshape(x, -1, numInputGroups, inputGroupSize)

	if klog.V(2).Enabled() {
		klog.Infof("kan pwlLayer (%s): 2 x %d x %d x %d = %d weights, splits_trainable=%v\n",
			ctx.Scope(), c.numControlPoints, numInputGroups, numOutputNodes,
			2*c.numControlPoints*numInputGroups*numOutputNodes, c.pwl.splitPointsTrainable)
	}

	// Create control points for piecewise-constant function (PCF)
	initialSeed := context.GetParamOr(ctx, initializers.ParamInitialSeed, initializers.NoSeed)
	controlPointsInitializer := func(graph *Graph, shape shapes.Shape) *Node {
		// Values initialized with a fixed slope sampled from N(0, 1).
		indices := Iota(graph, shapes.Make(dtypes.Int32, shape.Dimensions...), -1)
		isFirstInput := Equal(indices, ScalarZero(graph, dtypes.Int32))

		// Apply a random constant.
		var slope *Node
		slopeShape := shape.Clone()
		slopeShape.Dimensions[shape.Rank()-1] = 1
		initializers.UseRngState(graph, initialSeed, func(rngState *Node) (newRngState *Node) {
			newRngState, slope = RandomNormal(rngState, slopeShape)
			return newRngState
		})

		// Return the random slope as the first value, and zero elsewhere.
		return Where(isFirstInput, BroadcastToShape(slope, shape), Zeros(graph, shape))
	}
	controlPointsVar := ctx.WithInitializer(controlPointsInitializer).
		VariableWithShape("kan_pwl_control_points", shapes.Make(dtype, numOutputNodes, numInputGroups, c.numControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)

	// bias term:
	biasPointsVar := ctx.WithInitializer(initializers.Zero).
		VariableWithShape("kan_pwl_bias", shapes.Make(dtype, numOutputNodes, numInputGroups))
	bias := biasPointsVar.ValueGraph(g)

	// splitPoints: start from values distributed between -1 and 1, and let them be trained.
	var splitPoints *Node
	initialSplitPointsT := c.pwl.splitPointInitialValue
	if initialSplitPointsT == nil {
		keys := make([]float64, c.numControlPoints)
		if c.numControlPoints > 2 {
			// Initialize split points uniformly from rangeMin (-1.0) to rangeMax (1.0)
			rangeMin, rangeMax := c.pwl.rangeMin, c.pwl.rangeMax
			rangeLen := rangeMax - rangeMin
			for ii := range keys {
				keys[ii] = rangeLen*float64(ii)/float64(c.numControlPoints-1) + rangeMin
			}
		}
		initialSplitPointsT = tensors.FromValue(keys)
	}
	if c.pwl.splitPointsTrainable {
		// Trainable split points: one per input.
		// * We could also make it learn one per output ... at the cost of more parameters.
		splitPointsVar := ctx.WithInitializer(initializers.BroadcastTensorToShape(initialSplitPointsT)).
			VariableWithShape("kan_pwl_split_points", shapes.Make(dtype, numInputGroups, c.numControlPoints))
		if c.pwl.splitPointsFrozen {
			splitPointsVar.Trainable = false
		}
		splitPoints = splitPointsVar.ValueGraph(g)

		// At the end of each training step, project splitPoints back to monotonically increasing values, so they
		// don't overlap.
		train.AddPerStepUpdateGraphFn(ctx.In("kan_pwl_split_points_projection"), g, func(ctx *context.Context, g *Graph) {
			splitPoints := splitPointsVar.ValueGraph(g)
			margin := Scalar(g, splitPoints.DType(), c.pwl.splitPointsMargin)
			splitPoints = optimizers.MonotonicProjection(splitPoints, margin, -1)
			splitPointsVar.SetValueGraph(splitPoints)
		})

	} else {
		// Fixed split points.
		splitPoints = ConstCachedTensor(g, initialSplitPointsT)
		splitPoints = ConvertDType(splitPoints, dtype)
		splitPoints = Reshape(splitPoints, 1, c.numControlPoints)
	}

	// Calculate PiecewiseLinear function
	expandedX := ExpandDims(x, -1 /* numControlPoints */)
	expandedSplitPoints := ExpandDims(splitPoints, 0 /*batch*/, -2 /* inputGroupSize */)
	relativeX := Sub(expandedX, expandedSplitPoints)
	zero := ScalarZero(g, dtype)
	if c.pwl.linearExtrapolation {
		// Linear extrapolation:
		// - relativeX is not clipped at 0 for the first of the split-points.
		indicesShape := relativeX.Shape().Clone()
		indicesShape.DType = dtypes.Int32
		isFirstIndex := Equal(Iota(g, indicesShape, -1), ScalarZero(g, dtypes.Int32))
		relativeX = Where(isFirstIndex, relativeX, Max(relativeX, zero))
	} else {
		// Constant extrapolation:
		lastSplitPoint := SliceAxis(expandedSplitPoints, -1, AxisElem(-1))
		maxAllowedRelativeX := Sub(lastSplitPoint, expandedSplitPoints)
		relativeX = Max(relativeX, zero)
		relativeX = Min(relativeX, maxAllowedRelativeX)
	}
	//relativeX.SetLogged("clamped relativeX")

	// EinSum terms:
	// B -> batchSize
	// O -> numOutputNodes
	// I -> numInputGroups
	// G -> inputGroupSize
	// N -> numControlPoints
	output := Einsum("BIGN,OIN->BOIG", relativeX, controlPoints)
	expandedBias := ExpandDims(bias, 0, -1) // Expand to include batch (B) and input group axes (G).
	output = Add(output, expandedBias)

	output.AssertDims(batchSize, numOutputNodes, numInputGroups, inputGroupSize)
	output = Reshape(output, batchSize, numOutputNodes, numInputNodes) // De-group inputs.

	// Reduce the inputs to get the outputs.
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
