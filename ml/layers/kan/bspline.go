package kan

import (
	"github.com/gomlx/bsplines"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	xbsplines "github.com/gomlx/gomlx/ml/layers/bsplines"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/types/shapes"
	"k8s.io/klog/v2"
)

// This file implements the B-Spline based KAN model, the original one described in https://arxiv.org/pdf/2404.19756.

var (
	// ParamBSplineDegree is the hyperparameter that defines the default value for the bspline degree used in
	// the univariate KAN functions.
	// Default is 2.
	ParamBSplineDegree = "kan_bspline_degree"

	// ParamBSplineMagnitudeL2 is the hyperparameter that defines the default L2 regularization amount for the bspline
	// learned magnitude parameters.
	// Default is 0.
	ParamBSplineMagnitudeL2 = "kan_bspline_magnitude_l2"

	// ParamBSplineMagnitudeL1 is the hyperparameter that defines the default L1 regularization amount for the bspline
	// learned magnitude parameters.
	// Default is 0.
	ParamBSplineMagnitudeL1 = "kan_bspline_magnitude_l1"
)

// bsplineConfig with configuration exclusive to BSplines.
type bsplineConfig struct {
	Degree               int
	MagnitudeTerms       bool
	MagnitudeRegularizer regularizers.Regularizer
}

// initBSpline initializes the default values for Discrete-KANs based on context.
func (c *Config) initBSpline(ctx *context.Context) {
	c.bspline.Degree = context.GetParamOr(ctx, ParamBSplineDegree, 2)
	c.bspline.MagnitudeTerms = true

	var magRegs []regularizers.Regularizer
	magL2 := context.GetParamOr(ctx, ParamBSplineMagnitudeL2, 0.0)
	if magL2 > 0 {
		magRegs = append(magRegs, regularizers.L2(magL2))
	}
	magL1 := context.GetParamOr(ctx, ParamBSplineMagnitudeL1, 0.0)
	if magL1 > 0 {
		magRegs = append(magRegs, regularizers.L1(magL1))
	}
	c.bspline.MagnitudeRegularizer = regularizers.Combine(magRegs...)
}

// BSpline configures the KAN to use b-splines to model \phi(x), the univariate function described in the KAN the paper.
// It also sets the number of control points to use to model the function.
//
// This is default KAN function.
func (c *Config) BSpline() *Config {
	c.useDiscrete = false
	c.useRational = false
	return c
}

// WithBSplineMagnitudeRegularizer to be applied to the magnitude weights for BSpline.
// Default is none, but can be changed with hyperparameters ParamBSplineMagnitudeL2 and ParamBSplineMagnitudeL1.
//
// For BSpline models it applies the regularizer to the control-points.
func (c *Config) WithBSplineMagnitudeRegularizer(regularizer regularizers.Regularizer) *Config {
	c.bspline.MagnitudeRegularizer = regularizer
	return c
}

// bsplineLayer implements one KAN bsplineLayer. x is expected to be rank-2.
func (c *Config) bsplineLayer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	g := x.Graph()
	dtype := x.DType()
	residual := x
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	batchSize := x.Shape().Dimensions[0]
	if c.inputGroupSize > 1 {
		exceptions.Panicf("B-Spline KAN does not support input groups (kan_input_group_size > 1) yet -- kan_input_group_size set to %d.", c.inputGroupSize)
	}

	if klog.V(2).Enabled() {
		klog.Infof("kan bsplineLayer (%s): (%d+2) x %d x %d = %d weights\n",
			ctx.Scope(), c.numControlPoints, numInputNodes, numOutputNodes,
			(c.numControlPoints+2)*numInputNodes*numOutputNodes)
	}

	// Magnitude terms w_b (residual) and w_s (spline) from the paper.
	// Notice we dropped the initializations suggested by the paper because the seemed much worse in all my experiments.
	var weightsSplines, weightsResidual *Node
	if c.bspline.MagnitudeTerms {
		weightsSplinesVar := ctx.WithInitializer(initializers.One).
			VariableWithShape("w_splines", shapes.Make(dtype, 1, numOutputNodes, numInputNodes))
		if c.bspline.MagnitudeRegularizer != nil {
			c.bspline.MagnitudeRegularizer(ctx, g, weightsSplinesVar)
		}
		weightsSplines = weightsSplinesVar.ValueGraph(g)
		if c.useResidual {
			weightsResidualVar := ctx.WithInitializer(initializers.XavierUniformFn(ctx)).
				VariableWithShape("w_residual", shapes.Make(dtype, 1, numOutputNodes, numInputNodes))
			weightsResidual = weightsResidualVar.ValueGraph(g)
			if c.bspline.MagnitudeRegularizer != nil {
				c.bspline.MagnitudeRegularizer(ctx, g, weightsResidualVar)
			}
		}
	}

	// Apply B-spline:
	b := bsplines.NewRegular(c.bspline.Degree, c.numControlPoints).WithExtrapolation(bsplines.ExtrapolateLinear)
	controlPointsVar := ctx.WithInitializer(initializers.RandomNormalFn(ctx, 0.01)).
		VariableWithShape("bspline_control_points", shapes.Make(dtype, numInputNodes, numOutputNodes, c.numControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)
	output := xbsplines.Evaluate(b, x, controlPoints)
	output.AssertDims(batchSize, numOutputNodes, numInputNodes) // Shape=[batch, outputs, inputs]

	// Per section "2.2 Kan Architecture"
	if c.bspline.MagnitudeTerms {
		output = Mul(output, weightsSplines)
	}
	if c.useResidual {
		residual = activations.Apply(c.activation, residual)
		residual = InsertAxes(residual, 1)
		residual.AssertDims(batchSize, 1, numInputNodes)
		if c.bspline.MagnitudeTerms {
			residual = Mul(residual, weightsResidual)
		}
		output = Add(output, residual)
	}

	// Reduce the inputs to get the outputs:
	if c.useMean {
		output = ReduceMean(output, -1)
	} else {
		// ReduceSum requires Xavier initialization (initializer.XavierUniformFn)
		// whose magnitude is Sqrt(6/(fanIn+fanOut)) not to grow exponentially with the number of layers.
		output = ReduceSum(output, -1)
	}
	output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]
	return output
}
