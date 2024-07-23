// Package kan implements a generic Kolmogorov–Arnold Networks, as described in https://arxiv.org/pdf/2404.19756
//
// Start with New(ctx, x, numOutputNodes). Configure further as desired. When finished, call Done, and it will
// return KAN(x), per configuration.
//
// It is highly customizable, but the default ties to follow the description on section "2.2 KAN architecture" of
// the paper.
//
// TODO: Implement "the good" variants from https://github.com/mintisan/awesome-kan
package kan

import (
	"fmt"
	"github.com/gomlx/bsplines"
	xbsplines "github.com/gomlx/bsplines/gomlx"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/types/shapes"
	"k8s.io/klog/v2"
	"slices"
)

const (
	// ParamNumHiddenLayers is the hyperparameter that defines the default number of hidden layers.
	// The default is 0, so no hidden layers.
	ParamNumHiddenLayers = "kan_num_hidden_layers"

	// ParamNumHiddenNodes is the hyperparameter that defines the default number of hidden nodes for KAN hidden layers.
	// Default is 10.
	ParamNumHiddenNodes = "kan_num_hidden_nodes"

	// ParamBSplineNumControlPoints is the hyperparameter that defines the default number of control points
	// for the bsplines used in the univariate KAN functions.
	// Default is 20.
	ParamBSplineNumControlPoints = "kan_bspline_num_points"

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

// Config is created with New and can be configured with its methods, or simply setting the corresponding
// hyperparameters in the context.
type Config struct {
	ctx                             *context.Context
	input                           *Node
	numOutputNodes                  int
	numHiddenLayers, numHiddenNodes int
	activation                      string
	regularizer                     regularizers.Regularizer

	bsplineNumControlPoints, bsplineDegree int
	bsplineMagnitudeTerms, bsplineResidual bool
	bsplineMagnitudeRegularizer            regularizers.Regularizer
}

// New returns the configuration for a KAN layer(s) to be applied to the input x.
// See methods for optional configurations.
// When finished configuring call Done, and it will return "KAN(x)".
//
// The input is expected to have shape `[<batch dimensions...>, <featureDimension>]`, the output will have
// shape `[<batch dimensions...>, <numOutputNodes>]`.
//
// It will apply KAN-like transformations to the last axis (the "feature axis") of x, while preserving all leading
// axes (we'll call them "batch axes").
func New(ctx *context.Context, input *Node, numOutputNodes int) *Config {
	c := &Config{
		ctx:             ctx,
		input:           input,
		numOutputNodes:  numOutputNodes,
		numHiddenLayers: context.GetParamOr(ctx, ParamNumHiddenLayers, 0),
		numHiddenNodes:  context.GetParamOr(ctx, ParamNumHiddenNodes, 10),
		activation:      context.GetParamOr(ctx, layers.ParamActivation, "silu"),
		regularizer:     regularizers.FromContext(ctx),

		bsplineNumControlPoints: context.GetParamOr(ctx, ParamBSplineNumControlPoints, 20),
		bsplineDegree:           context.GetParamOr(ctx, ParamBSplineDegree, 2),
		bsplineResidual:         true,
		bsplineMagnitudeTerms:   true,
	}

	var magRegs []regularizers.Regularizer
	magL2 := context.GetParamOr(ctx, ParamBSplineMagnitudeL2, 0.0)
	if magL2 > 0 {
		magRegs = append(magRegs, regularizers.L2(magL2))
	}
	magL1 := context.GetParamOr(ctx, ParamBSplineMagnitudeL1, 0.0)
	if magL1 > 0 {
		magRegs = append(magRegs, regularizers.L1(magL1))
	}
	c.bsplineMagnitudeRegularizer = regularizers.Combine(magRegs...)

	if input.Rank() < 2 {
		exceptions.Panicf("kan: input must be rank at least 2, got input.shape=%s", input.Shape())
	}

	return c
}

// NumHiddenLayers configure the number of hidden layers between the input and the output.
// Each layer will have numHiddenNodes nodes.
//
// The default is 0 (no hidden layers), but it will be overridden if the hyperparameter
// ParamNumHiddenLayers is set in the context (ctx).
// The value for numHiddenNodes can also be configured with the hyperparameter ParamNumHiddenNodes.
func (c *Config) NumHiddenLayers(numLayers, numHiddenNodes int) *Config {
	if numLayers < 0 || (numLayers > 0 && numHiddenNodes < 1) {
		exceptions.Panicf("kan: numHiddenLayers (%d) must be greater or equal to 0 and numHiddenNodes (%d) must be greater or equal to 1",
			numLayers, numHiddenNodes)
	}
	c.numHiddenLayers = numLayers
	c.numHiddenNodes = numHiddenNodes
	return c
}

// Activation sets the activation for the KAN, which is applied after the sum.
//
// Following the paper, it defaults to "silu" (== "swish"), but it will be overridden if the hyperparameter
// layers.ParamActivation (="activation") is set in the context.
func (c *Config) Activation(activation string) *Config {
	c.activation = activation
	return c
}

// Regularizer to be applied to the learned weights.
// Default is none.
//
// To use more than one type of Regularizer, use regularizers.Combine, and set the returned combined regularizer here.
//
// For BSpline models it applies the regularizer to the control-points.
//
// The default is no regularizer, but it can be configured by regularizers.ParamL1 and regularizers.ParamL2.
func (c *Config) Regularizer(regularizer regularizers.Regularizer) *Config {
	c.regularizer = regularizer
	return c
}

// BSpline configures the KAN to use b-splines to model \phi(x) (the function used in the paper), and set the number
// of control points to use to model the function.
//
// The numControlPoints must be greater or equal to 3, and it defaults to 20 and can also be set by using the
// hyperparameter ParamBSplineNumControlPoints ("kan_bspline_num_points").
func (c *Config) BSpline(numControlPoints int) *Config {
	return c
}

// WithBSplineMagnitudeRegularizer to be applied to the magnitude weights for BSpline.
// Default is none, but can be changed with hyperparameters ParamBSplineMagnitudeL2 and ParamBSplineMagnitudeL1.
//
// For BSpline models it applies the regularizer to the control-points.
func (c *Config) WithBSplineMagnitudeRegularizer(regularizer regularizers.Regularizer) *Config {
	c.bsplineMagnitudeRegularizer = regularizer
	return c
}

// Done takes the configuration and apply the KAN layer(s) configured.
func (c *Config) Done() *Node {
	ctx := c.ctx

	// Reshape to rank-2: [batch, features]
	numInputNodes := c.input.Shape().Dimensions[c.input.Rank()-1]
	x := c.input
	if x.Rank() != 2 {
		x = Reshape(x, -1, numInputNodes)
	}

	// Apply hidden layers.
	for ii := range c.numHiddenLayers {
		layerCtx := ctx.In(fmt.Sprintf("kan_hidden_%d", ii))
		x = c.layer(layerCtx, x, c.numHiddenNodes)
	}

	// Apply last layer.
	x = c.layer(ctx.In("kan_output_layer"), x, c.numOutputNodes)

	// Reshape back the batch axes.
	// x here is shaped [batchSize, numOutputNodes]
	if c.input.Rank() != 2 {
		outputShape := slices.Clone(c.input.Shape().Dimensions)
		outputShape[len(outputShape)-1] = c.numOutputNodes
		x = Reshape(x, outputShape...)
	}
	return x
}

// layer implements one KAN layer. x is expected to be rank-2.
func (c *Config) layer(ctx *context.Context, x *Node, numOutputNodes int) *Node {
	g := x.Graph()
	dtype := x.DType()
	residual := x
	numInputNodes := x.Shape().Dimensions[x.Rank()-1]
	batchSize := x.Shape().Dimensions[0]

	if klog.V(2).Enabled() {
		klog.Infof("kan layer (%s): (%d+2) x %d x %d = %d weights\n",
			ctx.Scope(), c.bsplineNumControlPoints, numInputNodes, numOutputNodes,
			(c.bsplineNumControlPoints+2)*numInputNodes*numOutputNodes)
	}

	// Magnitude terms w_b (residual) and w_s (spline) from the paper.
	// Notice we dropped the initializations suggested by the paper because the seemed much worse in all my experiments.
	var weightsSplines, weightsResidual *Node
	if c.bsplineMagnitudeTerms {
		weightsSplinesVar := ctx.WithInitializer(initializers.One).
			VariableWithShape("w_splines", shapes.Make(dtype, 1, numOutputNodes, numInputNodes))
		if c.bsplineMagnitudeRegularizer != nil {
			c.bsplineMagnitudeRegularizer(ctx, g, weightsSplinesVar)
		}
		weightsSplines = weightsSplinesVar.ValueGraph(g)
		if c.bsplineResidual {
			weightsResidualVar := ctx.WithInitializer(initializers.XavierFn(0, numInputNodes, numOutputNodes)).
				VariableWithShape("w_residual", shapes.Make(dtype, 1, numOutputNodes, numInputNodes))
			weightsResidual = weightsResidualVar.ValueGraph(g)
			if c.bsplineMagnitudeRegularizer != nil {
				c.bsplineMagnitudeRegularizer(ctx, g, weightsResidualVar)
			}
		}
	}

	// Apply B-spline:
	b := bsplines.NewRegular(c.bsplineDegree, c.bsplineNumControlPoints).WithExtrapolation(bsplines.ExtrapolateLinear)
	controlPointsVar := ctx.WithInitializer(initializers.RandomNormalFn(0, 0.01)).
		VariableWithShape("bspline_control_points", shapes.Make(dtype, numInputNodes, numOutputNodes, c.bsplineNumControlPoints))
	if c.regularizer != nil {
		c.regularizer(ctx, g, controlPointsVar)
	}
	controlPoints := controlPointsVar.ValueGraph(g)
	output := xbsplines.Evaluate(b, x, controlPoints)
	output.AssertDims(batchSize, numOutputNodes, numInputNodes) // Shape=[batch, outputs, inputs]

	// Per section "2.2 Kan Architecture"
	if c.bsplineMagnitudeTerms {
		output = Mul(output, weightsSplines)
	}
	if c.bsplineResidual {
		residual = layers.Activation(c.activation, residual)
		residual = ExpandDims(residual, 1)
		residual.AssertDims(batchSize, 1, numInputNodes)
		if c.bsplineMagnitudeTerms {
			residual = Mul(residual, weightsResidual)
		}
		output = Add(output, residual)
	}

	// ReduceSum the inputs to get the outputs.
	output = ReduceSum(output, -1)
	output.AssertDims(batchSize, numOutputNodes) // Shape=[batch, outputs]
	return output
}
