// Package kan implements a generic Kolmogorov–Arnold Networks, as described in https://arxiv.org/pdf/2404.19756
//
// Start with New(ctx, x, numOutputNodes). Configure further as desired. When finished, call Config.Done, and it will
// return KAN(x), per configuration.
//
// The original KAN paper used B-Spline functions as the univariate functions. This package implements the
// B-Spline and the following alternatives: Discrete-KAN (aka. piecewise-constant functions or "staircase functions"),
// GR-KAN (using rational functions, in the form w.P(x)/Q(x) where P(x) and Q(x) are learnable polynomial functions).
//
// It is highly customizable, each class of univariate function has several hyperparameters.
//
// TODO: Implement "the good" variants from https://github.com/mintisan/awesome-kan
package kan

// This file holds the "generic" KAN implementation. Each specific univariate function type defintes its own
// "layer" function as well as hyperparameters.

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"slices"
)

const (
	// ParamNumHiddenLayers is the hyperparameter that defines the default number of hidden layers.
	// The default is 0, so no hidden layers.
	ParamNumHiddenLayers = "kan_num_hidden_layers"

	// ParamNumHiddenNodes is the hyperparameter that defines the default number of hidden nodes for KAN hidden layers.
	// Default is 10.
	ParamNumHiddenNodes = "kan_num_hidden_nodes"

	// ParamNumControlPoints is the hyperparameter that defines the default number of control points
	// for the KAN versions that use that.
	//
	// It works for both B-Splines KAN (the original) and Discrete-KAN.
	//
	// Default is 20.
	ParamNumControlPoints = "kan_num_points"

	// ParamMean instructs KAN to use mean instead of the sum of the various individual univariate functions.
	// See https://arxiv.org/abs/2407.20667.
	//
	// Default is true.
	ParamMean = "kan_mean"

	// ParamResidual defines whether to use residual connection between the layers. Default to true.
	ParamResidual = "kan_residual"

	// ParamInputGroupSize defines the size of groups in this inputs should be split. Inputs on the same
	// group share weights. Setting this to 2 will effectively divide the number of parameters by 2,
	// but force some inputs to use the same weights.
	//
	// The last dimension of the input must be divisible by this number.
	//
	// Only implemented by Discrete-KAN for now.
	//
	// Default is 0, which means no grouping.
	ParamInputGroupSize = "kan_input_group_size"

	// ParamConstantRegularizationL1 adds a regularization of the difference among the different control points, pushing
	// the learned function to be a constant.
	//
	// It works for both B-Splines KAN (the original) and Discrete-KAN.
	//
	// Default 0.0
	ParamConstantRegularizationL1 = "kan_const_l1_reg"
)

// Config is created with New and can be configured with its methods, or simply setting the corresponding
// hyperparameters in the context.
type Config struct {
	ctx                             *context.Context
	input                           *Node
	numOutputNodes                  int
	numHiddenLayers, numHiddenNodes int
	activation                      activations.Type
	regularizer                     regularizers.Regularizer
	useResidual, useMean            bool
	inputGroupSize                  int
	numControlPoints                int

	// BSpline (standard) KAN
	bspline bsplineConfig

	// Discrete-KAN
	useDiscrete bool
	discrete    discreteConfig

	// GR-KAN (Rational functions)
	useRational bool
	rational    rationalConfig

	// PWL-KAN (Piecewise-Linear KAN)
	usePWL bool
	pwl    pwlConfig
}

// New returns the configuration for a KAN bsplineLayer(s) to be applied to the input x.
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
		ctx:              ctx,
		input:            input,
		numOutputNodes:   numOutputNodes,
		numHiddenLayers:  context.GetParamOr(ctx, ParamNumHiddenLayers, 0),
		numHiddenNodes:   context.GetParamOr(ctx, ParamNumHiddenNodes, 10),
		activation:       activations.FromName(context.GetParamOr(ctx, activations.ParamActivation, "none")),
		regularizer:      regularizers.FromContext(ctx),
		useResidual:      context.GetParamOr(ctx, ParamResidual, true),
		useMean:          context.GetParamOr(ctx, ParamMean, true),
		inputGroupSize:   context.GetParamOr(ctx, ParamInputGroupSize, int(0)),
		numControlPoints: context.GetParamOr(ctx, ParamNumControlPoints, 20),
		useDiscrete:      context.GetParamOr(ctx, ParamDiscrete, false),
		useRational:      context.GetParamOr(ctx, ParamRational, false),
		usePWL:           context.GetParamOr(ctx, ParamPiecewiseLinear, false),
	}
	c.initBSpline(ctx)
	c.initDiscrete(ctx)
	c.initRational(ctx)
	c.initPiecewiseLinear(ctx)

	constL1amount := context.GetParamOr(ctx, ParamConstantRegularizationL1, 0.0)
	if constL1amount > 0 {
		c.regularizer = regularizers.Combine(c.regularizer, regularizers.ConstantL1(constL1amount))
	}

	if input.Rank() < 2 {
		exceptions.Panicf("kan: input must be rank at least 2, got input.shape=%s", input.Shape())
	}
	return c
}

// NumHiddenLayers configure the number of hidden layers between the input and the output.
// Each bsplineLayer will have numHiddenNodes nodes.
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
// Following the paper, it defaults to "none", but it will be overridden if the hyperparameter
// layers.ParamActivation (="activation") is set in the context.
//
// Most KAN functions are already non-linear, and don't require activation.
func (c *Config) Activation(activation activations.Type) *Config {
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

// UseMean instead of sum when reducing the outputs.
//
// The original paper uses sum, but mean is more stable in many scenarios.
//
// The default is true. It can be configured with the ParamMean hyperparameter.
func (c *Config) UseMean(useMean bool) *Config {
	c.useMean = useMean
	return c
}

// UseResidual sets the flag to determine the use of residuals and returns the updated configuration object.
//
// Residual connections are used to reduce the number of parameters, and to improve the performance of the model.
//
// The residual connections are implemented by adding a residual connection to the output of the previous layer,
// and then adding the residual to the output of the current layer.
//
// The default is true. It can be configured with the ParamResidual hyperparameter.
func (c *Config) UseResidual(useResidual bool) *Config {
	c.useResidual = useResidual
	return c
}

// InputGroupSize defines the size of groups in this inputs should be split. Inputs on the same
// group share weights. Setting this to 2 will effectively divide the number of parameters by 2,
// but force some inputs to use the same weights.
//
// The last dimension of the input must be divisible by this number.
//
// Only implemented by Discrete-KAN and GR-KAN for now.
//
// The default is 0, which means no grouping. It can be configured with the ParamInputGroupSize hyperparameter.
func (c *Config) InputGroupSize(inputGroupSize int) *Config {
	c.inputGroupSize = inputGroupSize
	return c
}

// NumControlPoints is used by BSpline (standard) KAN and Discrete-KAN to define the number of points
// used to control the univariate functions.
//
// The default is 10, and it can be configured with the ParamNumControlPoints hyperparameter.
func (c *Config) NumControlPoints(numControlPoints int) *Config {
	c.numControlPoints = numControlPoints
	return c
}

// Done takes the configuration and apply the KAN bsplineLayer(s) configured.
func (c *Config) Done() *Node {
	ctx := c.ctx

	// Reshape to rank-2: [batch, features]
	numInputNodes := c.input.Shape().Dimensions[c.input.Rank()-1]
	if c.inputGroupSize > 1 {
		if numInputNodes%c.inputGroupSize != 0 {
			exceptions.Panicf("KAN configured with input group size %d, but input (shape %s) last dimension %d is not divisible by %d",
				c.inputGroupSize, c.input.Shape(), numInputNodes, c.inputGroupSize)
		}
	}

	// Normalize x to rank-2.
	x := c.input
	if x.Rank() != 2 {
		x = Reshape(x, -1, numInputNodes)
	}

	// Apply hidden layers.
	// Notice residual connections, regularization, dropout, are related layers are applied within the layers themselves.
	for ii := range c.numHiddenLayers {
		if c.useDiscrete {
			layerCtx := ctx.In(fmt.Sprintf("discrete_kan_hidden_%d", ii))
			x = c.discreteLayer(layerCtx, x, c.numHiddenNodes)
		} else if c.useRational {
			layerCtx := ctx.In(fmt.Sprintf("gr_kan_hidden_%d", ii))
			x = c.rationalLayer(layerCtx, x, c.numHiddenNodes)
		} else if c.usePWL {
			layerCtx := ctx.In(fmt.Sprintf("pwl_kan_hidden_%d", ii))
			x = c.pwlLayer(layerCtx, x, c.numHiddenNodes)
		} else {
			layerCtx := ctx.In(fmt.Sprintf("bspline_kan_hidden_%d", ii))
			x = c.bsplineLayer(layerCtx, x, c.numHiddenNodes)
		}
	}

	// Apply last layer.
	if c.useDiscrete {
		x = c.discreteLayer(ctx.In("discrete_kan_output_layer"), x, c.numOutputNodes)
	} else if c.useRational {
		x = c.rationalLayer(ctx.In("gr_kan_output_layer"), x, c.numOutputNodes)
	} else if c.usePWL {
		x = c.pwlLayer(ctx.In("pwl_kan_output_layer"), x, c.numOutputNodes)
	} else {
		x = c.bsplineLayer(ctx.In("bspline_kan_output_layer"), x, c.numOutputNodes)
	}

	// Reshape back the batch axes.
	// x here is shaped [batchSize, numOutputNodes]
	if c.input.Rank() != 2 {
		outputShape := slices.Clone(c.input.Shape().Dimensions)
		outputShape[len(outputShape)-1] = c.numOutputNodes
		x = Reshape(x, outputShape...)
	}
	return x
}
