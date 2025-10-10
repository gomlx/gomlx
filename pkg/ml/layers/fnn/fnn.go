// Package fnn implements a generic FNN (Feedforward Neural Network) with various configurations.
// It should suffice for the common cases and can be extended as needed.
//
// It also provides support for various hyperparameter configuration -- so the defaults can be given by
// the context parameters.
//
// E.g: A FNN for a multi-class classification model with NumClasses classes.
//
//	func MyMode(ctx *context.Context, inputs []*Node) (outputs []*Node) {
//		x := inputs[0]
//		logits := fnn.New(ctx.In("model"), x, NumClasses).
//			NumHiddenLayers(3).
//			Apply("swish").
//			Dropout(0.3).
//			UseResidual(true).
//			Done()
//		return []*Node{logits}
//	}
package fnn

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

const (
	// ParamNumHiddenLayers is the hyperparameter that defines the default number of hidden layers.
	// The default is 0 (int), so no hidden layers.
	ParamNumHiddenLayers = "fnn_num_hidden_layers"

	// ParamNumHiddenNodes is the hyperparameter that defines the default number of hidden nodes for KAN hidden layers.
	// The default is 10 (int).
	ParamNumHiddenNodes = "fnn_num_hidden_nodes"

	// ParamResidual is the hyperparameter that defines whether to use residual connections between consecutive hidden layers.
	// If set, and the feature dimension (the last one) is the same between the input, it also adds a residual to the
	// input. Same with the outputDimensions.
	// Default is false (bool).
	ParamResidual = "fnn_residual"

	// ParamNormalization is the name of the normalization to use in between layers.
	// It is only applied if there are hidden layers.
	// See layers.KnownNormalizer: "layer" and "batch" are the most common normalization strategies.
	//
	// Defaults to the parameter "normalization" (layers.ParamNormalization) and if that is not set, to "none"
	// (same as ""), which is no normalization.
	ParamNormalization = "fnn_normalization"

	// ParamDropoutRate is the name of the dropout rate to use in between layers.
	// It is only applied if there are hidden layers.
	//
	// Defaults to the parameter "dropout_rate" (layers.ParamDropoutRate) and if that is not set, to 0.0 (no dropout)
	ParamDropoutRate = "fnn_dropout_rate"
)

// Config is created with New and can be configured with its methods, or simply setting the corresponding
// hyperparameters in the context.
type Config struct {
	ctx                             *context.Context
	input                           *Node
	outputDimensions                []int
	numHiddenLayers, numHiddenNodes int
	activation                      activations.Type
	normalization                   string
	dropoutRatio                    float64
	useBias, useResidual            bool

	regularizer regularizers.Regularizer
}

// New creates a configuration for a FNN (Feedforward Neural Network).
// This can be further configured through various methods and when finished,
// call Done to actually add the FNN computation graph and get the output.
//
// The input is expected to have shape `[<batch dimensions...>, featureDimension]`, the output will have
// shape `[<batch dimensions...>, <outputDimensions...>]`.
//
// Configuration options have defaults, but can also be configured through hyperparameters
// set in the context. See corresponding configuration methods for details.
//
// E.g: A FNN for a multi-class classification model with NumClasses classes.
//
//	func MyModel(ctx *context.Context, inputs []*Node) (outputs []*Node) {
//		x := inputs[0]
//		logits := fnn.New(ctx.In("model"), x, NumClasses).
//			NumHiddenLayers(3).
//			Apply("swish").
//			Dropout(0.3).
//			Normalization("layer").
//			UseResidual(true).
//			Done()
//		return []*Node{logits}
//	}
func New(ctx *context.Context, input *Node, outputDimensions ...int) *Config {
	if input.Rank() < 2 {
		exceptions.Panicf("fnn: input must be rank at least 2, got input.shape=%s", input.Shape())
	}
	if len(outputDimensions) == 0 {
		exceptions.Panicf("fnn: at least one outputDimension must be given for layers.Dense, got 0 -- use outputDims=[1] for a scalar output")
	}
	for _, dim := range outputDimensions {
		if dim <= 0 {
			exceptions.Panicf("fnn: dimensions for outputDimensions must be > 0, got %v", outputDimensions)
		}
	}

	c := &Config{
		ctx:              ctx,
		input:            input,
		outputDimensions: outputDimensions,
		numHiddenLayers:  context.GetParamOr(ctx, ParamNumHiddenLayers, 0),
		numHiddenNodes:   context.GetParamOr(ctx, ParamNumHiddenNodes, 10),
		activation:       activations.FromName(context.GetParamOr(ctx, activations.ParamActivation, "relu")),
		normalization:    context.GetParamOr(ctx, ParamNormalization, ""),
		regularizer:      regularizers.FromContext(ctx),
		dropoutRatio:     context.GetParamOr(ctx, ParamDropoutRate, -1.0),
		useResidual:      context.GetParamOr(ctx, ParamResidual, false),
		useBias:          true,
	}

	// Fallback parameters.
	if c.normalization == "" {
		c.normalization = context.GetParamOr(ctx, layers.ParamNormalization, "none")
	}
	if c.dropoutRatio < 0 {
		c.dropoutRatio = context.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
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
		exceptions.Panicf("fnn: numHiddenLayers (%d) must be greater or equal to 0 and numHiddenNodes (%d) must be greater or equal to 1",
			numLayers, numHiddenNodes)
	}
	c.numHiddenLayers = numLayers
	c.numHiddenNodes = numHiddenNodes
	return c
}

// UseBias configures whether to add a bias term to each node.
// Almost always you want this to be true, and that is the default.
func (c *Config) UseBias(useBias bool) *Config {
	c.useBias = useBias
	return c
}

// Activation sets the activation for the FNN, in between each layer.
// The input and output layers don't get an activation layer.
//
// The default is "relu", but it can be overridden by setting the hyperparameter layers.ParamActivation (="activation")
// in the context.
func (c *Config) Activation(activation activations.Type) *Config {
	c.activation = activation
	return c
}

// Residual configures if residual connections in between layers with the same number of nodes should be used.
// They are very useful for deep models.
//
// The default is false, and it may be configured with the hyperparameter ParamResidual.
func (c *Config) Residual(useResidual bool) *Config {
	c.useResidual = useResidual
	return c
}

// Normalization sets the normalization type to use in between layers.
// The input and output layers don't get a normalization layer.
//
// The default is "none", but it can be overridden by setting the hyperparameter ParamNormalization (="fnn_normalization")
// in the context.
func (c *Config) Normalization(normalization string) *Config {
	_, found := layers.KnownNormalizers[normalization]
	if normalization != "" && !found {
		exceptions.Panicf("fnn: unknown normalization %q given: valid values are %v or \"\"",
			normalization, xslices.SortedKeys(layers.KnownNormalizers))
	}
	c.normalization = normalization
	return c
}

// Regularizer to be applied to the learned weights (but not the biases).
// Default is none.
//
// To use more than one type of Regularizer, use regularizers.Combine, and set the returned combined regularizer here.
//
// The default is regularizers.FromContext, which is configured by regularizers.ParamL1 and regularizers.ParamL2.
func (c *Config) Regularizer(regularizer regularizers.Regularizer) *Config {
	c.regularizer = regularizer
	return c
}

// Dropout sets the dropout ratio for the FNN, in between each layer. The output layer doesn't get dropout.
// It uses the normalized form of dropout (see layers.DropoutNormalize).
//
// If set to 0.0, no dropout is used.
//
// The default is 0.0, but it can be overridden by setting the hyperparameter layers.ParamDropoutRate (="dropout_rate")
// in the context.
func (c *Config) Dropout(ratio float64) *Config {
	if ratio < 0 || ratio >= 1.0 {
		exceptions.Panicf("fnn: invalid dropout ratio %f -- set to 0.0 to disable it, and it must be < 1.0 otherwise everything is dropped out",
			ratio)
	}
	c.dropoutRatio = ratio
	return c
}

// Done takes the configuration and apply the FNN as configured.
func (c *Config) Done() *Node {
	ctx := c.ctx
	x := c.input
	g := x.Graph()
	dtype := x.DType()

	var dropoutRatio *Node
	if c.dropoutRatio > 0.0 {
		dropoutRatio = Scalar(g, dtype, c.dropoutRatio)
	}
	var residual *Node

	// For hidden layers, we have only one axis, with numHiddenNodes.
	outputDims := []int{c.numHiddenNodes}
	for ii := range c.numHiddenLayers + 1 {
		if ii == c.numHiddenLayers {
			// For the output layer, we have the specified outputDimensions.
			outputDims = c.outputDimensions
		}
		// Scope for this layer
		var layerCtx *context.Context
		if ii < c.numHiddenLayers {
			layerCtx = ctx.Inf("fnn_hidden_layer_%d", ii)
		} else {
			layerCtx = ctx.In("fnn_output_layer")
		}

		// In between-layers: some don't apply to the input (ii == 0)
		if ii > 0 {
			x = activations.Apply(c.activation, x)
		}
		if dropoutRatio != nil {
			x = layers.DropoutNormalize(layerCtx, x, dropoutRatio, true)
		}
		if ii > 0 {
			if c.normalization != "" && c.normalization != "none" {
				x = layers.MustNormalizeByName(layerCtx, c.normalization, x)
			}
		}
		if c.useResidual {
			if residual != nil && residual.Shape().Equal(x.Shape()) {
				x = Add(x, residual)
			}
			residual = x
		}

		// Linear transformation
		inputLastDimension := x.Shape().Dimensions[x.Rank()-1]
		weightsDims := make([]int, 1+len(outputDims))
		weightsDims[0] = inputLastDimension
		copy(weightsDims[1:], outputDims)
		weightsVar := layerCtx.VariableWithShape("weights", shapes.Make(dtype, weightsDims...))
		if c.regularizer != nil {
			// Only for the weights, not for the bias.
			c.regularizer(layerCtx, g, weightsVar)
		}
		weights := weightsVar.ValueGraph(g)
		if x.Rank() <= 2 && len(outputDims) == 1 {
			// Vanilla version: input = [batch_size, feature_size], output = [batch_size, output_dim].
			x = Dot(x, weights)
		} else {
			// DotGeneral:
			xContractingAxes := []int{-1}
			var xBatchAxes, weightsBatchAxes []int // Since weights has no batch axes, we take no matching batch axes.
			weightsContractingAxes := []int{0}
			x = DotGeneral(x, xContractingAxes, xBatchAxes, weights, weightsContractingAxes, weightsBatchAxes)
		}

		// Add bias
		if c.useBias {
			biasVar := layerCtx.VariableWithShape("biases", shapes.Make(dtype, outputDims...))
			bias := biasVar.ValueGraph(g)
			expandedBiasShape := x.Shape().Clone()
			for ii := range expandedBiasShape.Dimensions[:x.Rank()-len(outputDims)] {
				expandedBiasShape.Dimensions[ii] = 1
			}
			expandedBias := ReshapeWithShape(bias, expandedBiasShape)
			x = Add(x, expandedBias)
		}
	}
	return x
}
