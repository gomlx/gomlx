// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package fnn implements a generic FNN (Feedforward Neural Network) with various configurations.
// It should suffice for the common cases and can be extended as needed.
//
// It also provides support for various hyperparameter configuration -- so the defaults can be given by
// the context parameters.
//
// E.g: A FNN for a multi-class classification model with NumClasses classes.
//
//	func MyMode(ctx *model.Context, inputs []*Node) (outputs []*Node) {
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
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/support/exceptions"
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
// hyperparameters in the model.
type Config struct {
	ctx                             *model.Context
	input                           *Node
	outputDimensions                []int
	numHiddenLayers, numHiddenNodes int
	ensembleSize                    int
	ensembleAxis                    int
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
// set in the model. See corresponding configuration methods for details.
//
// E.g: A FNN for a multi-class classification model with NumClasses classes.
//
//	func MyModel(ctx *model.Context, inputs []*Node) (outputs []*Node) {
//		x := inputs[0]
//		logits := fnn.New(ctx.In("model"), x, NumClasses).
//			NumHiddenLayers(3, 128).
//			Activation(activations.TypeSilu).
//			Dropout(0.3).
//			Normalization("layer").
//			Residual(true).
//			Done()
//		return []*Node{logits}
//	}
func New(ctx *model.Context, input *Node, outputDimensions ...int) *Config {
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
		ensembleAxis:     -1,
		numHiddenLayers:  model.GetParamOr(ctx, ParamNumHiddenLayers, 0),
		numHiddenNodes:   model.GetParamOr(ctx, ParamNumHiddenNodes, 10),
		activation:       activations.FromName(model.GetParamOr(ctx, activations.ParamActivation, "relu")),
		normalization:    model.GetParamOr(ctx, ParamNormalization, ""),
		regularizer:      regularizers.FromContext(ctx),
		dropoutRatio:     model.GetParamOr(ctx, ParamDropoutRate, -1.0),
		useResidual:      model.GetParamOr(ctx, ParamResidual, false),
		useBias:          true,
	}

	// Fallback parameters.
	if c.normalization == "" {
		c.normalization = model.GetParamOr(ctx, layers.ParamNormalization, "none")
	}
	if c.dropoutRatio < 0 {
		c.dropoutRatio = model.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
	}
	return c
}

// WithEnsembleSize configure an ensemble size greater than 1, which adds an extra "ensemble axis"
// to the variables and intermediary layers, executing the FNN as an ensemble.
//
// See WithEnsembleAxis for an alternative way to configure ensembles, when your input is already
// split into the ensemble.
func (c *Config) WithEnsembleSize(ensembleSize int) *Config {
	c.ensembleSize = ensembleSize
	return c
}

// WithEnsembleAxis defined an axis from the input operand that should be
// considered as separate per model of the ensemble.
//
// It initializes the ensemble size from the input shape's given axis.
//
// This is used when constructing upper layers of an ensemble model, when the input has
// already been transformer be per model of the ensemble -- hence it already has an ensemble axis.
//
// See WithEnsembleSize if you want to configure an ensemble for a plain input, which hasn't been
// transformed per model of the ensemble, and hence doesn't have an ensemble axis.
func (c *Config) WithEnsembleAxis(ensembleAxis int) *Config {
	if ensembleAxis < 0 {
		ensembleAxis = MustAdjustAxis(ensembleAxis, c.input)
	}
	c.ensembleAxis = ensembleAxis
	dim := c.input.Shape().Dimensions[ensembleAxis]
	if c.ensembleSize > 0 && c.ensembleSize != dim {
		exceptions.Panicf("fnn: WithEnsembleAxis(%d), input shape is %s, corresponding dimension is %d, but ensembleSize is already %d",
			ensembleAxis, c.input.Shape(), dim, c.ensembleSize)
	}
	c.ensembleSize = dim
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
// in the model.
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
// in the model.
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
// in the model.
func (c *Config) Dropout(ratio float64) *Config {
	if ratio >= 1.0 {
		exceptions.Panicf("fnn: invalid dropout ratio %f -- set to <= 0.0 to disable it, and it must be < 1.0 otherwise everything is dropped out",
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

	isEnsemble := c.ensembleSize > 1

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
		var layerCtx *model.Context
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

			// Adjust residual shape alignment to match the structural shape
			// transition executed by DotGeneral for layers passing from ii=1 to ii>1
			if isEnsemble && c.ensembleAxis < 0 && ii == 1 {
				perm := make([]int, residual.Rank())
				batchRank := residual.Rank() - 1 - len(outputDims)
				perm[0] = batchRank
				for j := 0; j < batchRank; j++ {
					perm[j+1] = j
				}
				for j := batchRank + 1; j < residual.Rank(); j++ {
					perm[j] = j
				}
				residual = TransposeAllAxes(residual, perm...)
			}
		}

		// Linear transformation
		inputLastDimension := x.Shape().Dimensions[x.Rank()-1]
		weightsDims := make([]int, 0, 2+len(outputDims))
		if isEnsemble {
			weightsDims = append(weightsDims, c.ensembleSize)
		}
		weightsDims = append(weightsDims, inputLastDimension)
		weightsDims = append(weightsDims, outputDims...)
		weightsVar := layerCtx.VariableWithShape("weights", shapes.Make(dtype, weightsDims...))
		if c.regularizer != nil {
			// Only for the weights, not for the bias.
			c.regularizer(layerCtx, g, weightsVar)
		}
		weights := weightsVar.ValueGraph(g)
		var biasNode *Node
		if c.useBias {
			biasDims := make([]int, 0, 1+len(outputDims))
			if isEnsemble {
				biasDims = append(biasDims, c.ensembleSize)
			}
			biasDims = append(biasDims, outputDims...)
			biasVar := layerCtx.VariableWithShape("biases", shapes.Make(dtype, biasDims...))
			biasNode = biasVar.ValueGraph(g)
		}

		if !isEnsemble {
			x = nn.Dense(x, weights, biasNode)
		} else if c.ensembleAxis >= 0 {
			if ii == 0 {
				x = Dot(x, weights).General([]int{x.Rank() - 1}, []int{c.ensembleAxis}, []int{1}, []int{0})
			} else {
				x = Dot(x, weights).General([]int{x.Rank() - 1}, []int{0}, []int{1}, []int{0})
			}
			if biasNode != nil {
				batchRank := x.Rank() - 1 - len(outputDims)
				axesToInsert := make([]int, batchRank)
				for j := 0; j < batchRank; j++ {
					axesToInsert[j] = 1 + j
				}
				x = Add(x, ExpandAxes(biasNode, axesToInsert...))
			}
		} else {
			switch ii {
			case 0:
				x = Dot(x, weights).General([]int{x.Rank() - 1}, nil, []int{1}, nil)
				if biasNode != nil {
					x = Add(x, ExpandLeftToRank(biasNode, x.Rank()))
				}
			case 1:
				x = Dot(x, weights).General([]int{x.Rank() - 1}, []int{x.Rank() - 2}, []int{1}, []int{0})
				if biasNode != nil {
					batchRank := x.Rank() - 1 - len(outputDims)
					axesToInsert := make([]int, batchRank)
					for j := 0; j < batchRank; j++ {
						axesToInsert[j] = 1 + j
					}
					x = Add(x, ExpandAxes(biasNode, axesToInsert...))
				}
			default:
				x = Dot(x, weights).General([]int{x.Rank() - 1}, []int{0}, []int{1}, []int{0})
				if biasNode != nil {
					batchRank := x.Rank() - 1 - len(outputDims)
					axesToInsert := make([]int, batchRank)
					for j := 0; j < batchRank; j++ {
						axesToInsert[j] = 1 + j
					}
					x = Add(x, ExpandAxes(biasNode, axesToInsert...))
				}
			}
		}
	}

	if isEnsemble {
		needsTranspose := false
		pos := x.Rank() - 1 - len(c.outputDimensions)
		if c.ensembleAxis >= 0 {
			pos = c.ensembleAxis
			needsTranspose = true
		} else if c.numHiddenLayers > 0 {
			needsTranspose = true
		}

		if needsTranspose {
			perm := make([]int, x.Rank())
			for j := 0; j < pos; j++ {
				perm[j] = j + 1
			}
			perm[pos] = 0
			for j := pos + 1; j < x.Rank(); j++ {
				perm[j] = j
			}
			x = TransposeAllAxes(x, perm...)
		}
	}

	return x
}
