// Package vnn implements the Vector Neural Networks operators described in https://arxiv.org/abs/2104.12229
//
// It operates on 3D vectors -- so most tensors will has the last axis with dimension 3.
//
// The operators are equivariant to rotations: that means that for every F, Rotate(F(x)) = F(Rotate(x)).
package vnn

import (
	"slices"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	// ParamNumHiddenLayers is the hyperparameter that defines the default number of hidden layers.
	// The default is 0 (int), so no hidden layers.
	ParamNumHiddenLayers = "vnn_num_hidden_layers"

	// ParamNumHiddenNodes is the hyperparameter that defines the default number of hidden nodes for KAN hidden layers.
	// The default is 10 (int).
	ParamNumHiddenNodes = "vnn_num_hidden_nodes"

	// ParamResidual is the hyperparameter that defines whether to use residual connections between consecutive hidden layers.
	// If set, and the feature dimension (the last one) is the same between the input, it also adds a residual to the
	// input. Same with the outputDimensions.
	// Default is false (bool).
	ParamResidual = "vnn_residual"

	// ParamNormalization is the name of the normalization to use in between layers.
	// Only those available in vnn package (they have to be SO(3) equivariant, see paper).
	// It is only applied if there are hidden layers.
	//
	// It defaults to the parameter "normalization" (layers.ParamNormalization) and if that is not set, to "none"
	// (same as ""), which is no normalization.
	ParamNormalization = "vnn_normalization"

	// ParamDropoutRate is the name of the dropout rate to use in between layers.
	// It is only applied if there are hidden layers.
	//
	// Defaults to the parameter "dropout_rate" (layers.ParamDropoutRate) and if that is not set, to 0.0 (no dropout)
	ParamDropoutRate = "vnn_dropout_rate"

	// ParamActivation is the name of the activation to use in between layers.
	// It can be "relu" (default), "none" or "" -- the last two mean no activation.
	// Notice "relu" refers to vnn.Relu, which is a rotation-equivariant activation.
	ParamActivation = "vnn_activation"

	// ParamConcatenateNormalizedInput is the name of the hyperparameter that defines whether to augment the input
	// with a normalized version of itself.
	ParamConcatenateNormalizedInput = "vnn_concatenate_normalized_input"
)

// ActivationFn applies a non-linearity to the operand.
type ActivationFn func(ctx *context.Context, operand *Node) *Node

// Config is created with New and can be configured with its methods or simply setting the corresponding
// hyperparameters in the context.
type Config struct {
	ctx                             *context.Context
	input                           *Node
	outputChannels                  int
	numHiddenLayers, numHiddenNodes int
	activationName                  string
	activationFn                    ActivationFn
	normalization                   string
	dropoutRatio                    float64
	useResidual                     bool
	concatenateNormalizedInput      bool

	regularizer regularizers.Regularizer
}

// New creates a configuration for a VNN (Vector Neural Network), which is SO(3) invariant,
// which means that a rotation of the input will yield the same rotation on the output.
//
// It is initialized with good defaults and with hyperparameters from the given Context.
// It can then be further configured with its various methods.
//
// Once configured, call Config.Done to add the VNN computation graph and get the output.
//
// The input is expected to have the shape `[batchSize, ..., inputFeatures, 3]`, and the
// returned output will have the shape `[batchSize, ..., outputChannels, 3]`.
//
// E.g.: A VNN for a multi-class classification model with NumClasses classes, rotation invariant.
//
//	func MyModel(ctx *context.Context, inputs []*Node) (outputs []*Node) {
//		pointCloud := inputs[0]  // [batchSize, numPoints, 3]
//		ctx = ctx.In("model")
//		base := vnn.New(ctx.In("base"), pointCloud, 128).Done()  // [batchSize, 128, 3]
//		V := vnn.New(ctx.In("V"), base, NumClasses).NumHiddenLayers(0, 0)  // [batchSize, NumClasses, 3]
//		T := vnn.New(ctx.In("T"), base, NumClasses).NumHiddenLayers(0, 0)  // [batchSize, NumClasses, 3]
//		logits := InvariantDotProduct(V, T)  // [batchSize, NumClasses]
//		return []*Node{logits}
//	}
func New(ctx *context.Context, input *Node, outputChannels int) *Config {
	if input.Rank() < 2 {
		exceptions.Panicf("vnn: input must be rank at least 2, got input.shape=%s", input.Shape())
	}
	if input.Shape().Dim(-1) != 3 {
		exceptions.Panicf("vnn: the input last dimensions must be 3 -- it works with 3D vectors only for now")
	}
	if outputChannels <= 0 {
		exceptions.Panicf("vnn: outputChannels must be > 0, got %v", outputChannels)
	}

	c := &Config{
		ctx:                        ctx,
		input:                      input,
		outputChannels:             outputChannels,
		numHiddenLayers:            context.GetParamOr(ctx, ParamNumHiddenLayers, 0),
		numHiddenNodes:             context.GetParamOr(ctx, ParamNumHiddenNodes, 10),
		activationName:             context.GetParamOr(ctx, ParamActivation, "relu"),
		normalization:              context.GetParamOr(ctx, ParamNormalization, ""),
		regularizer:                regularizers.FromContext(ctx),
		dropoutRatio:               context.GetParamOr(ctx, ParamDropoutRate, 0.0),
		useResidual:                context.GetParamOr(ctx, ParamResidual, false),
		concatenateNormalizedInput: context.GetParamOr(ctx, ParamConcatenateNormalizedInput, false),
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
		exceptions.Panicf("vnn: numHiddenLayers (%d) must be greater or equal to 0 and numHiddenNodes (%d) must be greater or equal to 1",
			numLayers, numHiddenNodes)
	}
	c.numHiddenLayers = numLayers
	c.numHiddenNodes = numHiddenNodes
	return c
}

// Activation sets the activation for the VNN, in between each layer.
// The input and output layers don't get an activation layer.
//
// The default and currently the only rotation-equivariant activation defined is "relu", if not defined as a hyperparameter.
//
// Other valid values are "" or "none" for no activation function.
//
// See also ActivationFn.
func (c *Config) Activation(activation string) *Config {
	c.activationName = activation
	return c
}

// ActivationFn is an alternative way to set the activation, by providing an activation function.
// It takes precedence over the one set by Config.Activation.
func (c *Config) ActivationFn(fn ActivationFn) *Config {
	c.activationFn = fn
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
// The default is "none", but it can be overridden by setting the hyperparameter ParamNormalization (="vnn_normalization")
// in the context.
func (c *Config) Normalization(normalization string) *Config {
	if slices.Index([]string{"", "none", "layer"}, normalization) == -1 {
		exceptions.Panicf("vnn: invalid normalization %q given: valid values are \"layer\", \"\" or \"none\"",
			normalization)
	}
	c.normalization = normalization
	return c
}

// ConcatenateNormalizedInput configures if the input should be augmented with a normalized version of itself.
// This is useful to allow the model to manipulate only the directions of the inputs (and ignore magnitude).
//
// The default is false, since it has a cost in parameters and flops.
//
// It may be configured with the hyperparameter ParamConcatenateNormalizedInput (="concatenate_normalized_input")
// in the context.
func (c *Config) ConcatenateNormalizedInput(enabled bool) *Config {
	c.concatenateNormalizedInput = enabled
	return c
}

// Regularizer to be applied to the learned weights.
// Default is none.
//
// To use more than one type of Regularizer, use regularizers.Combine, and set the returned combined regularizer here.
//
// The default is regularizers.FromContext, which is configured by regularizers.ParamL1 and regularizers.ParamL2.
func (c *Config) Regularizer(regularizer regularizers.Regularizer) *Config {
	c.regularizer = regularizer
	return c
}

// Dropout sets the dropout ratio for the VNN, in between each layer. The output layer doesn't get dropout.
// It uses the normalized form of dropout (see layers.DropoutNormalize).
//
// If set to 0.0, no dropout is used.
//
// The default is 0.0, but it can be overridden by setting the hyperparameter layers.ParamDropoutRate (="dropout_rate")
// in the context.
func (c *Config) Dropout(ratio float64) *Config {
	if ratio < 0 || ratio >= 1.0 {
		exceptions.Panicf("vnn: invalid dropout ratio %f -- set to 0.0 to disable it, and it must be < 1.0 otherwise everything is dropped out",
			ratio)
	}
	c.dropoutRatio = ratio
	return c
}

// Done takes the configuration and applies the VNN as configured.
func (c *Config) Done() *Node {
	ctx := c.ctx
	x := c.input
	g := x.Graph()
	dtype := x.DType()
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)

	// Activation function.
	activationFn := c.activationFn
	if activationFn == nil {
		switch c.activationName {
		case "", "none":
			// No activation, leave it as nil.
		case "relu":
			activationFn = ReluFromContext
		default:
			exceptions.Panicf("vnn: invalid activation %q given: valid values are \"relu\", \"\" or \"none\"", c.activationName)
		}
	}

	// Normalization function.
	var normalizationFn func(ctx *context.Context, x *Node) *Node
	switch c.normalization {
	case "", "none":
		// No activation, leave it as nil.
	case "layer":
		normalizationFn = func(ctx *context.Context, x *Node) *Node {
			if x.DType() == dtypes.BFloat16 {
				return LayerNormalization(x, 1e-4)
			} else {
				return LayerNormalization(x, 1e-8)
			}
		}
	default:
		exceptions.Panicf("vnn: invalid normalization %q given: valid values are \"layer\", \"\" or \"none\"", c.normalization)
	}

	// Normalize X shape to rank-3: [batchSize, inputFeatures, 3]
	x = Reshape(x, -1, x.Shape().Dim(-2), x.Shape().Dim(-1))

	var dropoutRatio *Node
	if c.dropoutRatio > 0.0 {
		dropoutRatio = Scalar(g, dtype, c.dropoutRatio)
	}
	var residual *Node

	// For hidden layers, we have only one axis, with numHiddenNodes.
	for ii := range c.numHiddenLayers + 1 {
		// Scope for this layer
		var layerCtx *context.Context
		if ii < c.numHiddenLayers {
			layerCtx = ctx.Inf("vnn_hidden_layer_%d", ii)
		} else {
			layerCtx = ctx.In("vnn_output_layer")
		}

		// In between-layers: some don't apply to the input (ii == 0)
		if ii > 0 && activationFn != nil {
			x = activationFn(layerCtx, x)
		}
		if dropoutRatio != nil {
			x = DropoutNormalize(layerCtx, x, dropoutRatio, true)
		}
		if ii > 0 && normalizationFn != nil {
			x = normalizationFn(layerCtx, x)
		}
		if c.useResidual {
			if residual != nil && residual.Shape().Equal(x.Shape()) {
				x = Add(x, residual)
			}
			residual = x
		}

		// Concatenate normalized input if requested.
		if c.concatenateNormalizedInput {
			l2X := L2Norm(x, -1)
			l2X = Where(Equal(l2X, zero), one, l2X)
			l2X = StopGradient(l2X)
			xUnit := Div(x, l2X)
			x = Concatenate([]*Node{x, xUnit}, -2)
		}

		// Output channels: numHiddenNodes for intermediary layers, Config.outputChannels for
		// the final (output) layer.
		inputChannels := x.Shape().Dim(1)
		outputChannels := c.numHiddenNodes
		if ii == c.numHiddenLayers {
			outputChannels = c.outputChannels
		}

		// Linear transformation
		weightsVar := layerCtx.VariableWithShape("weights", shapes.Make(dtype, inputChannels, outputChannels))
		if c.regularizer != nil {
			// Only for the weights, not for the bias.
			c.regularizer(layerCtx, g, weightsVar)
		}
		weights := weightsVar.ValueGraph(g)
		// The output 3D vectors are a linear combination of the input vectors -> they are SO(3) equivariant.
		// b->batchSize, i->inputChannels, v->3 (vector) o-> outputChannels
		x = Einsum("biv,io->bov", x, weights)
	}

	// Denormalize X shape back to the original form (except for the outputChannels):
	dims := slices.Clone(c.input.Shape().Dimensions)
	dims[len(dims)-2] = x.Shape().Dim(-2)
	x = Reshape(x, dims...)
	return x
}

// InvariantDotProduct does a dot product on the last axis (presumably of dimension 3)
// of two VNN projections (rotation invariant) from the same tensor of 3d vectors.
func InvariantDotProduct(v0, v1 *Node) *Node {
	if !v0.Shape().Equal(v1.Shape()) {
		exceptions.Panicf("shapes must be the same for InvaiantDotProduct, got v0.shape=%s and v1.shape=%s", v0.Shape(), v1.Shape())
	}
	if v0.Shape().Dim(-1) != 3 {
		exceptions.Panicf("VNNs only work with 3D vectors (the last axis must have dimension 3), got v0.shape = v1.shape =%s", v0.Shape())
	}
	batchAxes := xslices.Iota(0, v0.Rank()-1) // Same for both tensors.
	return DotGeneral(v0, []int{-1}, batchAxes, v1, []int{-1}, batchAxes)
}
