// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package inceptionv3

import (
	"fmt"
	"path"
	"strings"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/pkg/errors"
)

// ClassificationImageSize if using the Inception V3's model for classification.
// The image should be 299 x 299.
const ClassificationImageSize = 299

// MinimumImageSize for width and height required.
const MinimumImageSize = 75

// EmbeddingSize output (it not using the top).
const EmbeddingSize = 2048

// NumberOfClasses when using the top layer.
const NumberOfClasses = 1000

// BuildScope is used by BuildGraph as a new sub-scope for the InceptionV3 layers.
const BuildScope = "InceptionV3"

// BuildGraph for InceptionV3 model.
//
// For a model with pre-trained weights, call Config.PreTrained.
//
// It returns a Config object that can be further configured.
// Once the configuration is finished, call `Done` and it will return the embedding
// (or classification) of the given image.
//
// See example in the package inceptionv3 documentation.
//
// Parameters:
//   - ctx: model.Context where variables are created and loaded. Variables
//     will be re-used if they were already created before in the current scope.
//     That means one can call BuildGraph more than once, and have the same
//     model be used for more than one input -- for instance, for 2-tower models.
//     To instantiate more than one model with different weights, just use
//     the context in a different scope.
//   - Image: image tensor (`*Node`) on which to apply the model. There must be
//     3 channels, and they must be scaled from -1.0 to 1.0 -- see PreprocessImage
//     to scale image accordingly if needed. If using ClassificationTop(true),
//     the images must be of size 299x299 (defined as a constant `ClassificationImageSize`).
//     Otherwise the minimum image size is 75x75.
//
// The original model has weights in `dtypes.Float32`. (TODO: If the image has
// a different `DType`, it will try to convert the weights and work the model
// fully on the image's `DType`. This hasn't been extensively tested, so no
// guarantees of quality.)
//
// The implementation follows closely the definition in
// https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/inception_v3.py
func BuildGraph(scope *model.Scope, image *Node) *Config {
	cfg := &Config{
		scope:            scope.In(BuildScope),
		image:            image,
		trainable:        true,
		batchNormEpsilon: 0.001,
		batchNormScale:   false,
	}
	cfg.ChannelsAxis(images.ChannelsLast)
	return cfg
}

// Config for instantiating an InceptionV3 model.
// After the configuration is set, call Done, and it will build the InceptionV3
// graph with the loaded variables.
//
// See Build to construct a Config object and a usage example.
type Config struct {
	scope   *model.Scope
	image   *Node
	baseDir string

	channelsAxisConfig images.ChannelsAxisConfig
	channelsAxis       int
	spatialAxes        []int
	trainable          bool

	includeTop       bool
	pooling          Pooling
	batchNormEpsilon float64
	batchNormScale   bool

	conv2dCount, batchNormCount int

	useAliases bool
}

// PreTrained configures the graph to load the pre-trained weights.
func (cfg *Config) PreTrained(baseDir string) *Config {
	cfg.baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	return cfg
}

// Trainable configures whether the variables created will be set as trainable or not -- see `model.Variable`.
func (cfg *Config) Trainable(trainable bool) *Config {
	cfg.trainable = trainable
	return cfg
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension.
func (cfg *Config) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *Config {
	cfg.channelsAxisConfig = channelsAxisConfig
	cfg.channelsAxis = images.GetChannelsAxis(cfg.image, channelsAxisConfig)
	cfg.spatialAxes = images.GetSpatialAxes(cfg.image, channelsAxisConfig)
	return cfg
}

// ClassificationTop configures whether to use the very top classification
// layer at the top of the model.
func (cfg *Config) ClassificationTop(useTop bool) *Config {
	cfg.includeTop = useTop
	return cfg
}

// Pooling to be used at the top of the model
type Pooling int

//go:generate go tool enumer -type=Pooling model.go

const (
	NoPooling Pooling = iota
	MaxPooling
	MeanPooling
)

// BatchNormScale sets whether to a scaling variable in BatchNorm.
func (cfg *Config) BatchNormScale(value bool) *Config {
	cfg.batchNormScale = value
	return cfg
}

// SetPooling configures whether to use a MaxPool at the very top of the model.
func (cfg *Config) SetPooling(pooling Pooling) *Config {
	cfg.pooling = pooling
	return cfg
}

// WithAliases will create aliases to the output of each layer.
func (cfg *Config) WithAliases(useAliases bool) *Config {
	cfg.useAliases = useAliases
	return cfg
}

// Done builds the graph based on the configuration set.
func (cfg *Config) Done() (output *Node) {
	scope := cfg.scope
	x := cfg.image

	// Sanity checking:
	g := x.Graph()
	if cfg.baseDir == "" && !cfg.trainable {
		Panicf("inceptionv3.BuildGraph(): cannot have Trainable(false) if not using pre-trained weights")
	}
	if x.Rank() != 4 {
		Panicf("inceptionv3.BuildGraph(): input image tensor must be of rank 3: e.g.: [batch_size, ..., channels], got shape %s instead", x.Shape())
	}
	if x.DType() != dtypes.Float32 {
		Panicf("inceptionv3.BuildGraph(): only Float32 supported at this time, got dtype %s instead", x.DType())
	}
	if x.Shape().Dimensions[cfg.channelsAxis] != 3 {
		Panicf("inceptionv3.BuildGraph(): image must have 3 channels, scaled from -1.0 to 1.0, got shape %s instead", x.Shape())
	}
	if cfg.includeTop {
		if cfg.baseDir == "" {
			Panicf("inceptionv3.BuildGraph(): classification top is only available is using pre-trained weights, see PreTrained method")
		}
		spatialAxes := images.GetSpatialAxes(x, cfg.channelsAxisConfig)
		for _, spatialAxis := range spatialAxes {
			if x.Shape().Dimensions[spatialAxis] != 299 {
				Panicf("inceptionv3.BuildGraph(): image dimensions must be 299x299 if using classification top,  got shape %s instead", x.Shape())
			}
		}
	}

	// Node aliases scope.
	if cfg.useAliases {
		g.PushAliasScope("inceptionV3")
		defer g.PopAliasScope()
	}

	// Build model:
	x = cfg.conv2DWithBatchNorm(scope, x, 32, 3, 3, []int{2, 2}, false)
	x = cfg.conv2DWithBatchNorm(scope, x, 32, 3, 3, nil, false)
	x = cfg.conv2DWithBatchNorm(scope, x, 64, 3, 3, nil, true)
	x = MaxPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(2).NoPadding().Done()

	x = cfg.conv2DWithBatchNorm(scope, x, 80, 1, 1, nil, false)
	x = cfg.conv2DWithBatchNorm(scope, x, 192, 3, 3, nil, false)
	x = MaxPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(2).NoPadding().Done()

	// Mixed sizes convolutions 0: 35x35x256 or 7x7x256 (depending on image size)
	branch1x1 := cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)
	_ = branch1x1

	branch5x5 := cfg.conv2DWithBatchNorm(scope, x, 48, 1, 1, nil, true)
	branch5x5 = cfg.conv2DWithBatchNorm(scope, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl := cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool := MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
	branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 32, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 1: 35x35x288 or 7x7x288
	branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)

	branch5x5 = cfg.conv2DWithBatchNorm(scope, x, 48, 1, 1, nil, true)
	branch5x5 = cfg.conv2DWithBatchNorm(scope, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
	branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 64, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 2: 35x35x288 or 7x7x288
	branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)

	branch5x5 = cfg.conv2DWithBatchNorm(scope, x, 48, 1, 1, nil, true)
	branch5x5 = cfg.conv2DWithBatchNorm(scope, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
	branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 64, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 3:
	branch3x3 := cfg.conv2DWithBatchNorm(scope, x, 384, 3, 3, []int{2, 2}, false)

	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, x, 64, 1, 1, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 96, 3, 3, []int{2, 2}, false)

	branchPool = MaxPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(2).NoPadding().Done()
	x = Concatenate([]*Node{branch3x3, branch3x3Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 4: 768 channels
	branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)

	branch7x7 := cfg.conv2DWithBatchNorm(scope, x, 128, 1, 1, nil, true)
	branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 128, 1, 7, nil, true)
	branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 192, 7, 1, nil, true)

	branch7x7Dbl := cfg.conv2DWithBatchNorm(scope, x, 128, 1, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 128, 7, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 128, 1, 7, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 128, 7, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 1, 7, nil, true)

	branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
	branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 192, 1, 1, nil, true)

	x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 5 & 6: 768 channels
	for range 2 {
		branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)

		branch7x7 = cfg.conv2DWithBatchNorm(scope, x, 160, 1, 1, nil, true)
		branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 160, 1, 7, nil, true)
		branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 192, 7, 1, nil, true)

		branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, x, 160, 1, 1, nil, true)
		branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 160, 7, 1, nil, true)
		branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 160, 1, 7, nil, true)
		branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 160, 7, 1, nil, true)
		branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 1, 7, nil, true)

		branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
		branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 192, 1, 1, nil, true)
		x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, cfg.channelsAxis)
	}

	// Mixed convolutions 7: 768 channels
	branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)

	branch7x7 = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)
	branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 192, 1, 7, nil, true)
	branch7x7 = cfg.conv2DWithBatchNorm(scope, branch7x7, 192, 7, 1, nil, true)

	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 7, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 1, 7, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 7, 1, nil, true)
	branch7x7Dbl = cfg.conv2DWithBatchNorm(scope, branch7x7Dbl, 192, 1, 7, nil, true)

	branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
	branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 192, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 8: 768 channels
	branch3x3 = cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)
	branch3x3 = cfg.conv2DWithBatchNorm(scope, branch3x3, 320, 3, 3, []int{2, 2}, false)

	branch7x7x3 := cfg.conv2DWithBatchNorm(scope, x, 192, 1, 1, nil, true)
	branch7x7x3 = cfg.conv2DWithBatchNorm(scope, branch7x7x3, 192, 1, 7, nil, true)
	branch7x7x3 = cfg.conv2DWithBatchNorm(scope, branch7x7x3, 192, 7, 1, nil, true)
	branch7x7x3 = cfg.conv2DWithBatchNorm(scope, branch7x7x3, 192, 3, 3, []int{2, 2}, false)

	branchPool = MaxPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(2).NoPadding().Done()
	x = Concatenate([]*Node{branch3x3, branch7x7x3, branchPool}, cfg.channelsAxis)

	// Mixed convolutions 9 & 10: 2048 channels
	for range 2 {
		branch1x1 = cfg.conv2DWithBatchNorm(scope, x, 320, 1, 1, nil, true)

		branch3x3 = cfg.conv2DWithBatchNorm(scope, x, 384, 1, 1, nil, true)
		branch3x3Branch1 := cfg.conv2DWithBatchNorm(scope, branch3x3, 384, 1, 3, nil, true)
		branch3x3Branch2 := cfg.conv2DWithBatchNorm(scope, branch3x3, 384, 3, 1, nil, true)
		branch3x3 = Concatenate([]*Node{branch3x3Branch1, branch3x3Branch2}, cfg.channelsAxis)

		branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, x, 448, 1, 1, nil, true)
		branch3x3Dbl = cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 384, 3, 3, nil, true)
		branch3x3DblBranch1 := cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 384, 1, 3, nil, true)
		branch3x3DblBranch2 := cfg.conv2DWithBatchNorm(scope, branch3x3Dbl, 384, 3, 1, nil, true)
		branch3x3Dbl = Concatenate([]*Node{branch3x3DblBranch1, branch3x3DblBranch2}, cfg.channelsAxis)

		branchPool = MeanPool(x).ChannelsAxis(cfg.channelsAxisConfig).Window(3).Strides(1).PadSame().Done()
		branchPool = cfg.conv2DWithBatchNorm(scope, branchPool, 192, 1, 1, nil, true)
		x = Concatenate([]*Node{branch1x1, branch3x3, branch3x3Dbl, branchPool}, cfg.channelsAxis)
	}

	if cfg.includeTop {
		// Returns the logits at the top, for the 1000 classes.
		x = ReduceMean(x, cfg.spatialAxes...) // Global mean pooling across spatial dimensions, shape=[batch_size, 2048].
		ctxWithWeights := cfg.readPredictionsWeights(scope, g)
		x = layers.DenseWithBias(ctxWithWeights, x, 1000)
		if cfg.useAliases {
			x = x.WithAlias("logits")
		}

	} else {
		// Embeddings
		switch cfg.pooling {
		case NoPooling:
			// No-op.
		case MaxPooling:
			// Global pooling across spatial dimensions, shape=[batch_size, 2048].
			x = ReduceMax(x, cfg.spatialAxes...)
		case MeanPooling:
			// Global pooling across spatial dimensions, shape=[batch_size, 2048].
			x = ReduceMean(x, cfg.spatialAxes...)
		default:
			Panicf("inceptionv3.BuildGraph(): invalid pooling option %s", cfg.pooling)
			return
		}
	}
	output = x

	// Set all variables non-trainable, if model frozen:
	if !cfg.trainable {
		currentScope := scope.Scope()
		for v := range scope.IterVariables() {
			if strings.HasPrefix(v.Scope(), currentScope) {
				v.SetTrainable(false)
			}
		}
	}

	return
}

// conv2DWithBatchNorm adds a 2D convolution, followed by batch normalization and an activation.
func (cfg *Config) conv2DWithBatchNorm(scope *model.Scope, x *Node, kernelFilters, kernelHeight, kernelWidth int,
	strides []int, padding bool) (output *Node) {
	g := x.Graph()
	if cfg.useAliases {
		g.PushAliasScope(fmt.Sprintf("conv_%03d", cfg.conv2dCount))
		defer g.PopAliasScope()
	}

	// 2D Convolution:
	ctxWithWeights := cfg.readNextConv2D(scope, g) // Create a new context scope and read weights from `.h5` file.
	convCfg := layers.Convolution(ctxWithWeights, x).CurrentScope().ChannelsAxis(cfg.channelsAxisConfig).
		Channels(kernelFilters).UseBias(false).KernelSizePerAxis(kernelHeight, kernelWidth)
	if len(strides) > 0 {
		convCfg = convCfg.StridePerAxis(strides...)
	}
	if padding {
		convCfg = convCfg.PadSame()
	} else {
		convCfg = convCfg.NoPadding()
	}
	x = convCfg.Done()

	// Batch Normalization:
	ctxWithWeights = cfg.readNextBatchNormalization(scope, g) // Create a new context scope and read weights from `.h5` file.
	x = batchnorm.New(ctxWithWeights, x, cfg.channelsAxis).CurrentScope().
		Scale(cfg.batchNormScale).Epsilon(cfg.batchNormEpsilon).Trainable(cfg.trainable).
		UseBackendInference(false).
		FrozenAverages(cfg.baseDir != ""). // If we are loading the weights, we don't want the averages to move.
		Done()

	// Apply:
	x = activations.Relu(x)

	output = x
	if cfg.useAliases {
		output = output.WithAlias("output")
	}
	return
}

// loadTensorToVariable loads the tensor from a file.
func (cfg *Config) loadTensorToVariable(scope *model.Scope, graph *Graph, tensorFileName, variableName string) {
	if cfg.baseDir == "" {
		// Do not use pre-trained weights.
		return
	}

	if scope.GetVariable(variableName) != nil {
		// Assume it's already correctly loaded.
		return
	}
	tensorPath := path.Join(cfg.baseDir, UnpackedWeightsName, tensorFileName)
	local, err := tensors.Load(tensorPath)
	if err != nil {
		panic(errors.WithMessagef(err, "inceptionv3.ModelGraph(): failed to read weights from %q", tensorPath))
	}
	// We don't need the value, since the layer will re-load it.
	_ = scope.VariableWithValue(variableName, local)
}

// readNextConv2D enters a new scope and initializes it with the pre-trained weights for the next Conv2D layer.
func (cfg *Config) readNextConv2D(scope *model.Scope, graph *Graph) (ctxInScope *model.Scope) {
	// Set scope name to something similar to the original model layer names (cosmetic only).
	ctxInScope = scope
	if cfg.conv2dCount == 0 {
		ctxInScope = scope.In("conv2d")
	} else {
		ctxInScope = scope.In("conv2d_%d", cfg.conv2dCount)
	}
	cfg.conv2dCount += 1

	// h5 names start with 1 instead of 0 (!!)
	h5Name := fmt.Sprintf("conv2d_%d/conv2d_%d/kernel:0", cfg.conv2dCount, cfg.conv2dCount)
	cfg.loadTensorToVariable(ctxInScope, graph, h5Name, "weights")

	return
}

// readPredictionsWeights enters a new scope and initializes it with the pre-trained dense weights.
func (cfg *Config) readPredictionsWeights(scope *model.Scope, graph *Graph) (ctxInScope *model.Scope) {
	ctxInScope = scope.In("predictions")
	ctxTmp := ctxInScope.In("dense") // layers.Dense will create a sub-scope, which we need to match.
	cfg.loadTensorToVariable(ctxTmp, graph, "predictions/predictions/kernel:0", "weights")
	cfg.loadTensorToVariable(ctxTmp, graph, "predictions/predictions/bias:0", "biases")
	return
}

// readNextBatchNormalization enters a new scope and initializes it with the pre-trained weights.
func (cfg *Config) readNextBatchNormalization(scope *model.Scope, graph *Graph) (ctxInScope *model.Scope) {
	ctxInScope = scope

	// Set scope name to something similar to the original model layer names (cosmetic only).
	if cfg.batchNormCount == 0 {
		ctxInScope = scope.In("batch_normalization")
	} else {
		ctxInScope = scope.In("batch_normalization_%d", cfg.batchNormCount)
	}
	cfg.batchNormCount += 1

	// h5 names start with 1 instead of 0 (!!)
	h5Group := fmt.Sprintf("batch_normalization_%d/batch_normalization_%d/", cfg.conv2dCount, cfg.conv2dCount)
	cfg.loadTensorToVariable(ctxInScope, graph, h5Group+"moving_mean:0", "mean")
	cfg.loadTensorToVariable(ctxInScope, graph, h5Group+"moving_variance:0", "variance")
	cfg.loadTensorToVariable(ctxInScope, graph, h5Group+"beta:0", "offset")

	return
}
