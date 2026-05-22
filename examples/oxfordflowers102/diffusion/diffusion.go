// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package diffusion contains an example diffusion model, trained on Oxford Flowers 102 dataset.
//
// See the accompanying jupyter notebook for some results, and how to call it.
//
// The subdirectory `train/` has the command line binary that can be executed for training.
//
// Based on the Keras tutorial in https://keras.io/examples/generative/ddim/, and recreated for GoMLX, with
// many small modifications.
//
// Flags are defined on the files that use them, so they are spread over the code.
package diffusion

import (
	"math"
	"strconv"
	"strings"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
	timages "github.com/gomlx/gomlx/core/tensors/images"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
	"github.com/gomlx/gomlx/support/exceptions"
)

var nanLogger *nanlogger.NanLogger

// SinusoidalEmbedding provides embeddings of `x` for different frequencies.
// This is applied to the variance of the noise, and facilitates the NN model to easily map different ranges
// of the signal/noise ratio.
func SinusoidalEmbedding(scope *model.Scope, x *Node) *Node {
	g := x.Graph()

	// Generate geometrically spaced frequencies: only 1/2 of *flagEmbeddingDims because we use half for sine numbers, half for cosine numbers.
	halfEmbed := model.GetParamOr(scope, "sinusoidal_embed_size", 32) / 2
	logMinFreq := math.Log(model.GetParamOr(scope, "sinusoidal_min_freq", 1.0))
	logMaxFreq := math.Log(model.GetParamOr(scope, "sinusoidal_max_freq", 1000.0))
	frequencies := IotaFull(g, shapes.Make(x.DType(), halfEmbed))
	frequencies = AddScalar(
		MulScalar(frequencies, (logMaxFreq-logMinFreq)/float64(halfEmbed-1.0)),
		logMinFreq)
	frequencies = Exp(frequencies)
	frequencies.AssertDims(halfEmbed) // Geometrically spaced frequencies.

	// Generate sine/cosine embeddings.
	angularSpeeds := MulScalar(frequencies, 2.0*math.Pi)
	if !x.Shape().IsScalar() {
		angularSpeeds = ExpandLeftToRank(angularSpeeds, x.Rank())
	}
	angles := Mul(angularSpeeds, x)
	return Concatenate([]*Node{Sin(angles), Cos(angles)}, -1)
}

var (
	flagChannelsList = xslices.Flag("channels_list", []int{32, 64, 96, 128},
		"Number of channels (features) for each image size (progressively smaller) in U-Net model",
		strconv.Atoi)
)

// NormalizeLayer behaves according to the `--norm` flag.
// It works with `x` with rank 4 and rank 3.
func NormalizeLayer(scope *model.Scope, x *Node) *Node {
	normType := model.GetParamOr(scope, layers.ParamNormalization, "none")
	switch normType {
	case "none":
		// No-op.
	case "batch":
		x = norm.BatchNorm(scope, x, -1).Center(false).Scale(false).Done()
	case "layer":
		//x = norm.LayerNorm(scope, x, -1).Done()
		x = norm.LayerNorm(scope, x, 1, 2).Done()
	}
	nanLogger.TraceFirstNaN(x)
	return x
}

// concatScopeFeatures to x, by broadcasting scopeFeature to x spatial dimensions.
func concatScopeFeatures(x, scopeFeatures *Node) *Node {
	if scopeFeatures == nil {
		return x
	}
	broadcastDims := scopeFeatures.Shape().Clone().Dimensions
	for _, axis := range timages.GetSpatialAxes(x, timages.ChannelsLast) {
		broadcastDims[axis] = x.Shape().Dimensions[axis]
	}
	scopeFeatures = BroadcastToDims(scopeFeatures, broadcastDims...)
	return Concatenate([]*Node{x, scopeFeatures}, -1)
}

// ResidualBlock on the input with `outputChannels` (axis 3) in the output.
//
// The parameter `x` must be of rank 4, shaped `[BatchSize, height, width, channels]`.
func ResidualBlock(scope *model.Scope, nanLogger *nanlogger.NanLogger, x *Node, outputChannels int) *Node {
	x.AssertRank(4)
	inputChannels := x.Shape().Dimensions[3]
	residual := x
	layerNum := 0
	nextScope := func(name string) (newScope *model.Scope) {
		newScope = scope.In("%03d-%s", layerNum, name)
		layerNum++
		return
	}

	if inputChannels != outputChannels {
		residual = layers.Dense(nextScope("residual_projection"), x, true, outputChannels)
		//residual = NormalizeLayer(nextScope("residual_normalization"), residual)
	}
	nanLogger.TraceFirstNaN(residual, "residual")

	x = NormalizeLayer(nextScope("norm"), x)
	nanLogger.TraceFirstNaN(x, "x = NormalizeLayer(nextScope(\"norm\"), x)")

	version := model.GetParamOr(scope, "diffusion_residual_version", 1)
	switch version {
	case 1: // Version 1: the original.
		x = layers.Convolution(nextScope("conv"), x).Channels(outputChannels).KernelSize(3).PadSame().Done()
		x = layers.DropBlock(scope, x).ChannelsAxis(timages.ChannelsLast).Done()
		x = activation.ApplyFromScope(scope, x)
		x = layers.Convolution(nextScope("conv"), x).Channels(outputChannels).KernelSize(3).PadSame().Done()
		x = layers.DropBlock(scope, x).ChannelsAxis(timages.ChannelsLast).Done()
		nanLogger.TraceFirstNaN(x, "x (Version 1)")

	case 2: // Version 2: slimmer.
		residual = activation.ApplyFromScope(scope, residual)
		convScope := nextScope("conv").WithInitializer(initializer.Zero)
		x = layers.Convolution(convScope, x).Channels(outputChannels).KernelSize(3).PadSame().Done()
		x = layers.DropBlock(scope, x).ChannelsAxis(timages.ChannelsLast).Done()
		nanLogger.TraceFirstNaN(x, "x (Version 2)")

	default:
		exceptions.Panicf("ResidualBlock(): invalid \"diffusion_residual_version\" %d: valid values are 1 or 2", version)
	}

	x = layers.DropPathFromScope(scope, x)
	nanLogger.TraceFirstNaN(x, "x = layers.DropPathFromScope(scope, x)")
	x = Add(x, residual)
	nanLogger.TraceFirstNaN(x, "x = Add(x, residual)")
	return x
}

// DownBlock applies `numBlocks` residual blocks followed by an average pooling of size 2, halving the spatial size.
// It pushes the values between each residual blocks to the `skips` stack, to build the skip connections later.
//
// It returns the transformed `x` and `skips` with newly stacked skip connections.
func DownBlock(scope *model.Scope, nanLoggger *nanlogger.NanLogger, x *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	for ii := range numBlocks {
		x = ResidualBlock(scope.In("%03d-residual", ii), nanLogger, x, outputChannels)
		skips = append(skips, x)
	}
	poolType := model.GetParamOr(scope, "diffusion_pool", "mean")
	switch poolType {
	case "mean":
		x = MeanPool(x).Window(2).NoPadding().Done()
	case "max":
		x = MaxPool(x).Window(2).NoPadding().Done()
	case "sum":
		x = SumPool(x).Window(2).NoPadding().Done()
	case "concat":
		x = ConcatPool(x).Window(2).NoPadding().Done()
	default:
		exceptions.Panicf(`invalid "diffusion_pool" setting %q: valid values are mean, max, sum or concat`, poolType)
	}
	nanLogger.TraceFirstNaN(x)
	return x, skips
}

func UpSampleImages(images *Node) *Node {
	shape := images.Shape()
	batchSize := shape.Dimensions[0]
	height, width := shape.Dimensions[1], shape.Dimensions[2]
	numChannels := shape.Dimensions[3]
	upSampled := Concatenate([]*Node{images, images}, 3)
	upSampled = Reshape(upSampled, batchSize, height, 2*width, numChannels)
	upSampled = Concatenate([]*Node{upSampled, upSampled}, 2)
	upSampled = Reshape(upSampled, batchSize, 2*height, 2*width, numChannels)
	return upSampled
}

// UpBlock is the counter-part to DownBlock. It performs up-scaling convolutions and connects skip-connections popped
// from `skips`.
//
// It returns `x` and `skips` after popping the consumed skip connections.
func UpBlock(scope *model.Scope, nanLogger *nanlogger.NanLogger, x *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	nanLogger.PushScope("UpBlock")
	defer nanLogger.PopScope()

	//x = Interpolate(x, timages.GetUpSampledSizes(x, timages.ChannelsLast, 2)...).Nearest().Done()
	x = UpSampleImages(x)
	nanLogger.TraceFirstNaN(x, "UpSampleImage")
	for ii := range numBlocks {
		blockScope := scope.In("%03d-residual", ii)
		nanLogger.PushScope(blockScope.Scope())
		var skip *Node
		skip, skips = xslices.Pop(skips)
		x = Concatenate([]*Node{x, skip}, -1)
		x = ResidualBlock(blockScope, nanLogger, x, outputChannels)
		nanLogger.TraceFirstNaN(x)
		nanLogger.PopScope()
	}
	return x, skips
}

func shapeToStr(shape shapes.HasShape) string {
	parts := make([]string, 1, shape.Shape().Rank())
	parts[0] = "BatchSize"
	parts = append(parts, xslices.Map(shape.Shape().Dimensions[1:], strconv.Itoa)...)
	return strings.Join(parts, ",")
}

const UNetModelScope = "u-net"

const IsV1Test = true

// UNetModelGraph builds the U-Net model.
//
// Parameters:
//   - noisyImages: image shaped `[batch_size, size, size, channels=3]`.
//   - noiseVariance: One value [0.0-1.0] per example in the batch, shaped `[batch_size, 1, 1, 1]`.
//   - flowerIds: One int32 value between [0, 102] (flower class) per example in the batch, shaped `[batch_size]`.
//
// Hyperparameters set in scope:
//
//   - "diffusion_channels_list" (static hyperparameter): number of channels (embedding size) to use in the model.
//     For each value `diffusion_num_residual_blocks` are applied and then the image is pooled and reduced by a factor of 2 --
//     later to be up-sampled again. So at most `log2(size)` values.
//   - "diffusion_num_residual_blocks" (static hyperparameter): number of blocks to use per numChannelsList element.
func UNetModelGraph(scope *model.Scope, nanLogger *nanlogger.NanLogger, noisyImages, noiseVariances, flowerIds *Node) *Node {
	dtype := noisyImages.DType()
	scope = scope.In(UNetModelScope).WithInitializer(initializer.XavierNormalFn(scope))

	// nextScope return a new scope prefixed with a counter, to give a nice ordering to the variables.
	layerNum := 0
	nextScope := func(format string, args ...any) (newScope *model.Scope) {
		newScope = scope.In("%03d-"+format, append([]any{layerNum}, args...)...)
		layerNum++
		return
	}

	batchSize := noisyImages.Shape().Dimensions[0]
	imgSize := noisyImages.Shape().Dimensions[1]
	imageChannels := noisyImages.Shape().Dimensions[3] // Always 3, but if some day we want to predict the alpha, this may be 4.
	noisyImages.AssertDims(batchSize, imgSize, imgSize, imageChannels)
	noiseVariances.AssertDims(batchSize, 1, 1, 1)
	flowerIds.AssertDims(batchSize)

	nanLogger.TraceFirstNaN(noisyImages, "UNetModelGraph:noisyImages")
	nanLogger.TraceFirstNaN(noiseVariances, "UNetModelGraph:noiseVariances")

	// Parameters from flags.
	numChannelsList := model.GetParamOr(scope, "diffusion_channels_list", []int{32, 64, 96, 128})
	numBlocks := model.GetParamOr(scope, "diffusion_num_residual_blocks", 2)

	nanLogger.TraceFirstNaN(noisyImages)
	nanLogger.TraceFirstNaN(noiseVariances)

	// Get variance sinusoidal representation, always included, and broadcast them to the spatial dimensions.
	sinEmbed := SinusoidalEmbedding(scope, noiseVariances)
	nanLogger.TraceFirstNaN(sinEmbed)
	scopeFeatures := sinEmbed
	nanLogger.TraceFirstNaN(sinEmbed, "UNetModelGraph:sinEmbed")

	// Get flower embeddings.
	flowerIds = InsertAxes(flowerIds, -1, -1, -1) // Expand axis to the match noisyImages rank.
	flowerEmbedSize := model.GetParamOr(scope, "flower_type_embed_size", 16)
	if flowerEmbedSize > 0 {
		flowerTypeEmbed := layers.Embedding(
			nextScope("FlowerEmbeddings").WithInitializer(initializer.RandomNormalFn(scope, 1.0/float64(flowerEmbedSize))),
			flowerIds, dtype, flowers.NumLabels, flowerEmbedSize, false)
		nanLogger.TraceFirstNaN(flowerTypeEmbed, "UNetModelGraph:flowerTypeEmbed")
		scopeFeatures = Concatenate([]*Node{scopeFeatures, flowerTypeEmbed}, -1)
	}

	// Adjust imageChannels to initial num channels.
	x := noisyImages
	x = layers.Dense(nextScope("StartingChannelsProjection"), x, true, numChannelsList[0])
	nanLogger.TraceFirstNaN(x, "UNetModelGraph:x")

	// Downward: keep pooling image to a smaller size.
	// Keep the `skips` features as we move "downward," so they can be "skip" connected later as we move upward.
	skips := make([]*Node, 0, numBlocks*len(numChannelsList))
	for ii, numChannels := range numChannelsList {
		blockScope := nextScope("DownBlock_%d", ii)
		nanLogger.PushScope(blockScope.Scope())
		// Apply scope features: noise rate as a sinusoidal embedding and flower types embeddings.
		x = concatScopeFeatures(x, scopeFeatures)
		x, skips = DownBlock(blockScope, nanLogger, x, skips, numBlocks, numChannels)
		nanLogger.TraceFirstNaN(x, "UNetModelGraph:x")
		nanLogger.PopScope()
	}

	// Innermost part of the model: smallest spatial shape, but usually the largest embedding size.
	//fmt.Printf("Inner shape: %s\n", x.Shape())
	numAttentionLayers := model.GetParamOr(scope, "unet_attn_layers", 0)
	if numAttentionLayers > 0 {
		blockScope := nextScope("Attention")
		nanLogger.PushScope(blockScope.Scope())
		x = TransformerBlock(blockScope, nanLogger, x)
		nanLogger.PopScope()

	} else {
		// Normal residual block for inner image:
		lastNumChannels := xslices.Last(numChannelsList)
		for ii := range numBlocks {
			blockScope := nextScope("IntermediaryBlock-%02d", ii)
			nanLogger.PushScope(blockScope.Scope())
			x = ResidualBlock(blockScope, nanLogger, x, lastNumChannels)
			nanLogger.TraceFirstNaN(x, "x")
			nanLogger.PopScope()
		}
	}

	// Upward: up-sample image back to original size, one block at a time.
	for ii := range numChannelsList {
		blockScope := nextScope("UpBlock_%d", ii)
		nanLogger.PushScope(blockScope.Scope())
		numChannels := numChannelsList[len(numChannelsList)-(ii+1)]
		x, skips = UpBlock(blockScope, nanLogger, x, skips, numBlocks, numChannels)
		nanLogger.TraceFirstNaN(x, "UNetModelGraph:x")
		nanLogger.PopScope()
	}
	if len(skips) != 0 {
		exceptions.Panicf("Ended with %d skips not accounted for!?", len(skips))
	}

	// Output initialized to 0, which is the mean of the target.
	x = layers.DenseWithBias(nextScope("Readout").WithInitializer(initializer.Zero), x, imageChannels)
	nanLogger.TraceFirstNaN(x, "UNetModelGraph:x")

	return x
}

// DiffusionSchedule calculates a ratio of noise and image that needs to be mixed,
// given the diffusion time `~ [0.0, 1.0]`.
//
// Diffusion time 0 means minimum diffusion -- the signal ratio will be set to -max_signal_ratio, default to 0.95 -- and
// diffusion time 1.0 means almost all noise -- the signal ratio will be set to -min_signal_ratio, default to 0.02.
// The returned ratio has the sum of their square total 1.
//
// Typically, the shape of `time` and the returned ratios will be `[batch_size, 1, 1, 1]`.
//
// If `clipStart` is set to false, the signal ratio is not clipped, and it can go all the way to 1.0.
//
// The ratios observe the element-wise constraint: signalRatios^2 + noiseRatios^2 = 1.
// This preserves the variance of the combined (image*signalRatio+noise*noiseRatio) to 1.
func DiffusionSchedule(scope *model.Scope, times *Node, clipStart bool) (signalRatios, noiseRatios *Node) {
	// diffusion times -> angles
	startAngle := 0.0
	if clipStart {
		startAngle = math.Acos(model.GetParamOr(scope, "diffusion_max_signal_ratio", 0.95))
	}

	endAngle := math.Acos(model.GetParamOr(scope, "diffusion_min_signal_ratio", 0.02))
	diffusionAngles := AddScalar(MulScalar(times, endAngle-startAngle), startAngle)

	// The ratios typically used is Sqrt(alpha) and Sqrt(1-alpha), because it has the nice property of preserving
	// the variance (of 1) during the process.
	signalRatios = Cos(diffusionAngles)
	noiseRatios = Sin(diffusionAngles)
	return
}

// Denoise tries to separate the noise from the image.
// It is given the signal and noise ratios.
func Denoise(scope *model.Scope, noisyImages, signalRatios, noiseRatios, flowerIds *Node) (
	predictedImages, predictedNoises *Node) {
	g := noisyImages.Graph()
	var modelScope *model.Scope

	useEMA := model.GetParamOr(scope, "use_ema", false)
	if useEMA && !scope.IsTraining(g) {
		// Use exponential moving average (EMA) for inference.
		modelScope = scope.In("ema")
	} else {
		modelScope = scope
	}

	// Noise variance: since the noise is expected to have variance 1, the adjusted
	// variance to the noiseRatio (just a multiplicative factor), the new variance
	// is:
	noiseVariances := Square(noiseRatios)

	// It's easy to model the noise than the image:
	predictedNoises = UNetModelGraph(modelScope, nil, noisyImages, noiseVariances, flowerIds)
	predictedImages = Sub(noisyImages, Mul(predictedNoises, noiseRatios))
	predictedImages = Div(predictedImages, signalRatios)

	emaCoef := model.GetParamOr(scope, "diffusion_ema", 0.0)
	if scope.IsTraining(g) && emaCoef > 0 {
		// Update moving average weights:
		prefixScope := scope.Scope()
		emaScope := scope.In("ema").WithInitializer(initializer.Zero)
		newPrefixScope := emaScope.Scope()
		// Enumerate the variables we care about, under the UNet model:
		for v := range scope.In(UNetModelScope).IterVariables() {
			if !strings.HasPrefix(v.Scope(), prefixScope) {
				exceptions.Panicf("unxpected variable %q in scope %q", v.Name(), v.Scope())
			}
			suffix := v.Scope()[len(prefixScope):]
			if !strings.HasPrefix(suffix, model.ScopeSeparator) {
				suffix = model.ScopeSeparator + suffix
			}
			newScope := newPrefixScope + suffix
			emaVar := emaScope.Store().Scope(newScope).VariableWithShape(v.Name(), v.Shape())
			emaValue := Add(
				MulScalar(emaVar.NodeValue(g), emaCoef),
				MulScalar(v.NodeValue(g), 1.0-emaCoef))
			emaVar.SetNodeValue(emaValue)
		}
	}
	return
}

// BuildTrainingModelGraph builds the model for training and evaluation.
func (c *Config) BuildTrainingModelGraph() train.ModelFn {
	return func(scope *model.Scope, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()

		// Prepare the input image and noise.
		images := inputs[0]
		batchSize := images.Shape().Dimensions[0]
		if _, ok := spec.(*flowers.BalancedDataset); ok {
			// For BalancedDataset we need to gather the images from the examples.
			examplesIdx := inputs[1]
			images = Gather(images, InsertAxes(examplesIdx, -1))
		}
		flowerIds := inputs[2]
		images = AugmentImages(scope, images) // Augment images, if not training.
		images = c.PreprocessImages(images, true)
		noises := scope.RandomNormal(g, images.Shape())
		nanLogger.TraceFirstNaN(images, "images")
		nanLogger.TraceFirstNaN(noises, "noises")

		dtype := images.DType()
		cosineschedule.New(scope, g, dtype).FromScope().Done()

		// Sample noise at different schedules.
		diffusionTimes := scope.RandomUniform(g, shapes.Make(dtype, batchSize, 1, 1, 1))
		diffusionTimes = Square(diffusionTimes) // Bias towards less noise (smaller diffusion times), since it's most impactful
		signalRatios, noiseRatios := DiffusionSchedule(scope, diffusionTimes, true)
		noisyImages := Add(
			Mul(images, signalRatios),
			Mul(noises, noiseRatios))
		noisyImages = StopGradient(noisyImages)
		predictedImages, predictedNoises := Denoise(scope, noisyImages, signalRatios, noiseRatios, flowerIds)

		// Calculate our loss inside the model: use losses.ParamLoss to define the loss, and if not set,
		// back-off to "diffusion_loss" hyperparam (for backward compatibility).
		// Defaults to "mae" (mean-absolute-error).
		lossName := model.GetParamOr(scope, loss.ParamLoss, "")
		if lossName == "" {
			lossName = model.GetParamOr(scope, "diffusion_loss", "mse")
		}
		scope.SetParam(loss.ParamLoss, lossName) // Needed for old models that used "diffusion_loss".
		lossFn := must1(loss.LossFromScope(scope))
		noisesLoss := lossFn([]*Node{noises}, []*Node{predictedNoises})
		if !noisesLoss.IsScalar() {
			noisesLoss = ReduceAllMean(noisesLoss)
		}
		imagesLoss := loss.MeanAbsoluteError([]*Node{images}, []*Node{predictedImages})
		if !imagesLoss.IsScalar() {
			imagesLoss = ReduceAllMean(imagesLoss)
		}
		noiseMAE := noisesLoss
		if lossName != "mae" {
			noiseMAE = loss.MeanAbsoluteError([]*Node{noises}, []*Node{predictedNoises})
		}

		return []*Node{c.DenormalizeImages(predictedImages), noisesLoss, imagesLoss, noiseMAE}
	}
}

// TransformerBlock takes embed shaped `[batchDim, spatialDim, embedDim]`, where the spatial dimension is
// the combined dimensions of the image.
func TransformerBlock(scope *model.Scope, nanLogger *nanlogger.NanLogger, x *Node) *Node {
	g := x.Graph()
	dtype := x.DType()
	batchDim := x.Shape().Dimensions[0]
	embedDim := x.Shape().Dimensions[3]

	numLayers := model.GetParamOr(scope, "unet_attn_layers", 0)
	numHeads := model.GetParamOr(scope, "unet_attn_heads", 4)
	keyQueryDim := model.GetParamOr(scope, "unet_attn_key_dim", 16)
	posEmbedDim := model.GetParamOr(scope, "unet_attn_pos_dim", 16)

	// Collapse spatial dimensions of the image.
	embed := Reshape(x, batchDim, -1, embedDim)
	shape := embed.Shape()
	spatialDim := shape.Dimensions[1]

	// Create positional embedding variable: it is 1 in every axis, but for the
	// sequence dimension -- there will be one embedding per position.
	// Shape: [1, maxLen, embedDim]
	posEmbedShape := shapes.Make(dtype, 1, spatialDim, posEmbedDim)
	posEmbedVar := scope.VariableWithShape("positional", posEmbedShape)
	posEmbed := posEmbedVar.NodeValue(g)
	posEmbed = BroadcastToDims(posEmbed, batchDim, spatialDim, posEmbedDim) // Broadcast batch axis so we can concatenate it later.

	// Add the requested number of attention layers.
	for ii := range numLayers {
		// Each layer in its own scope.
		layerScope := scope.In("AttLayer_%d", ii)
		residual := embed
		embed = Concatenate([]*Node{embed, posEmbed}, -1)
		embed = attention.MultiHeadAttention(layerScope, embed, embed, embed, numHeads, keyQueryDim).
			WithOutputDim(embedDim).
			WithValueHeadDim(embedDim).Done()
		nanLogger.TraceFirstNaN(embed)
		embed = layers.DropoutFromScope(layerScope, embed)
		embed = NormalizeLayer(layerScope.In("normalization_1"), embed)
		attentionOutput := embed

		// Transformers recipe: 2 dense layers after attention.
		embed = layers.Dense(layerScope.In("ffn_1"), embed, true, embedDim)
		embed = activation.ApplyFromScope(layerScope, embed)
		embed = layers.Dense(layerScope.In("ffn_2"), embed, true, embedDim)
		embed = layers.DropoutFromScope(layerScope, embed)
		embed = Add(embed, attentionOutput)
		embed = NormalizeLayer(layerScope.In("normalization_2"), embed)

		// Residual connection: not part of the usual transformer layer ...
		embed = Add(residual, embed)
		nanLogger.TraceFirstNaN(embed, "embed = Add(residual, embed)")
	}
	x = Reshape(embed, batchDim, x.Shape().Dimensions[1], x.Shape().Dimensions[2], -1)
	return x
}
