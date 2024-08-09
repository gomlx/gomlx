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
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/types/shapes"
	timage "github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"math"
	"strconv"
	"strings"
)

var (
	DType = dtypes.Float32 // TODO: encode this in context["DType"].
)

var nanLogger *nanlogger.NanLogger

// SinusoidalEmbedding provides embeddings of `x` for different frequencies.
// This is applied to the variance of the noise, and facilitates the NN model to easily map different ranges
// of the signal/noise ratio.
func SinusoidalEmbedding(ctx *context.Context, x *Node) *Node {
	g := x.Graph()

	// Generate geometrically spaced frequencies: only 1/2 of *flagEmbeddingDims because we use half for sine numbers, half for cosine numbers.
	halfEmbed := context.GetParamOr(ctx, "sinusoidal_embed_size", 32) / 2
	logMinFreq := math.Log(context.GetParamOr(ctx, "sinusoidal_min_freq", 1.0))
	logMaxFreq := math.Log(context.GetParamOr(ctx, "sinusoidal_max_freq", 1.0))
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
func NormalizeLayer(ctx *context.Context, x *Node) *Node {
	norm := context.GetParamOr(ctx, layers.ParamNormalization, "none")
	switch norm {
	case "none":
		// No-op.
	case "batch":
		x = batchnorm.New(ctx, x, -1).Center(false).Scale(false).Done()
	case "layer":
		x = layers.LayerNormalization(ctx, x, -1).Done()
		// Optionally, normalize over the spatial dimensions instead:
		// x = layers.LayerNormalization(ctx, x, 1, 2).Done()
	}
	nanLogger.Trace(x)
	return x
}

// concatContextFeatures to x, by broadcasting contextFeature to x spatial dimensions.
func concatContextFeatures(x, contextFeatures *Node) *Node {
	if contextFeatures == nil {
		return x
	}
	broadcastDims := contextFeatures.Shape().Clone().Dimensions
	for _, axis := range timage.GetSpatialAxes(x, timage.ChannelsLast) {
		broadcastDims[axis] = x.Shape().Dimensions[axis]
	}
	contextFeatures = BroadcastToDims(contextFeatures, broadcastDims...)
	return Concatenate([]*Node{x, contextFeatures}, -1)
}

// ResidualBlock on the input with `outputChannels` (axis 3) in the output.
//
// The parameter `x` must be of rank 4, shaped `[BatchSize, height, width, channels]`.
func ResidualBlock(ctx *context.Context, x, contextFeatures *Node, outputChannels int) *Node {
	x.AssertRank(4)
	inputChannels := x.Shape().Dimensions[3]
	residual := x
	layerNum := 0
	if inputChannels != outputChannels {
		residual = layers.Dense(ctx.Inf("%03d_projection", layerNum), x, true, outputChannels)
		layerNum++
	}
	x = NormalizeLayer(ctx.Inf("%03d-norm", layerNum), x)
	layerNum++
	x = concatContextFeatures(x, contextFeatures)
	convCtx := ctx.Inf("%03d_conv", layerNum).WithInitializer(initializers.XavierNormalFn(0))
	x = layers.Convolution(convCtx, x).Filters(outputChannels).KernelSize(3).PadSame().Done()
	layerNum++
	x = activations.ApplyFromContext(ctx, x)
	x = Add(x, residual)
	nanLogger.Trace(x)
	return x
}

// DownBlock applies `numBlocks` residual blocks followed by an average pooling of size 2, halving the spatial size.
// It pushes the values between each residual blocks to the `skips` stack, to build the skip connections later.
//
// It returns the transformed `x` and `skips` with newly stacked skip connections.
func DownBlock(ctx *context.Context, x, contextFeatures *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	for ii := 0; ii < numBlocks; ii++ {
		x = ResidualBlock(ctx.Inf("%03d-residual", ii), x, contextFeatures, outputChannels)
		skips = append(skips, x)
	}
	x = MeanPool(x).Window(2).NoPadding().Done()
	nanLogger.Trace(x)
	return x, skips
}

// UpBlock is the counter-part to DownBlock. It performs up-scaling convolutions and connects skip-connections popped
// from `skips`.
//
// It returns `x` and `skips` after popping the consumed skip connections.
func UpBlock(ctx *context.Context, x, contextFeatures *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	x = Interpolate(x, timage.GetUpSampledSizes(x, timage.ChannelsLast, 2)...).Nearest().Done()
	for ii := 0; ii < numBlocks; ii++ {
		var skip *Node
		skip, skips = xslices.Pop(skips)
		x = Concatenate([]*Node{x, skip}, -1)
		x = ResidualBlock(ctx.Inf("%03d-residual", ii), x, contextFeatures, outputChannels)
	}
	nanLogger.Trace(x)
	return x, skips
}

func shapeToStr(shape shapes.HasShape) string {
	parts := make([]string, 1, shape.Shape().Rank())
	parts[0] = "BatchSize"
	parts = append(parts, xslices.Map(shape.Shape().Dimensions[1:], strconv.Itoa)...)
	return strings.Join(parts, ",")
}

// UNetModelGraph builds the U-Net model.
//
// Parameters:
//   - noisyImages: image shaped `[batch_size, size, size, channels=3]`.
//   - noiseVariance: One value [0.0-1.0] per example in the batch, shaped `[batch_size, 1, 1, 1]`.
//   - flowerIds: One int32 value between [0, 102] (flower class) per example in the batch, shaped `[batch_size]`.
//
// Hyperparameters set in ctx:
//
//   - "diffusion_channels_list" (static hyperparameter): number of channels (embedding size) to use in the model.
//     For each value `diffusion_num_residual_blocks` are applied and then the image is pooled and reduced by a factor of 2 --
//     later to be up-sampled again. So at most `log2(size)` values.
//   - "diffusion_num_residual_blocks" (static hyperparameter): number of blocks to use per numChannelsList element.
func UNetModelGraph(ctx *context.Context, noisyImages, noiseVariances, flowerIds *Node) *Node {
	ctx = ctx.In("u-net").WithInitializer(initializers.GlorotUniformFn(0))
	layerNum := 0
	batchSize := noisyImages.Shape().Dimensions[0]
	imgSize := noisyImages.Shape().Dimensions[1]
	imageChannels := noisyImages.Shape().Dimensions[3] // Always 3, but if some day we want to predict the alpha, this may be 4.
	noisyImages.AssertDims(batchSize, imgSize, imgSize, imageChannels)
	noiseVariances.AssertDims(batchSize, 1, 1, 1)
	flowerIds.AssertDims(batchSize)

	// Parameters from flags.
	numChannelsList := context.GetParamOr(ctx, "diffusion_channels_list", []int{32, 64, 96, 128})
	numBlocks := context.GetParamOr(ctx, "diffusion_num_residual_blocks", 2)

	nanLogger.Trace(noisyImages)
	nanLogger.Trace(noiseVariances)

	// Get variance sinusoidal representation, always included, and broadcast them to the spatial dimensions.
	sinEmbed := SinusoidalEmbedding(ctx, noiseVariances)
	nanLogger.Trace(sinEmbed)
	contextFeatures := sinEmbed

	// Get flower embeddings.
	flowerIds = ExpandDims(flowerIds, -1, -1, -1) // Expand axis to the match noisyImages rank.
	flowerEmbedSize := context.GetParamOr(ctx, "flower_type_embed_size", 16)
	if flowerEmbedSize > 0 {
		scopeName := fmt.Sprintf("%03d-FlowerEmbeddings", layerNum)
		layerNum++
		flowerTypeEmbed := layers.Embedding(
			ctx.In(scopeName).WithInitializer(initializers.RandomNormalFn(0, 1.0/float64(flowerEmbedSize))),
			flowerIds, DType, flowers.NumLabels, flowerEmbedSize)
		contextFeatures = Concatenate([]*Node{contextFeatures, flowerTypeEmbed}, -1)
	}

	// Adjust imageChannels to initial num channels.
	x := noisyImages
	{
		scopeName := fmt.Sprintf("%03d-StartingChannels_%s", layerNum, shapeToStr(x))
		layerNum++
		x = concatContextFeatures(x, contextFeatures)
		x = layers.Dense(ctx.In(scopeName), x, true, numChannelsList[0])
	}
	if !context.GetParamOr(ctx, "diffusion_context_features", false) {
		// If contextFeatures disabled across model, set it to nil.
		contextFeatures = nil
	}

	// Downward: keep pooling image to a smaller size.
	// Keep the `skips` features as we move "downward," so they can be "skip" connected later as we move upward.
	skips := make([]*Node, 0, numBlocks*len(numChannelsList))
	for _, numChannels := range numChannelsList {
		scopeName := fmt.Sprintf("%03d-DownBlock_%s", layerNum, shapeToStr(x))
		layerNum++
		nanLogger.PushScope(scopeName)
		blockCtx := ctx.In(scopeName)
		// Use flower types as an extra embedding.
		x, skips = DownBlock(blockCtx, x, contextFeatures, skips, numBlocks, numChannels)
		nanLogger.PopScope()
	}

	// Intermediary fixed size blocks.
	if *flagNumAttLayers > 0 {
		// Optional transformer layer.
		scopeName := fmt.Sprintf("%03d-TransformerBlock", layerNum)
		layerNum++
		nanLogger.PushScope(scopeName)
		x = TransformerBlock(ctx.In(scopeName), x)
		nanLogger.PopScope()
	}
	lastNumChannels := xslices.Last(numChannelsList)
	for ii := 0; ii < numBlocks; ii++ {
		scopeName := fmt.Sprintf("%03d-IntermediaryBlock_%s", layerNum, shapeToStr(x))
		layerNum++
		nanLogger.PushScope(scopeName)
		x = ResidualBlock(ctx.In(scopeName), x, contextFeatures, lastNumChannels)
		nanLogger.PopScope()
	}

	// Upward: up-sample image back to original size, one block at a time.
	for ii := range numChannelsList {
		scopeName := fmt.Sprintf("%03d-UpBlock_%s", layerNum, shapeToStr(x))
		layerNum++
		nanLogger.PushScope(scopeName)
		numChannels := numChannelsList[len(numChannelsList)-(ii+1)]
		x, skips = UpBlock(ctx.In(scopeName), x, contextFeatures, skips, numBlocks, numChannels)
		nanLogger.PopScope()
	}
	if len(skips) != 0 {
		exceptions.Panicf("Ended with %d skips not accounted for!?", len(skips))
	}

	// Output initialized to 0, which is the mean of the target.
	scopeName := fmt.Sprintf("%03d-Readout_%s", layerNum, shapeToStr(x))
	layerNum++
	x = layers.DenseWithBias(ctx.In(scopeName), x, imageChannels)
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
func DiffusionSchedule(ctx *context.Context, times *Node, clipStart bool) (signalRatios, noiseRatios *Node) {
	// diffusion times -> angles
	startAngle := 0.0
	if clipStart {
		startAngle = math.Acos(context.GetParamOr(ctx, "diffusion_max_signal_ratio", 0.95))
	}

	endAngle := math.Acos(context.GetParamOr(ctx, "diffusion_min_signal_ratio", 0.02))
	diffusionAngles := AddScalar(MulScalar(times, endAngle-startAngle), startAngle)

	// The ratios typically used is Sqrt(alpha) and Sqrt(1-alpha), because it has the nice property of preserving
	// the variance (of 1) during the process.
	signalRatios = Cos(diffusionAngles)
	noiseRatios = Sin(diffusionAngles)
	return
}

// Denoise tries to separate the noise from the image.
// It is given the signal and noise ratios.
func Denoise(ctx *context.Context, noisyImages, signalRatios, noiseRatios, flowerIds *Node) (
	predictedImages, predictedNoises *Node) {

	// Noise variance: since the noise is expected to have variance 1, the adjusted
	// variance to the noiseRatio (just a multiplicative factor), the new variance
	// is:
	noiseVariances := Square(noiseRatios)

	// It's easy to model the noise than the image:
	predictedNoises = UNetModelGraph(ctx, noisyImages, noiseVariances, flowerIds)
	predictedImages = Sub(noisyImages, Mul(predictedNoises, noiseRatios))
	predictedImages = Div(predictedImages, signalRatios)
	return
}

// BuildTrainingModelGraph builds the model for training and evaluation.
func (c *Config) BuildTrainingModelGraph() train.ModelFn {
	return func(ctx *context.Context, _ any, inputs []*Node) []*Node {
		g := inputs[0].Graph()

		// Prepare the input image and noise.
		images := inputs[0]
		flowerIds := inputs[2]
		batchSize := images.Shape().Dimensions[0]

		images = c.PreprocessImages(images, true)
		noises := ctx.RandomNormal(g, images.Shape())
		nanLogger.Trace(images, "images")
		nanLogger.Trace(noises, "noises")

		// Sample noise at different schedules.
		diffusionTimes := ctx.RandomUniform(g, shapes.Make(DType, batchSize, 1, 1, 1))
		diffusionTimes = Square(diffusionTimes) // Bias towards less noise (smaller diffusion times), since it's most impactful
		signalRatios, noiseRatios := DiffusionSchedule(ctx, diffusionTimes, true)
		noisyImages := Add(
			Mul(images, signalRatios),
			Mul(noises, noiseRatios))
		noisyImages = StopGradient(noisyImages)
		predictedImages, predictedNoises := Denoise(ctx, noisyImages, signalRatios, noiseRatios, flowerIds)

		// Calculate our custom loss: mean absolute error from the noise to the predictedNoise.
		var lossFn train.LossFn
		lossName := context.GetParamOr(ctx, "diffusion_loss", "mae")
		switch lossName {
		case "mae":
			lossFn = losses.MeanAbsoluteError
		case "mse":
			lossFn = losses.MeanSquaredError
		case "huber":
			lossFn = losses.MakeHuberLoss(context.GetParamOr(ctx, "huber_delta", 0.2))
		default:
			exceptions.Panicf("Invalid value for --loss=%q. Valid values are \"mae\", \"mse\" or \"huber\"", lossName)
		}
		noisesLoss := lossFn([]*Node{noises}, []*Node{predictedNoises})
		if !noisesLoss.IsScalar() {
			noisesLoss = ReduceAllMean(noisesLoss)
		}
		imagesLoss := lossFn([]*Node{images}, []*Node{predictedImages})
		if !imagesLoss.IsScalar() {
			imagesLoss = ReduceAllMean(imagesLoss)
		}

		return []*Node{c.DenormalizeImages(predictedImages), noisesLoss, imagesLoss}
	}
}

var (
	flagDropoutRate         = flag.Float64("dropout", 0.15, "Dropout rate")
	flagNumAttHeads         = flag.Int("att_heads", 4, "Number of attention heads, if --att_layers > 0.")
	flagNumAttLayers        = flag.Int("att_layers", 0, "Number of stacked attention layers. Set to 0 to disable.")
	flagAttPosEmbedSize     = flag.Int("att_pos_embed", 8, "Size of learned embedding.")
	flagAttKeyQueryEmbedDim = flag.Int("att_key_dim", 8, "Dimension of the Key/Query attention embedding.")
)

// TransformerBlock takes embed shaped `[batchDim, spatialDim, embedDim]`, where the spatial dimension is
// the combined dimensions of the image.
func TransformerBlock(ctx *context.Context, x *Node) *Node {
	if *flagNumAttLayers == 0 {
		return x
	}
	g := x.Graph()
	batchDim := x.Shape().Dimensions[0]
	embedDim := x.Shape().Dimensions[3]

	// Collapse spatial dimensions of the image.
	embed := Reshape(x, batchDim, -1, embedDim)
	shape := embed.Shape()
	spatialDim := shape.Dimensions[1]

	var dropoutRate *Node
	if *flagDropoutRate > 0 {
		dropoutRate = ConstAsDType(g, DType, *flagDropoutRate)
	}

	// Create positional embedding variable: it is 1 in every axis, but for the
	// sequence dimension -- there will be one embedding per position.
	// Shape: [1, maxLen, embedDim]
	posEmbedShape := shapes.Make(DType, 1, spatialDim, *flagAttPosEmbedSize)
	posEmbedVar := ctx.VariableWithShape("positional", posEmbedShape)
	posEmbed := posEmbedVar.ValueGraph(g)
	posEmbed = BroadcastToDims(posEmbed, batchDim, spatialDim, *flagAttPosEmbedSize) // Broadcast positional embeddings to each example in batch.

	// Add the requested number of attention layers.
	for ii := 0; ii < *flagNumAttLayers; ii++ {
		// Each layer in its own scope.
		ctx := ctx.In(fmt.Sprintf("AttLayer_%d", ii))
		residual := embed
		embed = Concatenate([]*Node{embed, posEmbed}, -1)
		embed = layers.MultiHeadAttention(ctx, embed, embed, embed, *flagNumAttHeads, *flagAttKeyQueryEmbedDim).
			SetOutputDim(embedDim).
			SetValueHeadDim(embedDim).Done()
		nanLogger.Trace(embed)
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = NormalizeLayer(ctx.In("normalization_1"), embed)
		attentionOutput := embed

		// Transformers recipe: 2 dense layers after attention.
		embed = layers.Dense(ctx.In("ffn_1"), embed, true, embedDim)
		embed = activations.ApplyFromContext(ctx, embed)
		embed = layers.Dense(ctx.In("ffn_2"), embed, true, embedDim)
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = Add(embed, attentionOutput)
		embed = NormalizeLayer(ctx.In("normalization_2"), embed)

		// Residual connection: not part of the usual transformer layer ...
		if ii > 0 {
			embed = Add(residual, embed)
			nanLogger.Trace(embed)
		}
	}
	x = Reshape(embed, batchDim, x.Shape().Dimensions[1], x.Shape().Dimensions[2], -1)
	return x
}
