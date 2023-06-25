package diffusion

import (
	"flag"
	"fmt"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"math"
	"strconv"
)

var (
	manager *Manager
	DType   shapes.DType
)

var (
	flagEmbeddingDims           = flag.Int("embedding_dim", 32, "Size of the sinusoidal embeddings, should be an even number")
	flagEmbeddingMaxFrequency   = flag.Float64("embed_max_freq", 1000.0, "Embedding max frequency")
	flagEmbeddingMinFrequency   = flag.Float64("embed_min_freq", 1.0, "Embedding max frequency")
	flagFlowerTypeEmbeddingSize = flag.Int("flower_type_dim", 0, "If > 0, use embedding of the flower type of the given dimension.")

	flagNanLogger = flag.Bool("nan_debug", false, "If set to true, it will add some traces monitoring for NaN values, "+
		"and it will report the first location where it happens. It slows down a bit the training.")
)

var nanLogger = nanlogger.New()

// SinusoidalEmbedding provides embeddings of `x` for different frequencies.
// This is applied to the variance of the noise, and facilitates the NN model to easily map different ranges
// of the signal/noise ratio.
func SinusoidalEmbedding(x *Node) *Node {
	Init()

	g := x.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}

	// Generate geometrically spaced frequencies: only 1/2 of *flagEmbeddingDims because we use half for Sine, half for Cosine.
	halfEmbed := *flagEmbeddingDims / 2
	logMinFreq := math.Log(*flagEmbeddingMinFrequency)
	logMaxFreq := math.Log(*flagEmbeddingMaxFrequency)
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
	flagChannelsList = slices.Flag("channels_list", []int{32, 64, 96, 128},
		"Number of channels (features) for each image size (progressively smaller) in U-Net model",
		strconv.Atoi)
	flagNumBlocks     = flag.Int("blocks", 2, "Number of blocks per image size in U-Net model")
	flagActivation    = flag.String("activation", "swish", "One of: \"swish\", \"sigmoid\", \"tanh\" or \"relu\"")
	flagNormalization = flag.String("norm", "batch", "One of: \"none\", \"batch\" or \"layer\"")
)

// NormalizeLayer behaves according to the `--norm` flag.
// It works with `x` with rank 4 and rank 3.
func NormalizeLayer(ctx *context.Context, x *Node) *Node {
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	switch *flagNormalization {
	case "none":
		// No-op.
	case "batch":
		x = layers.BatchNormalization(ctx, x, -1).Center(false).Scale(false).Done()
	case "layer":
		x = layers.LayerNormalization(ctx, x, -1).Done()
	}
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	return x
}

// ActivationLayer can be configured.
func ActivationLayer(x *Node) *Node {
	x = layers.Activation(*flagActivation, x)
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	return x
}

// ResidualBlock on the input with `outputChannels` (axis 3) in the output.
//
// The parameter `x` must be of rank 4, shaped `[batchSize, height, width, channels]`.
func ResidualBlock(ctx *context.Context, x *Node, outputChannels int) *Node {
	g := x.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !x.AssertRank(4) {
		return g.InvalidNode()
	}
	inputChannels := x.Shape().Dimensions[3]
	residual := x
	if inputChannels != outputChannels {
		residual = layers.DenseWithBias(ctx.In("residual_channels"), x, outputChannels)
	}
	x = NormalizeLayer(ctx, x)
	x = layers.Convolution(ctx.In("conv-layer-1"), x).Filters(outputChannels).KernelSize(3).PadSame().Done()
	x = ActivationLayer(x)
	x = layers.Convolution(ctx.In("conv-layer-2"), x).Filters(outputChannels).KernelSize(3).PadSame().Done()
	x = Add(x, residual)
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	return x
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
	if !g.Ok() {
		return g.InvalidNode()
	}
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
		if *flagNanLogger {
			nanLogger.Trace(embed)
		}
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = NormalizeLayer(ctx.In("normalization_1"), embed)
		attentionOutput := embed

		// Transformers recipe: 2 dense layers after attention.
		embed = layers.Dense(ctx.In("ffn_1"), embed, true, embedDim)
		embed = ActivationLayer(embed)
		embed = layers.Dense(ctx.In("ffn_2"), embed, true, embedDim)
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = Add(embed, attentionOutput)
		embed = NormalizeLayer(ctx.In("normalization_2"), embed)

		// Residual connection: not part of the usual transformer layer ...
		if ii > 0 {
			embed = Add(residual, embed)
			if *flagNanLogger {
				nanLogger.Trace(embed)
			}
		}
	}
	x = Reshape(embed, batchDim, x.Shape().Dimensions[1], x.Shape().Dimensions[2], -1)
	return x
}

// DownBlock applies `numBlocks` residual blocks followed by an average pooling of size 2, halfing the spatial size.
// It pushes the values between each residual blocks to the `skips` stack, to build the skip connections later.
//
// It returns the transformed `x` and `skips` with newly stacked skip connections.
func DownBlock(ctx *context.Context, x *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	for ii := 0; ii < numBlocks; ii++ {
		name := fmt.Sprintf("down-residual-%d", ii+1)
		x = ResidualBlock(ctx.In(name), x, outputChannels)
		skips = append(skips, x)
	}
	x = MeanPool(x).Window(2).NoPadding().Done()
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	return x, skips
}

// UpBlock is the counter-part to DownBlock. It performs up-scaling convolutions and connects skip-connections popped
// from `skips`.
//
// It returns `x` and `skips` after popping the consumed skip connections.
func UpBlock(ctx *context.Context, x *Node, skips []*Node, numBlocks, outputChannels int) (*Node, []*Node) {
	x = Interpolate(x, timage.GetUpSampledSizes(x, timage.ChannelsLast, 2)...).Nearest().Done()
	for ii := 0; ii < numBlocks; ii++ {
		name := fmt.Sprintf("up-residual-%d", ii+1)
		var skip *Node
		skip, skips = slices.Pop(skips)
		x = Concatenate([]*Node{x, skip}, -1)
		x = ResidualBlock(ctx.In(name), x, outputChannels)
	}
	if *flagNanLogger {
		nanLogger.Trace(x)
	}
	return x, skips
}

// FlowerTypeEmbedding will if configured (`--flower_type_dim` flag), concatenate a flower embedding to `x`.
// If `--flower_type_dim==0` it returns `x` unchanged.
func FlowerTypeEmbedding(ctx *context.Context, flowerIds, x *Node) *Node {
	if *flagFlowerTypeEmbeddingSize <= 0 {
		return x
	}
	flowerIds = ExpandDims(flowerIds, -1, -1, -1) // Expand axis to the match noisyImages rank.
	flowerTypeEmbed := layers.Embedding(ctx, flowerIds, DType, flowers.NumLabels, *flagFlowerTypeEmbeddingSize)
	broadcastDims := flowerTypeEmbed.Shape().Copy().Dimensions
	for _, axis := range timage.GetSpatialAxes(x, timage.ChannelsLast) {
		broadcastDims[axis] = x.Shape().Dimensions[axis]
	}
	flowerTypeEmbed = BroadcastToDims(flowerTypeEmbed, broadcastDims...)
	x = Concatenate([]*Node{x, flowerTypeEmbed}, -1)
	return x
}

// UNetModelGraph builds the U-Net model.
//
// Parameters:
//   - noisyImages: image shaped `[batch_size, size, size, channels=3]`.
//   - noiseVariance: One value per example in the batch, shaped `[batch_size, 1, 1, 1]`.
//   - numChannelsList (static hyperparameter): number of channels to use in the model. For each value `numBlocks` are applied
//     and then the image is pooled and reduced by a factor of 2 -- later to be up-sampled again. So at most `log2(size)` values.
//   - numBlocks (static hyperparameter): number of blocks to use per numChannelsList element.
func UNetModelGraph(ctx *context.Context, noisyImages, noiseVariances, flowerIds *Node) *Node {
	Init()

	// Parameters from flags.
	numChannelsList := *flagChannelsList
	numBlocks := *flagNumBlocks

	g := noisyImages.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if *flagNanLogger {
		nanLogger.Trace(noisyImages)
		nanLogger.Trace(noiseVariances)
	}

	// Adjust imageChannels to initial num channels.
	imageChannels := slices.Last(noisyImages.Shape().Dimensions)
	x := layers.DenseWithBias(ctx.In("starting_channels"), noisyImages, numChannelsList[0])
	concatFeatures := []*Node{x}

	// Get sinusoidal features, always included, and broadcast them to the spatial dimensions.
	sinEmbed := SinusoidalEmbedding(noiseVariances)
	if *flagNanLogger {
		nanLogger.Trace(sinEmbed)
	}
	broadcastDims := sinEmbed.Shape().Copy().Dimensions
	for _, axis := range timage.GetSpatialAxes(noisyImages, timage.ChannelsLast) {
		broadcastDims[axis] = noisyImages.Shape().Dimensions[axis]
	}
	sinEmbed = BroadcastToDims(sinEmbed, broadcastDims...)
	concatFeatures = append(concatFeatures, sinEmbed)

	// Concatenate channels with extra features.
	x = Concatenate(concatFeatures, -1)

	// Downward: keep pooling image to a smaller size.
	// Keep the `skips` features as we move "downward," so they can be "skip" connected later as we move upward.
	skips := make([]*Node, 0, numBlocks*len(numChannelsList))
	for ii, numChannels := range numChannelsList {
		nanLogger.PushScope(fmt.Sprintf("DownBlock-%d", ii))
		name := fmt.Sprintf("DownBlock-%d", ii+1)
		// Use flower types as an extra embedding.
		x = FlowerTypeEmbedding(ctx.In(fmt.Sprintf("DownBlock-%d-flowerIds", ii)), flowerIds, x)
		x, skips = DownBlock(ctx.In(name), x, skips, numBlocks, numChannels)
		nanLogger.PopScope()
	}

	// Intermediary fixed size blocks.
	x = TransformerBlock(ctx.In("transformer_block"), x)
	lastNumChannels := slices.Last(numChannelsList)
	for ii := 0; ii < numBlocks; ii++ {
		nanLogger.PushScope(fmt.Sprintf("IntermediaryBlock-%d", ii))
		name := fmt.Sprintf("IntermediaryResidual-%d", ii+1)
		x = ResidualBlock(ctx.In(name), x, lastNumChannels)
		nanLogger.PopScope()
	}

	//fmt.Printf("Intermediary (smallest) shape: %s\n", x.Shape())

	// Upward: up-sample image back to original size, one block at a time.
	for ii := range numChannelsList {
		nanLogger.PushScope(fmt.Sprintf("UpBlock-%d", ii))
		name := fmt.Sprintf("UpBlock-%d", ii+1)
		numChannels := numChannelsList[len(numChannelsList)-(ii+1)]
		x, skips = UpBlock(ctx.In(name), x, skips, numBlocks, numChannels)
		nanLogger.PopScope()
	}
	if len(skips) != 0 {
		g.SetErrorf("Ended with %d skips not accounted for!?", len(skips))
		return g.InvalidNode()
	}

	// Output initialized to 0, which is the mean of the target.
	ctxReadOut := ctx.In("Readout").WithInitializer(initializers.Zero)
	x = layers.DenseWithBias(ctxReadOut, x, imageChannels)
	return x
}

var (
	flagMinSignalRatio = flag.Float64("min_signal_ratio", 0.02, "minimum of the signal to noise ratio when training.")
	flagMaxSignalRatio = flag.Float64("max_signal_ratio", 0.95, "maximum of the signal to noise ratio when training.")
)

// DiffusionSchedule calculates a ratio of noise and image that needs to be mixed,
// given the diffusion time `~ [0.0, 1.0]`.
// Diffusion time 0 means minimum diffusion -- the signal ratio will be set to -max_signal_ratio, default to 0.95 -- and
// diffusion time 1.0 means almost all noise -- the signal ratio will be set to -min_signal_ratio, default to 0.02.
// The returned ratio has the sum of their square total 1.
//
// Typically, the shape of `time` and the returned ratios will be `[batch_size, 1, 1, 1]`.
//
// If `clipStart` is set to false, the signal ratio is not clipped, and it can go all the way to 1.0.
func DiffusionSchedule(times *Node, clipStart bool) (signalRatios, noiseRatios *Node) {
	// diffusion times -> angles
	startAngle := 0.0
	if clipStart {
		startAngle = math.Acos(*flagMaxSignalRatio)
	}

	endAngle := math.Acos(*flagMinSignalRatio)
	diffusionAngles := AddScalar(MulScalar(times, endAngle-startAngle), startAngle)

	// sin^2(x) + cos^2(x) = 1
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

var (
	flagLoss = flag.String("loss", "mae", "Use 'mse' for mean squared error or 'mae' for mean absolute error")
)

// TrainingModelGraph builds the model for training and evaluation.
func TrainingModelGraph(ctx *context.Context, _ any, inputs []*Node) []*Node {
	g := inputs[0].Graph()

	// Prepare the input image and noise.
	images := inputs[0]
	flowerIds := inputs[2]
	batchSize := images.Shape().Dimensions[0]
	images = PreprocessImages(images, true)
	noises := ctx.RandomNormal(g, images.Shape())
	if *flagNanLogger {
		nanLogger.Trace(images, "images")
		nanLogger.Trace(noises, "noises")
	}

	// Sample noise at different schedules.
	diffusionTimes := ctx.RandomUniform(g, shapes.Make(DType, batchSize, 1, 1, 1))
	diffusionTimes = Square(diffusionTimes) // Bias towards less noise (smaller diffusion times), since it's most impactful
	signalRatios, noiseRatios := DiffusionSchedule(diffusionTimes, true)
	noisyImages := Add(
		Mul(images, signalRatios),
		Mul(noises, noiseRatios))
	noisyImages = StopGradient(noisyImages)
	predictedImages, predictedNoises := Denoise(ctx, noisyImages, signalRatios, noiseRatios, flowerIds)

	// Calculate our custom loss: mean absolute error from the noise to the predictedNoise.
	var lossFn train.LossFn
	switch *flagLoss {
	case "mae":
		lossFn = losses.MeanAbsoluteError
	case "mse":
		lossFn = losses.MeanSquaredError
	default:
		g.SetErrorf("Invalid value for --loss=%q. Valid values are \"mae\" or \"mse\"", *flagLoss)
		return nil
	}
	noisesLoss := lossFn([]*Node{noises}, []*Node{predictedNoises})
	imagesLoss := lossFn([]*Node{images}, []*Node{predictedImages})
	return []*Node{DenormalizeImages(predictedImages), noisesLoss, imagesLoss}
}
