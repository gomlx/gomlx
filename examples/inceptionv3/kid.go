package inceptionv3

import (
	. "github.com/gomlx/gomlx/graph"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
)

// KidMetric returns a metric that takes a generated image and a label image and returns a
// measure of similarity.
//
// [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) was proposed as a replacement for the popular
// [Frechet Inception Distance (FID) metric](https://arxiv.org/abs/1706.08500) for measuring image generation quality.
// Both metrics measure the difference in the generated and training distributions in the representation space of
// an InceptionV3 network pretrained on ImageNet.
//
// The implementation is based on the Keras one, described in
// https://keras.io/examples/generative/ddim/
//
// To directly calculate KID, as opposed to using it as a metric, see NewKidBuilder below.
//
// Parameters:
//
//   - `dataDir`: directory where to download and unpack the InceptionV3 weights.
//     They are reused from there in subsequent calls.
//   - `kidImageSize`: resize input images (labels and predictions) to `kidImageSize x kidImageSize` before
//     running the Kid metric calculation. It should be between 75 and 299. Smaller values make the metric faster.
//   - `maxImageValue`: Maximum value the images can take at any channel -- If set to 0
//     it doesn't rescale the pixel values, and the images are expected to have values between -1.0 and 1.0.
//     Passed to `PreprocessImage` function.
//   - `channelsConfig`: informs what is the channels axis, commonly set to `images.ChannelsLast`.
//     Passed to `PreprocessImage` function.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
func KidMetric(dataDir string, kidImageSize int, maxImageValue float64, channelsConfig images.ChannelsAxisConfig) metrics.Interface {
	builder := NewKidBuilder(dataDir, kidImageSize, maxImageValue, channelsConfig)
	return metrics.NewMeanMetric("Kernel Inception Distance", "KID", "KID", builder.BuildGraph, nil)
}

// KidBuilder builds the graph to calculate [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) between
// two sets of images.
// See details in KidMetric.
type KidBuilder struct {
	dataDir        string
	kidImageSize   int
	maxValue       float64
	channelsConfig images.ChannelsAxisConfig
}

// NewKidBuilder configures a KidBuilder.
//
// KidBuilder builds the graph to calculate [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) between
// `labels` and `predictions` batches of images. The metric is normalized by the `labels` images, so it's not
// symmetric.
//
// See details in KidMetric.
//
//   - `dataDir`: directory where to download and unpack the InceptionV3 weights.
//     They are reused from there in subsequent calls.
//   - `kidImageSize`: resize input images (labels and predictions) to `kidImageSize x kidImageSize` before
//     running the Kid metric calculation. It should be between 75 and 299. Smaller values make the metric faster.
//   - `maxImageValue`: Maximum value the images can take at any channel -- If set to 0
//     it doesn't rescale the pixel values, and the images are expected to have values between -1.0 and 1.0.
//     Passed to `PreprocessImage` function.
//   - `channelsConfig`: informs what is the channels axis, commonly set to `images.ChannelsLast`.
//     Passed to `PreprocessImage` function.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
func NewKidBuilder(dataDir string, kidImageSize int, maxImageValue float64, channelsConfig images.ChannelsAxisConfig) *KidBuilder {
	return &KidBuilder{
		dataDir:        dataDir,
		kidImageSize:   kidImageSize,
		maxValue:       maxImageValue,
		channelsConfig: channelsConfig,
	}
}

// BuildGraph returns the mean KID score of two batches, see KidMetric.
//
// It returns a scalar with the mean distance of the images provided in labels and predictions.
// The images
func (builder *KidBuilder) BuildGraph(ctx *context.Context, labels, predictions []*Node) (output *Node) {
	// Sanity checking:
	g := predictions[0].Graph()
	dtype := predictions[0].DType()
	if len(labels) != 1 || len(predictions) != 1 {
		Panicf("KidMetric expects only one images tensor in labels and predictions, got %d and %d",
			len(labels), len(predictions))
	}
	if builder.kidImageSize < 75 || builder.kidImageSize > 299 {
		Panicf("KidMetric was configured with an invalid target image size (for KID calculation) of %d -- "+
			"valid values are between 75 and 299", builder.kidImageSize)
	}

	imagesPair := [2]*Node{labels[0], predictions[0]}
	imagesShape := imagesPair[0].Shape()
	if !imagesShape.Equal(imagesPair[1].Shape()) {
		Panicf("Labels (%s) and predictions (%s) have different shapes",
			imagesPair[0].Shape(), imagesPair[1].Shape())
	}

	// Checks whether we need to resize images.
	spatialAxis := images.GetSpatialAxes(imagesShape, builder.channelsConfig)
	var needsResizing bool
	for _, axis := range spatialAxis {
		if imagesShape.Dimensions[axis] != builder.kidImageSize {
			needsResizing = true
			break
		}
	}
	if needsResizing {
		// Resize to kidImageSize x kidImageSize:
		newSizes := imagesPair[0].Shape().Clone().Dimensions
		for _, axis := range spatialAxis {
			newSizes[axis] = builder.kidImageSize
		}
		for imgIdx := range imagesPair {
			imagesPair[imgIdx] = Interpolate(imagesPair[imgIdx], newSizes...).Done()
		}
	}

	// Standard preprocessing of the image for Inception V3.
	for imgIdx := range imagesPair {
		imagesPair[imgIdx] = PreprocessImage(imagesPair[imgIdx], builder.maxValue, builder.channelsConfig)
	}

	// Apply InceptionV3 model to each image.
	ctx = ctx.In("kid_metric").Checked(false)
	var features [2]*Node
	for imgIdx := range imagesPair {
		features[imgIdx] = BuildGraph(ctx, imagesPair[imgIdx]).
			SetPooling(MeanPooling).ClassificationTop(false).PreTrained(builder.dataDir).
			ChannelsAxis(builder.channelsConfig).Trainable(false).Done()
	}
	batchSize := features[0].Shape().Dimensions[0]
	crossKernels := polynomialKernel(features)
	realKernels := polynomialKernel([2]*Node{features[0], features[0]})
	generatedKernels := polynomialKernel([2]*Node{features[1], features[1]})

	// Mean of the real and generated kernels: exclude the diagonal.
	normalizationNoDiagonal := 1.0 / float64(batchSize*(batchSize-1))
	identityBatch := DiagonalWithValue(ScalarOne(g, dtype), batchSize)
	meanRealKernels := ReduceSum(Mul(realKernels, OneMinus(identityBatch)))
	meanRealKernels = MulScalar(meanRealKernels, normalizationNoDiagonal)
	meanGenerateKernels := ReduceSum(Mul(generatedKernels, OneMinus(identityBatch)))
	meanGenerateKernels = MulScalar(meanGenerateKernels, normalizationNoDiagonal)
	meanCrossKernels := ReduceAllMean(crossKernels)

	return Sub(
		Add(meanGenerateKernels, meanRealKernels),
		Add(meanCrossKernels, meanCrossKernels))
}

func polynomialKernel(features [2]*Node) *Node {
	features[0].AssertRank(2)
	batchSize := features[0].Shape().Dimensions[0]
	numFeatures := features[0].Shape().Dimensions[1]
	features[1].AssertDims(batchSize, numFeatures)
	output := EinsumAxes(features[0], features[1], [][2]int{{1, 1}}, nil)
	output.AssertDims(batchSize, batchSize) // Cross of the batchSize, contraction of the numFeatures.
	output = AddScalar(
		MulScalar(output, 1.0/float64(numFeatures)),
		1.0)
	output = PowScalar(output, 3.0)
	return output
}
