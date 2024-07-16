package diffusion

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestUNetModelGraph(t *testing.T) {
	manager = graphtest.BuildTestBackend()
	Init()
	g := manager.NewGraph().WithName("test")
	ctx := context.NewContext(manager)
	numExamples := 5
	noisyImages := Zeros(g, shapes.Make(DType, numExamples, 64, 64, 3))
	flowerIds := Zeros(g, shapes.Make(dtypes.Int32, numExamples))
	fmt.Printf("  noisyImages.shape:\t%s\n", noisyImages.Shape())
	filtered := UNetModelGraph(ctx, noisyImages, Ones(g, shapes.Make(DType, numExamples, 1, 1, 1)), flowerIds)
	assert.True(t, noisyImages.Shape().Equal(filtered.Shape()), "Filtered images after UNetModelGraph should have the same shape as its input images")
	fmt.Printf("     filtered.shape:\t%s\n", filtered.Shape())
	fmt.Printf("U-Net Model #params:\t%d\n", ctx.NumParameters())
	fmt.Printf(" U-Net Model memory:\t%s\n", data.ByteCountIEC(ctx.Memory()))
}

// getZeroPredictions calls the TrainModelGraph on an empty context and graph.
// This can be used to check the predictions shape and also as a side effect to create
// the variables in the context `ctx`.
func getZeroPredictions(ctx *context.Context, g *Graph, numExamples int) []*Node {
	images := Zeros(g, shapes.Make(DType, numExamples, ImageSize, ImageSize, 3))
	imageIds := Zeros(g, shapes.Make(dtypes.Int32, numExamples))
	flowerIds := Zeros(g, shapes.Make(dtypes.Int32, numExamples))
	return TrainingModelGraph(ctx, nil, []*Node{images, imageIds, flowerIds})
}

func TestTrainingModelGraph(t *testing.T) {
	if testing.Short() {
		fmt.Println("TestTrainingModelGraph skipped with go test -short: it requires downloading and preprocessing data.")
		return
	}
	manager = graphtest.BuildTestBackend()
	Init()
	g := manager.NewGraph().WithName("test")
	ctx := context.NewContext(manager)
	numExamples := 5
	predictions := getZeroPredictions(ctx, g, numExamples)
	predictedImages, loss := predictions[0], predictions[1]
	assert.NoError(t, predictedImages.Shape().CheckDims(numExamples, ImageSize, ImageSize, 3))
	assert.True(t, loss.Shape().IsScalar(), "Loss must be scalar.")
	assert.Greater(t, ctx.NumParameters(), 0, "No context parameters created!?")
	fmt.Printf("predictedImages.shape:\t%s\n", predictions[0].Shape())
	fmt.Printf("           loss.shape:\t%s\n", predictions[1].Shape())
	fmt.Printf("        Model #params:\t%d\n", ctx.NumParameters())
	fmt.Printf("         Model memory:\t%s\n", data.ByteCountIEC(ctx.Memory()))
}

func TestImagesGenerator(t *testing.T) {
	Init()
	if testing.Short() {
		fmt.Println("TestGenerateImages skipped with go test -short: it requires downloading and preprocessing data.")
		return
	}

	numImages := 5
	numDiffusionSteps := 3

	manager = graphtest.BuildTestBackend()
	ctx := context.NewContext(manager)
	// ctx.RngStateReset() --> to truly randomize each run uncomment this.
	g := manager.NewGraph().WithName("test")
	_ = getZeroPredictions(ctx, g, 2) // Batch size won't matter, we only call this to create the model weights.
	noise := GenerateNoise(numImages)
	flowerIds := GenerateFlowerIds(numImages)
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)

	// Just the final images:
	images := generator.Generate()
	assert.NoError(t, images.Shape().CheckDims(numImages, ImageSize, ImageSize, 3))

	// With intermediary images:
	allImages, diffusionSteps, diffusionTimes := generator.GenerateEveryN(1)
	assert.Len(t, allImages, numDiffusionSteps)
	assert.EqualValues(t, []int{0, 1, 2}, diffusionSteps)
	assert.InDeltaSlice(t, []float64{0.6666666, .333333333, 0.0}, diffusionTimes, 0.001)

	for _, images = range allImages {
		assert.NoError(t, images.Shape().CheckDims(numImages, ImageSize, ImageSize, 3))
	}
}
