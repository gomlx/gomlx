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
	manager = graphtest.BuildTestManager()
	Init()
	g := manager.NewGraph("test")
	ctx := context.NewContext(manager)
	numExamples := 5
	noisyImages := Zeros(g, shapes.Make(DType, numExamples, 64, 64, 3))
	flowerIds := Zeros(g, shapes.Make(shapes.I32, numExamples))
	fmt.Printf("  noisyImages.shape:\t%s\n", noisyImages.Shape())
	filtered := UNetModelGraph(ctx, noisyImages, Ones(g, shapes.Make(DType, numExamples, 1, 1, 1)), flowerIds)
	assert.True(t, noisyImages.Shape().Eq(filtered.Shape()), "Filtered images after UNetModelGraph should have the same shape as its input images")
	fmt.Printf("     filtered.shape:\t%s\n", filtered.Shape())
	fmt.Printf("U-Net Model #params:\t%d\n", ctx.NumParameters())
	fmt.Printf(" U-Net Model memory:\t%s\n", data.ByteCountIEC(ctx.Memory()))
}

func TestTrainingModelGraph(t *testing.T) {
	manager = graphtest.BuildTestManager()
	Init()
	g := manager.NewGraph("test")
	ctx := context.NewContext(manager)
	numExamples := 5
	images := Zeros(g, shapes.Make(DType, numExamples, ImageSize, ImageSize, 3))
	imageIds := Zeros(g, shapes.Make(shapes.I32, numExamples))
	flowerIds := Zeros(g, shapes.Make(shapes.I32, numExamples))
	fmt.Printf("         images.shape:\t%s\n", images.Shape())
	predictions := TrainingModelGraph(ctx, nil, []*Node{images, imageIds, flowerIds})
	predictedImages, loss := predictions[0], predictions[1]
	assert.True(t, predictedImages.Shape().Eq(images.Shape()), "Original images and predicted images shape differ!?")
	assert.True(t, loss.Shape().IsScalar(), "Loss must be scalar.")
	assert.Greater(t, ctx.NumParameters(), 0, "No context parameters created!?")
	fmt.Printf("predictedImages.shape:\t%s\n", predictions[0].Shape())
	fmt.Printf("           loss.shape:\t%s\n", predictions[1].Shape())
	fmt.Printf("        Model #params:\t%d\n", ctx.NumParameters())
	fmt.Printf("         Model memory:\t%s\n", data.ByteCountIEC(ctx.Memory()))
}

func TestGenerateImages(t *testing.T) {
	numImages := 5
	images := GenerateImages(numImages, 3, 0)
	assert.Equal(t, []int{numImages, ImageSize, ImageSize, 3}, images.Shape().Dimensions)
}
