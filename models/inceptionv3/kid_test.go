package inceptionv3

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/stretchr/testify/require"
	"image"
	"os"
	"testing"
)

func loadImage(filePath string) (img image.Image, err error) {
	imgFile, err := os.Open(filePath)
	if err != nil {
		return
	}
	defer func() { _ = imgFile.Close() }()

	img, _, err = image.Decode(imgFile)
	return
}

// noisyImages add noise to the batch of images. The noise is simply an increasing
// value from -127.5 in the top left to 127.5 in the bottom right. It's deterministic.
func noisyImages(t *testing.T, manager *Manager, batch tensor.Tensor) tensor.Tensor {
	results, err := NewExec(manager, func(batch *Node) *Node {
		g := batch.Graph()
		oneImage := batch.Shape().Copy()
		oneImage.Dimensions[0] = 1
		noise := IotaFull(g, oneImage)
		scale := 255.0 / float64(noise.Shape().Size())
		noise = AddScalar(MulScalar(noise, scale), -127.5)
		noisyBatch := Add(batch, noise)
		noisyBatch = ClipScalar(noisyBatch, 0.0, 255.0)
		return noisyBatch
	}).Call(batch)
	require.NoError(t, err)
	return results[0]
}

func TestKidMetric(t *testing.T) {
	require.NoError(t, DownloadAndUnpackWeights(*flagDataDir))
	manager := graphtest.BuildTestManager()

	ImagePaths := []string{
		"gomlx_gopher_299.png",
		"zurich_see_299.png",
	}
	var Images []image.Image

	// Load images into a batch.
	var err error
	Images = make([]image.Image, len(ImagePaths))
	for ii, p := range ImagePaths {
		Images[ii], err = loadImage(p)
		require.NoError(t, err)
	}
	imagesBatch := timage.ToTensor(shapes.F32).MaxValue(255.0).Batch(Images)
	noisyBatch := noisyImages(t, manager, imagesBatch)

	kidBuilder := NewKidBuilder(*flagDataDir, 75, 255.0, timage.ChannelsLast)
	ctx := context.NewContext(manager)
	kidExec := context.NewExec(manager, ctx, func(ctx *context.Context, images []*Node) *Node {
		return kidBuilder.BuildGraph(ctx, []*Node{images[0]}, []*Node{images[1]})
	})
	results, err := kidExec.Call(imagesBatch, noisyBatch)
	require.NoError(t, err)
	kid := results[0].Local().Value().(float32)
	require.InDelta(t, -1.5861, kid, 0.001, "KID value different than expected for batch.")
}
