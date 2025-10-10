package inceptionv3

import (
	"fmt"
	"image"
	"os"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
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

var (
	noisyImagesExec     *Exec
	noisyImagesExecOnce sync.Once
)

// noisyImages add noise to the batch of images. The noise is simply an increasing
// value from -127.5 in the top left to 127.5 in the bottom right. It's deterministic.
func noisyImages(t *testing.T, manager backends.Backend, batch *tensors.Tensor) *tensors.Tensor {
	noisyImagesExecOnce.Do(func() {
		noisyImagesExec = MustNewExec(manager, func(batch *Node) *Node {
			g := batch.Graph()
			oneImage := batch.Shape().Clone()
			oneImage.Dimensions[0] = 1
			noise := IotaFull(g, oneImage)
			scale := 255.0 / float64(noise.Shape().Size())
			noise = AddScalar(MulScalar(noise, scale), -127.5)
			noisyBatch := Add(batch, noise)
			noisyBatch = ClipScalar(noisyBatch, 0.0, 255.0)
			return noisyBatch
		})
	})
	return noisyImagesExec.Call(batch)[0]
}

func TestKidMetric(t *testing.T) {
	if testing.Short() {
		fmt.Println("- github.com/gomlx/gomlx/models/inceptionv3: TestKidMetrics disabled for go test --short because it requires downloading a large file with weights.")
		return
	}
	require.NoError(t, DownloadAndUnpackWeights(*flagDataDir))
	manager := graphtest.BuildTestBackend()

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
	imagesBatch := images.ToTensor(dtypes.Float32).MaxValue(255.0).Batch(Images)
	noisyBatch := noisyImages(t, manager, imagesBatch)

	kidBuilder := NewKidBuilder(*flagDataDir, 75, 255.0, images.ChannelsLast)
	ctx := context.New()
	kidExec := context.MustNewExec(manager, ctx, func(ctx *context.Context, imagesPair []*Node) *Node {
		return kidBuilder.BuildGraph(ctx, []*Node{imagesPair[0]}, []*Node{imagesPair[1]})
	})
	kid := kidExec.Call(imagesBatch, noisyBatch)[0].Value().(float32)
	require.InDelta(t, -1.5861, kid, 0.001, "KID value different from expected for batch.")
}
