package inceptionv3

import (
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/gomlx/gomlx/types/tensor"
	"google3/third_party/golang/testify/assert/assert"
	"github.com/stretchr/testify/require"
)

var flagDataDir = flag.String("data", "/tmp/gomlx_inceptionv3", "Directory where to save and load model data.")

func TestBuildGraph(t *testing.T) {
	if !testing.Short() {
		fmt.Println("- github.com/gomlx/gomlx/models/inceptionv3: TestBuildGraph disabled for go test --short because it requires downloading a large file with weights.")
		return
	}
	manager := graphtest.BuildTestManager()

	// Load GoMLX gopher mascot image and scale to inception's size
	// for classification (299x299 == ClassificationImageSize).
	f, err := os.Open("gomlx_gopher_299.png")
	require.NoError(t, err)
	img, _, err := image.Decode(f)
	require.NoError(t, err)
	require.NoError(t, f.Close())
	var imgT tensor.Tensor
	imgT = timage.ToTensor(shapes.F32).MaxValue(255.0).Single(img)
	fmt.Printf("\tImage shape=%s\n", imgT.Shape())

	// Download InceptionV3 weights.
	require.NoError(t, DownloadAndUnpackWeights(*flagDataDir))

	// InceptionV3 classification.
	ctx := context.NewContext(manager)
	inceptionV3Exec := context.NewExec(manager, ctx, func(ctx *context.Context, img *Node) *Node {
		img = ExpandDims(img, 0) // Add batch dimension
		img = PreprocessImage(img, 255.0, timage.ChannelsLast)
		return BuildGraph(ctx, img).PreTrained(*flagDataDir).ClassificationTop(true).Done()
	})
	predictionT := inceptionV3Exec.Call(imgT)[0]
	prediction := predictionT.Value().([][]float32)[0] // The last [0] takes the first element of the batch of 1.

	// Compare with the expected result:
	wantT, err := tensor.Load("gomlx_gopher_classification_output.bin")
	require.NoError(t, err)
	want := wantT.Value().([][]float32)[0] // The last [0] takes the first element of the batch of 1.

	diffStats := NewExec(manager, func(a, b *Node) (max, mean *Node) {
		diff := Abs(Sub(a, b))
		max, mean = ReduceAllMax(diff), ReduceAllMean(diff)
		return
	})
	results := diffStats.Call(wantT, predictionT)
	fmt.Printf("\tPredictions difference to previous truth: max=%f, mean=%f\n",
		results[0].Value(), results[1].Value())

	// Assert difference is within a delta.
	assert.InDeltaSlice(t, want, prediction, 0.1,
		"InceptionV3 classification of our gopher image differs -- did the image change ?")
}
