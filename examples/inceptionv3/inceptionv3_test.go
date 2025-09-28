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
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

var flagDataDir = flag.String("data", "/tmp/gomlx_inceptionv3", "Directory where to save and load model data.")

func TestBuildGraph(t *testing.T) {
	if testing.Short() {
		fmt.Println("- github.com/gomlx/gomlx/models/inceptionv3: TestBuildGraph disabled for go test --short because it requires downloading a large file with weights.")
		return
	}
	backend := graphtest.BuildTestBackend()

	// Load GoMLX gopher mascot image and scale to inception's size
	// for classification (299x299 == ClassificationImageSize).
	f, err := os.Open("gomlx_gopher_299.png")
	require.NoError(t, err)
	img, _, err := image.Decode(f)
	require.NoError(t, err)
	require.NoError(t, f.Close())
	var imgT *tensors.Tensor
	imgT = images.ToTensor(dtypes.Float32).MaxValue(255.0).Single(img)
	fmt.Printf("\tImage shape=%s\n", imgT.Shape())

	// Download InceptionV3 weights.
	require.NoError(t, DownloadAndUnpackWeights(*flagDataDir))

	// InceptionV3 classification.
	ctx := context.New()
	inceptionV3Exec := context.NewExec(backend, ctx, func(ctx *context.Context, img *Node) *Node {
		img = InsertAxes(img, 0) // Add batch dimension
		img = PreprocessImage(img, 255.0, images.ChannelsLast)
		output := BuildGraph(ctx, img).
			PreTrained(*flagDataDir).
			ClassificationTop(true).
			WithAliases(true).
			Done()
		// checks that node aliases were created.
		g := output.Graph()
		fmt.Printf("\tAliased nodes in the graph:\n")
		for _, node := range g.IterAliasedNodes() {
			fmt.Printf("\t\t%s\n", node)
		}
		require.True(t, g.GetNodeByAlias("/inceptionV3/logits") == output)
		require.True(t, g.GetNodeByAlias("/inceptionV3/conv_000/output") != nil)
		// ...
		require.True(t, g.GetNodeByAlias("/inceptionV3/conv_093/output") != nil)
		return output
	})
	predictionT := inceptionV3Exec.Call(imgT)[0]
	prediction := predictionT.Value().([][]float32)[0] // The last [0] takes the first element of the batch of 1.

	// Compare with the expected result:
	wantT, err := tensors.Load("gomlx_gopher_classification_output.bin")
	require.NoError(t, err)
	want := wantT.Value().([][]float32)[0] // The last [0] takes the first element of the batch of 1.

	diffStats := MustNewExec(backend, func(a, b *Node) (max, mean *Node) {
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
