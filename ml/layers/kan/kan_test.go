package kan_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

type kanTestDataset struct {
	batchSize int
	tensor    *tensors.Tensor
}

func (ds *kanTestDataset) Name() string {
	return "kan_dataset"
}

func (ds *kanTestDataset) Reset() {}

func (ds *kanTestDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.tensor == nil {
		ds.tensor = tensors.FromShape(shapes.Make(dtypes.Float64, ds.batchSize, 2))
	}
	spec = ds
	inputs = []*tensors.Tensor{ds.tensor}
	return
}

// targetF is the function we are trying to model.
func targetF(x0, x1 *Node) *Node {
	return Exp(Add(
		Sin(MulScalar(x0, math.Pi)),
		x1))
}

// kanGraphModel will try to model targetF with the minimum number of nodes.
func kanGraphModel(ctx *context.Context, spec any, inputs []*Node) []*Node {
	dtype := dtypes.Float64
	_ = spec
	batchSize := inputs[0].Shape().Dimensions[0]
	g := inputs[0].Graph()
	g.SetTraced(true)

	x0 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	x1 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	labels := targetF(x0, x1)
	output := kan.New(ctx, Concatenate([]*Node{x0, x1}, -1), 1).NumHiddenLayers(1, 2).BSpline(30).Done()
	return []*Node{output, labels}
}

func lossGraphFn(labels []*Node, predictions []*Node) (loss *Node) {
	labels = predictions[1:]
	predictions = predictions[:1]
	loss = losses.MeanSquaredError(labels, predictions)
	return
}

func TestKan(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.NewContext()
	ctx.RngStateFromSeed(42)
	ds := &kanTestDataset{batchSize: 128}

	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, ctx, kanGraphModel,
		lossGraphFn, // a simple wrapper around losses.MeanSquaredError,
		opt,
		nil, // trainMetrics
		nil) // evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(ds, 5000)
	loss := metrics[1].Value().(float64)
	assert.Truef(t, loss < 0.04, "Expected a loss < 0.04, got %g instead", loss)
	require.NoErrorf(t, err, "Failed training: %+v", err)
	fmt.Printf("Metrics:\n")
	for ii, m := range metrics {
		fmt.Printf("\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
	}
}
