package kan_test

import (
	"fmt"
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

type kanTestDataset struct {
	batchSize int
}

func (ds *kanTestDataset) Name() string {
	return "kan_dataset"
}

func (ds *kanTestDataset) Reset() {}

func (ds *kanTestDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	spec = ds
	inputs = []*tensors.Tensor{tensors.FromShape(shapes.Make(dtypes.Float64, ds.batchSize, 2))}
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

	normalizeUniformFn := func(x *Node) *Node {
		return AddScalar(MulScalar(x, 2), -1)
	}
	x0 := normalizeUniformFn(ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1)))
	x1 := normalizeUniformFn(ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1)))
	labels := targetF(x0, x1)
	output := kan.New(ctx, Concatenate([]*Node{x0, x1}, -1), 1).
		NumHiddenLayers(1, 2).
		NumControlPoints(30).
		BSpline().
		Done()
	return []*Node{output, labels}
}

func lossGraphFn(labels []*Node, predictions []*Node) (loss *Node) {
	labels = predictions[1:]
	predictions = predictions[:1]
	loss = losses.MeanSquaredError(labels, predictions)
	return
}

func TestBSplineKAN(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	ds := &kanTestDataset{batchSize: 128}

	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, ctx, kanGraphModel,
		lossGraphFn, // a simple wrapper around losses.MeanSquaredError,
		opt,
		nil, // trainMetrics
		nil) // evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(ds, 10_000)
	require.NoErrorf(t, err, "Failed building the model / training")
	loss := metrics[1].Value().(float64)
	assert.Truef(t, loss < 1.0, "Expected a loss < 1.0, got %g instead", loss)
	fmt.Printf("Metrics:\n")
	for ii, m := range metrics {
		fmt.Printf("\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
	}
}

// kanLargeGraphModel will try to model targetF with extra unnecessary number of nodes.
func kanLargeGraphModel(ctx *context.Context, spec any, inputs []*Node) []*Node {
	dtype := dtypes.Float64
	_ = spec
	batchSize := inputs[0].Shape().Dimensions[0]
	g := inputs[0].Graph()
	g.SetTraced(true)

	x0 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	x1 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	labels := targetF(x0, x1)
	output := kan.New(ctx, Concatenate([]*Node{x0, x1}, -1), 1).
		BSpline().
		NumHiddenLayers(1, 4).
		NumControlPoints(30).
		Done()
	return []*Node{output, labels}
}

func TestBSplineKANRegularized(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	ctx.SetParam(kan.ParamBSplineMagnitudeL1, 0.01)
	ds := &kanTestDataset{batchSize: 128}

	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, ctx, kanLargeGraphModel,
		lossGraphFn, // a simple wrapper around losses.MeanSquaredError,
		opt,
		nil, // trainMetrics
		nil) // evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(ds, 20_000)
	require.NoErrorf(t, err, "Failed building the model / training")
	loss := metrics[1].Value().(float64)
	assert.Truef(t, loss < 0.4, "Expected a loss < 0.4, got %g instead", loss)
	fmt.Println("Metrics:")
	for ii, m := range metrics {
		fmt.Printf("\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
	}

	// We are going to count the number of zeros in the magnitude: we know the model kanLargeGraphModel is
	// over-dimensioned, so with L1 regularizer on the magnitudes, many will become 0
	// (if we remove the L1 regularization the test fails).
	var numZeros int
	fmt.Println("\nVariables:")
	//ctx.EnumerateVariables(func(v *context.Variable) {
	//	fmt.Printf("\t%s -> %v\n", v.ScopeAndName(), v.Value())
	//})
	for _, scope := range []string{"/bspline_kan_hidden_0", "/bspline_kan_output_layer"} {
		for _, vName := range []string{"w_splines", "w_residual"} {
			v := ctx.GetVariableByScopeAndName(scope, vName)
			require.NotNilf(t, v, "failed to inspect variable scope=%q, name=%q", scope, vName)
			tensor := v.MustValue()
			fmt.Printf("\t%s : %s -> %v\n", v.Scope(), v.Name(), tensor)
			tensors.MustConstFlatData[float64](tensor, func(flat []float64) {
				for _, element := range flat {
					if element == 0.0 {
						numZeros++
					}
				}
			})
		}
	}
	fmt.Printf("\nNumber of zeros in the magnitudes of the KAN network: %d\n", numZeros)
	// Most of the cases we get 12 zeros.
	require.GreaterOrEqual(
		t,
		numZeros,
		9,
		"We expected at least 9 zeros on the magnitudes of the KAN model, with L1 regularizer, we got only %d though",
		numZeros,
	)
}
