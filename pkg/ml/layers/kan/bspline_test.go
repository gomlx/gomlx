// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package kan_test

import (
	"fmt"
	"math"
	"path"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/gomlx/gomlx/ui/commandline"
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
func kanGraphModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	dtype := dtypes.Float64
	_ = spec
	batchSize := inputs[0].Shape().Dimensions[0]
	g := inputs[0].Graph()
	g.SetTraced(true)

	normalizeUniformFn := func(x *Node) *Node {
		return AddScalar(MulScalar(x, 2), -1)
	}
	x0 := normalizeUniformFn(scope.RandomUniform(g, shapes.Make(dtype, batchSize, 1)))
	x1 := normalizeUniformFn(scope.RandomUniform(g, shapes.Make(dtype, batchSize, 1)))
	labels := targetF(x0, x1)
	output := kan.New(scope, Concatenate([]*Node{x0, x1}, -1), 1).
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
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	store.RootScope().Store().ResetRNGState()
	store.SetParam(model.ParamInitialSeed, int64(42))
	ds := &kanTestDataset{batchSize: 128}

	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, store.RootScope(), kanGraphModel,
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
func kanLargeGraphModel(scope *model.Scope, spec any, inputs []*Node) []*Node {
	dtype := dtypes.Float64
	_ = spec
	batchSize := inputs[0].Shape().Dimensions[0]
	g := inputs[0].Graph()
	g.SetTraced(true)

	x0 := scope.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	x1 := scope.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
	labels := targetF(x0, x1)
	output := kan.New(scope, Concatenate([]*Node{x0, x1}, -1), 1).
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
	backend := testutil.BuildTestBackend()
	store := model.NewStore()
	store.RootScope().Store().ResetRNGState()
	store.SetParam(model.ParamInitialSeed, int64(42))
	store.SetParam(kan.ParamBSplineMagnitudeL1, 0.01)
	ds := &kanTestDataset{batchSize: 128}

	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, store.RootScope(), kanLargeGraphModel,
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
	for _, scopeName := range []string{"/bspline_kan_hidden_0", "/bspline_kan_output_layer"} {
		for _, vName := range []string{"w_splines", "w_residual"} {
			v := store.GetVariable(path.Join(scopeName, vName))
			require.NotNilf(t, v, "failed to inspect variable scope=%q, name=%q", scopeName, vName)
			tensor := v.MustValue()
			fmt.Printf("\t%s : %s -> %v\n", v.Scope(), v.Name(), tensor)
			tensors.MustConstFlatData(tensor, func(flat []float64) {
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
