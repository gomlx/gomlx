package fnn

import (
	"fmt"
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

type fnnTestDataset struct {
	batchSize int
}

func (ds *fnnTestDataset) Name() string {
	return "fnn_dataset"
}

func (ds *fnnTestDataset) Reset() {}

func (ds *fnnTestDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
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

type coreFnType func(ctx *context.Context, input *Node) *Node

// fnnGraphModelBuilder will try to model targetF using the given coreFn function.
func fnnGraphModelBuilder(coreFn coreFnType) train.ModelFn {
	return func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		dtype := dtypes.Float64
		_ = spec
		batchSize := inputs[0].Shape().Dimensions[0]
		g := inputs[0].Graph()
		g.SetTraced(true)

		x0 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
		x1 := ctx.RandomUniform(g, shapes.Make(dtype, batchSize, 1))
		labels := targetF(x0, x1)
		input := Concatenate([]*Node{x0, x1}, -1)
		output := coreFn(ctx, input)
		return []*Node{output, labels}
	}
}

func lossGraphFn(labels []*Node, predictions []*Node) (loss *Node) {
	labels = predictions[1:]
	predictions = predictions[:1]
	loss = losses.MeanSquaredError(labels, predictions)
	return
}

var (
	fnnVariations = []coreFnType{
		// Vanilla
		func(ctx *context.Context, input *Node) *Node {
			return New(ctx, input, 1).
				NumHiddenLayers(1, 128).
				Activation(activations.TypeRelu).
				Done()
		},

		// Residual+Normalization:
		func(ctx *context.Context, input *Node) *Node {
			return New(ctx, input, 1).
				NumHiddenLayers(8, 8).
				Normalization("layer").
				Residual(true).
				Activation(activations.TypeSigmoid).
				Done()
		},
	}

	fnnVariationsNames = []string{"Vanilla", "Residual+Normalization"}
	fnnVariationsSteps = []int{6000, 2000}
)

func TestFNN(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}

	backend := graphtest.BuildTestBackend()
	ds := &fnnTestDataset{batchSize: 128}

	for ii, coreFn := range fnnVariations {
		fmt.Printf("Variation #%d %q:\n", ii, fnnVariationsNames[ii])
		ctx := context.New()
		ctx.SetRNGStateFromSeed(42)
		opt := optimizers.Adam().LearningRate(0.001).Done()
		trainer := train.NewTrainer(backend, ctx, fnnGraphModelBuilder(coreFn),
			lossGraphFn, // a simple wrapper around losses.MeanSquaredError,
			opt,
			nil, // trainMetrics
			nil) // evalMetrics
		loop := train.NewLoop(trainer)
		commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
		metrics, err := loop.RunSteps(ds, fnnVariationsSteps[ii])
		require.NoErrorf(t, err, "Failed building the model / training")
		loss := metrics[1].Value().(float64)
		assert.Truef(t, loss < 0.1, "Expected a loss < 0.1, got %g instead", loss)
		fmt.Printf("\tMetrics:\n")
		for ii, m := range metrics {
			fmt.Printf("\t\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
		}
		fmt.Println()
	}
}

func TestFNNRegularized(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	ds := &fnnTestDataset{batchSize: 128}

	regularizedFn := func(ctx *context.Context, input *Node) *Node {
		return New(ctx, input, 1).
			NumHiddenLayers(2, 32).
			Residual(true).
			Normalization("layer").
			Regularizer(regularizers.L1(0.01)).
			Dropout(0).
			Done()
	}
	opt := optimizers.Adam().LearningRate(0.001).Done()
	trainer := train.NewTrainer(backend, ctx, fnnGraphModelBuilder(regularizedFn),
		lossGraphFn, // a simple wrapper around losses.MeanSquaredError,
		opt,
		nil, // trainMetrics
		nil) // evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(ds, 10_000)
	loss := metrics[1].Value().(float64)
	assert.Truef(t, loss < 0.07, "Expected a loss < 0.07, got %g instead", loss)
	require.NoErrorf(t, err, "Failed training: %+v", err)
	fmt.Println("Metrics:")
	for ii, m := range metrics {
		fmt.Printf("\t%s: %s\n", trainer.TrainMetrics()[ii].Name(), m)
	}

	// We are going to count the number of zeros in the weights.
	var numZeros int
	fmt.Println("\nVariables:")
	/*
		ctx.EnumerateVariables(func(v *context.Variable) {
			if strings.Index(v.Scope(), "Adam") != -1 {
				return
			}
			fmt.Printf("\t%s : %s -> %s\n", v.Scope(), v.Name(), v.Shape())
		})
	*/
	for _, scope := range []string{"/fnn_hidden_layer_0", "/fnn_hidden_layer_1", "/fnn_output_layer"} {
		vName := "weights"
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
	fmt.Printf("\nNumber of zeros in the weights of the FNN: %d\n", numZeros)
	require.GreaterOrEqual(
		t,
		numZeros,
		1000,
		"We expected at least 1000 zeros on the weights of the FNN, with L1 regularizer, we got only %d though",
		numZeros,
	)
}
