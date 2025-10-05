package graph_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
)

func TestFFT(t *testing.T) {
	graphtest.RunTestGraphFn(t, "FFT and InverseFFT", func(g *Graph) (inputs, outputs []*Node) {
		const numPoints = 1000
		const numPeriods = 10
		x := Iota(g, shapes.Make(dtypes.Float32, numPoints), -1)
		x = MulScalar(x, 2*math.Pi*numPeriods/numPoints)
		y := Sin(x)
		y.AssertDims(numPoints)
		inputs = []*Node{y}
		yC := ConvertDType(y, dtypes.Complex128)
		yC.AssertDims(numPoints)
		fft := FFT(yC)
		yHat := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(yC, yHat)))
		outputs = []*Node{diff}
		return
	}, []any{
		0.0,
	}, 1.0)

	graphtest.RunTestGraphFn(t, "RealFFT and InverseRealFFT", func(g *Graph) (inputs, outputs []*Node) {
		const batchDim = 2
		const numPoints = 1000
		x := Iota(g, shapes.Make(dtypes.Float32, batchDim, numPoints), -1)
		const numPeriods = 10
		y := Sin(MulScalar(x, 2*math.Pi*numPeriods/numPoints))
		y.AssertDims(batchDim, numPoints)
		inputs = []*Node{y}
		fft := RealFFT(y)
		fft.AssertDims(batchDim, numPoints/2+1)
		yHat := InverseRealFFT(fft)
		yHat.AssertDims(batchDim, numPoints)
		diff := ReduceAllSum(Abs(Sub(y, yHat)))
		outputs = []*Node{diff}
		return
	}, []any{
		float32(0.0),
	}, 1.0)
}

func TestGradientFFT(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("TestGradientFFT does not work in Darwin -- numerical differences.")
		return
	}
	// Create a small sine curve using 11 numbers (scaled from 0 to 2π),
	// calculate the FFT, takes the gradient of that w.r.t input.
	// Also checks that the InverseFFT takes it back to the input.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(dtypes.Float64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertDType(x, dtypes.Complex128)
		fft := FFT(x)
		fft.SetLogged("FFT")
		output := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(inv, x)))
		grad := Gradient(output, x)[0]
		return []*Node{x}, []*Node{output, argmax, diff, grad}
	}, []any{
		11.0,
		int32(1), // Max frequency is of 1, since it's exactly 1 period of a sin curve.
		0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{
			2.241402232696969 + 0i,
			0.8408081271818637 + 0i,
			4.848838858213708 + 0i,
			2.7934819834338587 + 0i,
			-0.5628044369683731 + 0i,
			0.6846870446443976 + 0i,
			4.140658304799644 + 0i,
			-0.16628391231716422 + 0i,
			0.3723748150049282 + 0i,
			-7.438217966151578 + 0i,
			3.245054949461747 + 0i,
		},
	}, 1e-3)

	// Similar to previous test, same input, but now we check the gradient of the diff of the reversed sequence
	// w.r.t the FFT values.
	// It should be close to 0, since the diff should be close to 0 -- it would be if the FFT were perfect.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(dtypes.Float64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertDType(x, dtypes.Complex128)
		fft := FFT(x)
		sumFft := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := Abs(ReduceAllSum(Sub(inv, x)))
		grad := Gradient(diff, fft)[0]
		return []*Node{x}, []*Node{sumFft, argmax, diff, grad}
	}, []any{
		11.0, int32(1), 0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{(-1 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i), (0 + 0i)},
	}, 1e-3)
}

// realFftExample returns (x, y) where: x is a sinusoidal curve with numPoints points,
// and with `frequency` full cycles; y is the RealFFT(x).
func realFftExample(backend backends.Backend, realDType dtypes.DType, numPoints int, frequency float64) (x, y *tensors.Tensor) {
	e := MustNewExec(backend, func(g *Graph) (x, y *Node) {
		x = Iota(g, shapes.Make(realDType, 1, numPoints), 1)
		x = MulScalar(x, 2.0*math.Pi*frequency/float64(numPoints))
		x = Sin(x)
		y = RealFFT(x)
		return
	})
	outputs := e.Call()
	x, y = outputs[0], outputs[1]
	return
}

/*
TODO: Renable once context package is fixed.

// TestGradientRealFFT tests it by checking that by gradient-descent we can
// invert RealFFT.
//
// See plots of this in `examples/fft/fft.ipynb`.
func TestGradientRealFFT(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	// trueX is real, and trueY is the fft, a complex tensor.
	trueX, trueY := realFftExample(backend, dtypes.Float32, 100, 2)
	ctx := context.NewContext(backend)
	ctx.SetParam(optimizers.ParamLearningRate, 0.01)
	ctx.RngStateFromSeed(42) // Make it deterministic.
	ctx = ctx.WithInitializer(initializers.Zero)
	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()
		learnedXVar := ctx.VariableWithShape("learnedX", trueX.Shape())
		y := RealFFT(learnedXVar.ValueGraph(g))
		return []*Node{y}
	}

	dataset, err := data.InMemoryFromData(backend, "dataset", []any{trueX}, []any{trueY})
	require.NoError(t, err)
	dataset.BatchSize(1, false).Infinite(true)
	trainer := train.NewTrainer(
		backend, ctx, modelFn,
		losses.MeanAbsoluteError,
		optimizers.Adam().Done(),
		nil, nil) // trainMetrics, evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)         // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(dataset, 800) // Typically we get a loss of ~0.01
	require.NoError(t, err)
	require.Greater(t, len(metrics), 0)
	loss := metrics[0].Value().(float32)
	fmt.Println(loss)
	assert.Lessf(t, loss, float32(0.1),
		"Optimizing using gradient descent on RealFFT should have approached an inverse to "+
			"an mean absolute error < 0.1, got %f instead", loss)
}


// TestGradientInverseRealFFT tests it by checking that by gradient-descent we can
// invert InverseRealFFT (so effectively we do a RealFFT).
//
// This works similar to TestGradientRealFFT, but inverts what we are predicting:
// we are trying to learn the FFT value that generates the sinusoidal curve.
func TestGradientInverseRealFFT(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	// We revert the x/y of realFftExample: trueX is the fft, a complex tensor, and trueY is the real sinusoidal curve.
	trueY, trueX := realFftExample(backend, dtypes.Float64, 10, 2)
	ctx := context.NewContext(backend)
	ctx.SetParam(optimizers.ParamLearningRate, 10.0)
	ctx.RngStateFromSeed(42) // Make it deterministic.
	ctx = ctx.WithInitializer(initializers.Zero)
	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()
		learnedXVar := ctx.VariableWithShape("learnedX", trueX.Shape())
		y := InverseRealFFT(learnedXVar.ValueGraph(g))
		return []*Node{y}
	}

	dataset, err := data.InMemoryFromData(backend, "dataset", []any{trueX}, []any{trueY})
	require.NoError(t, err)
	dataset.BatchSize(1, false).Infinite(true)
	trainer := train.NewTrainer(
		backend, ctx, modelFn,
		losses.MeanAbsoluteError,
		optimizers.StochasticGradientDescent(),
		nil, nil) // trainMetrics, evalMetrics
	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)         // Attaches a progress bar to the loop.
	metrics, err := loop.RunSteps(dataset, 100) // Typically we get a loss of ~0.01
	require.NoError(t, err)
	loss := metrics[0].Value().(float64)
	fmt.Println("\tLoss:", loss)
	assert.Lessf(t, loss, 0.1,
		"Optimizing using gradient descent on InverseRealFFT should have approached the original curve to "+
			"an mean absolute error < 0.1, got %f instead", loss)
}
*/
