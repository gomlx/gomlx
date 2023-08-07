package graph_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestFFT(t *testing.T) {
	graphtest.RunTestGraphFn(t, "FFT and InverseFFT", func(g *Graph) (inputs, outputs []*Node) {
		const numPoints = 1000
		const numPeriods = 10
		x := Iota(g, shapes.Make(shapes.F32, numPoints), -1)
		x = MulScalar(x, 2*math.Pi*numPeriods/numPoints)
		y := Sin(x)
		y.AssertDims(numPoints)
		inputs = []*Node{y}
		yC := ConvertType(y, shapes.Complex128)
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
		x := Iota(g, shapes.Make(shapes.F32, batchDim, numPoints), -1)
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
	// Create a small sine curve using 11 numbers (scaled from 0 to 2π),
	// calculate the FFT, takes the gradient of that w.r.t input.
	// Also checks that the InverseFFT takes it back to the input.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(shapes.F64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertType(x, shapes.Complex128)
		fft := FFT(x)
		output := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(inv, x)))
		grad := Gradient(output, x)[0]
		return []*Node{x}, []*Node{output, argmax, diff, grad}
	}, []any{
		11.0, int32(10), 0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{
			-0.7126987103881084 + 2.050415210607949i,
			-0.34708456852657743 + 2.1129871487498937i,
			6.221747438979876 - 0.7815639565770847i,
			1.2212951935321779 - 0.5731899738198121i,
			2.737096425081983 + 0.5834007264215436i,
			2.8156998840012886 - 2.042120467795749i,
			0.7909217660400658 + 2.4657154202830442i,
			-4.848589784434233 + 2.175014774879141i,
			2.2094212306941765 - 0.2007046968306172i,
			-1.3018177490725729 - 2.3496106014958604i,
			1.815039114738962 - 0.5046736716278869i},
	}, 1e-3)

	// Similar to previous test, same input, but now we check the gradient of the diff of the reversed sequence
	// w.r.t the FFT values.
	// It should be close to 0, since the diff should be close to 0 -- it would be if the FFT were perfect.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(shapes.F64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertType(x, shapes.Complex128)
		fft := FFT(x)
		sumFft := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(inv, x)))
		grad := Gradient(diff, inv)[0]
		return []*Node{x}, []*Node{sumFft, argmax, diff, grad}
	}, []any{
		11.0, int32(10), 0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{
			0.21951219512195125 + 0.9756097560975611i,
			-0.043097175436008814 + 0.9990708851074771i,
			0.8087360843031886 + 0.5881716976750463i,
			0.11383845836148958 + 0.9934992729729999i,
			0.15968884391337387 + 0.9871673987372204i,
			0.656144370283719 - 0.7546353856962857i,
			-0.9991847547814338 + 0.04037110120329026i,
			-0.9549934880178522 - 0.29662676521766623i,
			-0.6288692494311636 - 0.7775110720239841i,
			0.13621834047872816 - 0.9906788398452958i,
			0.9680364763016536 + 0.2508094506781557i},
	}, 1e-3)
}

// realFftExample returns (x, y) where: x is a sinusoidal curve with numPoints points,
// and with `frequency` full cycles; y is the RealFFT(x).
func realFftExample(manager *Manager, realDType shapes.DType, numPoints int, frequency float64) (x, y tensor.Tensor) {
	e := NewExec(manager, func(g *Graph) (x, y *Node) {
		x = Iota(g, shapes.Make(realDType, 1, numPoints), 1)
		x = MulScalar(x, 2.0*math.Pi*frequency/float64(numPoints))
		x = Sin(x)
		y = RealFFT(x)
		return
	})
	res := e.Call()
	x, y = res[0], res[1]
	return
}

// TestGradientRealFFT tests it by checking that by gradient-descent we can
// invert RealFFT.
//
// See plots of this in `examples/fft/fft.ipynb`.
func TestGradientRealFFT(t *testing.T) {
	manager := graphtest.BuildTestManager()
	// trueX is real, and trueY is the fft, a complex tensor.
	trueX, trueY := realFftExample(manager, shapes.F32, 100, 2)
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, 0.01)
	ctx.RngStateFromSeed(42) // Make it deterministic.
	ctx = ctx.WithInitializer(initializers.Zero)
	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()
		learnedXVar := ctx.VariableWithShape("learnedX", trueX.Shape())
		y := RealFFT(learnedXVar.ValueGraph(g))
		return []*Node{y}
	}

	dataset, err := data.InMemoryFromData(manager, "dataset", []any{trueX}, []any{trueY})
	require.NoError(t, err)
	dataset.BatchSize(1, false).Infinite(true)
	trainer := train.NewTrainer(
		manager, ctx, modelFn,
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
	manager := graphtest.BuildTestManager()
	// We revert the x/y of realFftExample: trueX is the fft, a complex tensor, and trueY is the real sinusoidal curve.
	trueY, trueX := realFftExample(manager, shapes.F64, 10, 2)
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, 10.0)
	ctx.RngStateFromSeed(42) // Make it deterministic.
	ctx = ctx.WithInitializer(initializers.Zero)
	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		g := inputs[0].Graph()
		learnedXVar := ctx.VariableWithShape("learnedX", trueX.Shape())
		y := InverseRealFFT(learnedXVar.ValueGraph(g))
		return []*Node{y}
	}

	dataset, err := data.InMemoryFromData(manager, "dataset", []any{trueX}, []any{trueY})
	require.NoError(t, err)
	dataset.BatchSize(1, false).Infinite(true)
	trainer := train.NewTrainer(
		manager, ctx, modelFn,
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
