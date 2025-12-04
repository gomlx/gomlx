package vnn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// TestLinearLayer checks that the linear layer of the VNN is equivariant to rotation.
func TestLinearLayer(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	y0 := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 1, 1, 1, 10, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// Linear function: fix seed so we always have the same values.
		ctx = ctx.Checked(false)
		linearFn := func(x *Node) *Node {
			return New(ctx, x, 2).
				NumHiddenLayers(0, 0).
				Activation("").Regularizer(nil).Done()
		}

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		out1 := RotateOnOrigin(linearFn(input), roll, pitch, yaw)
		require.NoError(t, out1.Shape().CheckDims(1, 1, 1, 2, 3))
		out2 := linearFn(RotateOnOrigin(input, roll, pitch, yaw))
		require.NoError(t, out2.Shape().CheckDims(1, 1, 1, 2, 3))
		diff := Abs(Sub(out1, out2))
		diff.SetLogged("Difference of rotation before/after linear transformation")
		return ReduceAllMean(diff)
	})
	fmt.Printf("\tMean absolute difference: %s\n", y0.GoStr())
	require.Less(t, tensors.ToScalar[float64](y0), 1e-3)
}

// TestRelu checks that the Relu activation with a learned projection is equivariant to rotation.
func TestRelu(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	baseCtx := context.New()
	baseCtx.SetRNGStateFromSeed(42)
	testShape := shapes.Make(dtypes.Float64, 20, 2, 3)
	//testShape := shapes.Make(dtypes.Float64, 1, 2, 3)
	for _, negativeSlope := range []float64{0, 0.2} {
		for _, shareNonLinearity := range []bool{false, true} {
			name := fmt.Sprintf("Leak=%.1f-Shared=%v", negativeSlope, shareNonLinearity)
			t.Run(name, func(t *testing.T) {
				ctx := baseCtx.In(name)
				outputs := context.MustExecOnceN(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
					pi2 := math.Pi * 2.0

					// Random inputs and rotations:
					// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
					input := ctx.RandomUniform(g, testShape)
					roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
					pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
					yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

					// Linear function: fix seed so we always have the same values.
					ctx = ctx.Checked(false)

					// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
					out1 := Relu(ctx, input).
						NegativeSlope(negativeSlope).
						ShareNonLinearity(shareNonLinearity).
						Done()
					diffRelu := ReduceAllMax(Abs(Sub(input, out1)))
					out1 = RotateOnOrigin(out1, roll, pitch, yaw)
					require.True(t, out1.Shape().Equal(testShape))

					out2 := Relu(ctx, RotateOnOrigin(input, roll, pitch, yaw)).
						NegativeSlope(negativeSlope).
						ShareNonLinearity(shareNonLinearity).
						Done()
					require.True(t, out2.Shape().Equal(testShape))

					diff := Abs(Sub(out1, out2))
					return []*Node{diffRelu, ReduceAllMean(diff)}
				})
				reluDiff, rotDiff := outputs[0], outputs[1]
				fmt.Printf("\tBefore/after relu abs difference: %s\n", reluDiff.GoStr())
				fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
				require.Greater(t, tensors.ToScalar[float64](reluDiff), 1e-3)
				require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
			})
		}
	}
}

// TestLayerNormalization checks that the LayerNormalization normalizes properly -- mean close to
// the origin -- and that it is equivariant to rotation.
func TestLayerNormalization(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	outputs := context.MustExecOnceN(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 1, 1_000, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// Linear function: fix seed so we always have the same values.
		ctx = ctx.Checked(false)

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		epsilon := 1e-5
		out1 := RotateOnOrigin(LayerNormalization(input, epsilon), roll, pitch, yaw)
		out1Mean := ReduceMean(out1, 1)
		meanAbsDiff := ReduceAllSum(Abs(out1Mean))
		require.True(t, out1.Shape().Equal(input.Shape()))
		out2 := LayerNormalization(RotateOnOrigin(input, roll, pitch, yaw), epsilon)
		require.True(t, out2.Shape().Equal(input.Shape()))
		diff := Abs(Sub(out1, out2))
		return []*Node{meanAbsDiff, ReduceAllMean(diff)}
	})
	meanAbsDiff, rotDiff := outputs[0], outputs[1]
	fmt.Printf("\tMean diff to origin: %s\n", meanAbsDiff.GoStr())
	fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
	require.Less(t, tensors.ToScalar[float64](meanAbsDiff), 1e-3)
	require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
}

// TestVNN_Equivariant checks that a fully configured VNN is SO(3) equivariant for rotations.
func TestVNN_Equivariant(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)
	rotDiff := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 2, 3, 20, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// vnn layer: fix seed so we always have the same values.
		ctx = ctx.Checked(false)
		vnnFn := func(x *Node) *Node {
			return New(ctx, x, 5).
				NumHiddenLayers(3, 10).
				Activation("relu").
				Normalization("layer").
				Regularizer(nil).
				Done()
		}

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		out1 := RotateOnOrigin(vnnFn(input), roll, pitch, yaw)
		require.NoError(t, out1.Shape().CheckDims(2, 3, 5, 3))
		out2 := vnnFn(RotateOnOrigin(input, roll, pitch, yaw))
		require.NoError(t, out2.Shape().CheckDims(2, 3, 5, 3))
		diff := Abs(Sub(out1, out2))
		return ReduceAllMean(diff)
	})
	fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
	require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
}

// TestVNNTrain checks whether a 2-layer VNN can learn whether 2 3D vectors are pointing to opposite quadrants.
// The function to learn is not rotation-invariant, so we expect this test to fail.
func TestVNNTrain(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.SetRNGStateFromSeed(42)

	// Model function
	numFeatures := 4
	modelFn := func(ctx *context.Context, spec any, inputs []*Node) []*Node {
		x := inputs[0] // Shape: [batch, 2, 3]
		ctx = ctx.In("vnn")
		vnn := New(ctx, x, numFeatures).
			NumHiddenLayers(1, numFeatures). // 2 hidden layers
			Activation("relu").
			Normalization("layer").
			Scaler(true).
			Regularizer(regularizers.L2(0.001)).
			Dropout(0).
			Done()

		// Invariant head for classification
		ctx = ctx.In("head")
		p0 := New(ctx.In("p0"), vnn, 1).
			NumHiddenLayers(0, 0).
			Scaler(true).
			Done()
		p1 := New(ctx.In("p1"), vnn, 1).
			NumHiddenLayers(0, 0).
			Scaler(true).
			Done()
		logit := InvariantDotProduct(p0, p1)
		logit = Reshape(logit, -1) // Shape: [batch]
		return []*Node{logit}
	}

	// Create Dataset
	const batchSize = 64
	const numSamples = 65536
	const numInputs = 2
	const vecDim = 3
	const numSteps = 8_000

	// Generate random operand vectors in the range [-1, 1]
	inputsData := make([]float32, numSamples*numInputs*vecDim)
	rng := rand.New(rand.NewPCG(0, 42))
	for i := range inputsData {
		inputsData[i] = rng.Float32()*2 - 1
	}

	// Generate labels based on whether the vectors have cos(angle)<0:
	labelsData := make([]float32, numSamples)
	for sampleIdx := range numSamples {
		v0Idx := sampleIdx * numInputs * vecDim
		v1Idx := v0Idx + vecDim
		v0x, v0y, v0z := inputsData[v0Idx], inputsData[v0Idx+1], inputsData[v0Idx+2]
		v1x, v1y, v1z := inputsData[v1Idx], inputsData[v1Idx+1], inputsData[v1Idx+2]
		if v0x*v1x+v0y*v1y+v0z*v1z < 0 {
			labelsData[sampleIdx] = 1.0
		} else {
			labelsData[sampleIdx] = 0.0
		}
	}

	// Create a dataset from the generated data.
	inputsTensor := tensors.FromFlatDataAndDimensions(inputsData, numSamples, numInputs, vecDim)
	labelsTensor := tensors.FromFlatDataAndDimensions(labelsData, numSamples)
	ds, err := datasets.InMemoryFromData(backend, "VNN: negative cosine distance",
		[]any{inputsTensor}, []any{labelsTensor})
	require.NoError(t, err)
	dsEval := ds.Copy().BatchSize(1, false)
	//ds.Shuffle().BatchSize(batchSize, true).Infinite(true)
	ds.BatchSize(batchSize, true).Infinite(true)

	trainer := train.NewTrainer(
		backend, ctx, modelFn, losses.BinaryCrossentropyLogits,
		optimizers.Adam().LearningRate(3e-5).Done(),
		[]metrics.Interface{metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Accuracy", "~acc", 0.01)},
		[]metrics.Interface{metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")})

	loop := train.NewLoop(trainer)
	//commandline.AttachProgressBar(loop)
	_, err = loop.RunSteps(ds, numSteps)
	require.NoError(t, err)
	lossAndMetrics, err := trainer.Eval(dsEval)
	require.NoError(t, err)
	for metricIdx, metricSpec := range trainer.EvalMetrics() {
		fmt.Printf("\t%q=%s\n", metricSpec.ShortName(), lossAndMetrics[metricIdx])
	}

	// We expect the VNN to fail to learn this function because it's not rotation-invariant.
	// Accuracy should be around 50% (random guessing).
	accuracy := lossAndMetrics[2].Value().(float32)
	require.GreaterOrEqual(t, accuracy, float32(0.8), "VNN was not able to learn rotation invariant simple task, accuracy=%.1f%%.", accuracy*100.0)

	sample := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		return ctx.RandomUniform(g, shapes.Make(dtypes.Float64))
	})
	fmt.Printf("Context random sample: %s\n", sample.GoStr())
}
