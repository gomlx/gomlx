package ogbnmag

import (
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/schollz/progressbar/v3"
	"github.com/stretchr/testify/require"
)

func findSmallestDegreeSubgraph(t *testing.T) int32 {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInference: it requires downloading OGBN-MAG data.")
		return 0
	}
	require.NoError(t, Download(*flagDataDir), "Download")
	magSampler, err := NewSampler(*flagDataDir)
	require.NoError(t, err, "NewSampler")
	const batchSize = 1
	minSeedId, minMaxDegree := int32(-1), int32(1000)
	for seedId := range int32(10000) {
		seedsIds := tensors.FromValue([]int32{seedId}) // We take only one seed for testing.
		strategy := NewSamplerStrategy(magSampler, batchSize, seedsIds)
		ds := strategy.NewDataset("lwinference_test")

		_, inputs, _, err := ds.Yield()
		require.NoError(t, err, "Dataset.Yield")
		nameToState, _ := sampler.MapInputsToStates(strategy, inputs)
		maxDegree := int32(-1)
		for name, state := range nameToState {
			if !strings.HasSuffix(name, ".degree") {
				continue
			}
			values := tensors.MustCopyFlatData[int32](state.Value)
			for _, v := range values {
				if v > maxDegree {
					maxDegree = v
				}
			}
		}
		if maxDegree < minMaxDegree {
			minMaxDegree = maxDegree
			minSeedId = seedId
		}
	}
	fmt.Printf("SeedId=%d, max degree is %d\n", minSeedId, minMaxDegree)
	return minSeedId
}

func configureLayerWiseTestContext(ctx *context.Context) {
	// Standard MAG parameters.
	ctx.SetParams(map[string]any{
		"checkpoint":      "",
		"num_checkpoints": 3,
		"train_steps":     0,
		"plots":           true,

		optimizers.ParamOptimizer:       "adam",
		optimizers.ParamLearningRate:    0.001,
		cosineschedule.ParamPeriodSteps: 0,
		optimizers.ParamClipStepByValue: 0.0,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",

		layers.ParamL2Regularization: 1e-5,
		layers.ParamDropoutRate:      0.2,
		activations.ParamActivation:  "swish",

		gnn.ParamEdgeDropoutRate:       0.0,
		gnn.ParamNumGraphUpdates:       6, // gnn_num_messages
		gnn.ParamPoolingType:           "mean|logsum",
		gnn.ParamUpdateStateType:       "residual",
		gnn.ParamUsePathToRootStates:   false,
		gnn.ParamGraphUpdateType:       "simultaneous",
		gnn.ParamUpdateNumHiddenLayers: 0,
		gnn.ParamMessageDim:            32, // 128 or 256 will work better, but takes way more time
		gnn.ParamStateDim:              32, // 128 or 256 will work better, but takes way more time
		gnn.ParamUseRootAsContext:      false,

		ParamEmbedDropoutRate:     0.0,
		ParamSplitEmbedTablesSize: 1,
		ParamReuseKernels:         true,
		ParamIdentitySubSeeds:     true,
		ParamDType:                "float32",
	})

	// Test parameters.
	ctx.SetParam(gnn.ParamNumGraphUpdates, 6)
	ctx.SetParam(ParamDType, "float32")
	ctx.SetParam(optimizers.ParamAdamDType, "float32")
}

// TestLayerWiseInferenceLogits checks that the logits from layer-wise and by sampling are the same, for one paper
// with a small subgraph that fits fully in memory.
//
// It uses [flagDataDir] to store downloaded data, which defaults to `~/work/ogbnmag` by default.
func TestLayerWiseInferenceLogits(t *testing.T) {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInference: it requires downloading OGBN-MAG data.")
		return
	}
	fmt.Printf("Creating dataset.\n")
	const batchSize = 1
	// Paper id with the least amount of degrees in its subgraph.
	// seedId := findSmallestDegreeSubgraph(t)
	const seedId = 3162
	seedsIds := tensors.FromValue([]int32{seedId}) // We take only one seed for testing.

	// Create inputs for `seedId`.
	require.NoError(t, Download(*flagDataDir), "Download")
	magSampler, err := NewSampler(*flagDataDir)
	require.NoError(t, err, "NewSampler")
	strategy := NewSamplerStrategy(magSampler, batchSize, seedsIds)
	ds := strategy.NewDataset("lwinference_test")
	_, inputs, _, err := ds.Yield()
	require.NoError(t, err, "Dataset.Yield")

	backend := graphtest.BuildTestBackend()
	for ctxSourceIdx := 0; ctxSourceIdx < 2; ctxSourceIdx++ {
		// Create context.
		ctx := context.New()
		if ctxSourceIdx == 0 {
			fmt.Printf("\nRandomly initialized context:\n")
			configureLayerWiseTestContext(ctx)
			UploadOgbnMagVariables(backend, ctx)

		} else {
			// Load from pre-trained checkpoint.
			_, fileName, _, ok := runtime.Caller(0)
			require.True(t, ok, "Failed to get caller information to find out test source directory.")
			baseDir := filepath.Dir(fileName)
			checkpoint, err := checkpoints.Build(ctx).DirFromBase("test_checkpoint", baseDir).Done()
			fmt.Printf("\nLoaded trained context: %s\n", checkpoint.Dir())
			require.NoError(t, err, "Checkpoint loading.")
			UploadOgbnMagVariables(backend, ctx)
			ctx = ctx.Reuse()
		}

		// Execute normal inference model for the inputs.
		executor := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputs []*Node) *Node {
			predictionsAndMask := MagModelGraph(ctx, strategy, inputs)
			return ConvertDType(predictionsAndMask[0], dtypes.Float32)
		})
		var results []*tensors.Tensor
		require.NotPanics(t, func() { results = executor.MustExec(inputs) })
		predictionsGNN := results[0]
		fmt.Printf("predictionsGNN:\n%s\n", predictionsGNN)

		// Layer-Wise inference
		modelFn := BuildLayerWiseInferenceModel(strategy, false) // Function that builds the LW inference model.
		executor = context.MustNewExec(backend, ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
			allPredictions := modelFn(ctx, g)
			return ConvertDType(
				Slice(allPredictions, AxisElem(seedId), AxisRange()),
				dtypes.Float32)
		})
		require.NotPanics(t, func() { results = executor.MustExec() })
		predictionsLW := results[0]
		fmt.Printf("\npredictionsLW:\n%s\n", predictionsLW)
		require.True(t, predictionsGNN.InDelta(predictionsLW, 0.05))
	}
}

// TestLayerWiseInferencePredictions checks that the layer-wise predictions of random initialized model and a trained model
// mostly matches the sampled GNN.
//
// It uses [flagDataDir] to store downloaded data, which defaults to `~/work/ogbnmag` by default.
func TestLayerWiseInferencePredictions(t *testing.T) {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInferencePredictions: it requires downloading OGBN-MAG data.")
		return
	}
	fmt.Printf("Creating dataset.\n")
	const batchSize = 32

	// Create dataset with all papers.
	require.NoError(t, Download(*flagDataDir), "Download")
	magSampler, err := NewSampler(*flagDataDir)
	require.NoError(t, err, "NewSampler")
	strategy := NewSamplerStrategy(magSampler, batchSize, nil /* all papers */)
	var ds train.Dataset
	ds = strategy.NewDataset("lwinference_test")
	ds = mldata.Map(ds, ExtractLabelsFromInput)

	// Create context and load from pre-trained checkpoint.
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	_, fileName, _, ok := runtime.Caller(0)
	require.True(t, ok, "Failed to get caller information to find out test source directory.")
	baseDir := filepath.Dir(fileName)
	checkpoint, err := checkpoints.Build(ctx).DirFromBase("test_checkpoint", baseDir).Done()
	require.NoError(t, err, "Checkpoint loading.")
	fmt.Printf("\nLoaded trained context: %s\n", checkpoint.Dir())
	fmt.Printf("\t%s=%q\n", ParamDType, context.GetParamOr(ctx, ParamDType, ""))
	UploadOgbnMagVariables(backend, ctx)
	ctx = ctx.Reuse()

	// Execute normal inference model for the inputs.
	executor := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputs []*Node) []*Node {
		labels := inputs[len(inputs)-1]
		inputs = inputs[:len(inputs)-1]
		predictionsAndMask := MagModelGraph(ctx, strategy, inputs)
		predictions := ArgMax(predictionsAndMask[0], -1, dtypes.Int32)
		mask := predictionsAndMask[1]
		correct := ConvertDType(Equal(predictions, Squeeze(labels, -1)), dtypes.Int32)
		correct = Where(mask, correct, ZerosLike(correct))
		correct = ReduceAllSum(correct)
		count := ReduceAllSum(ConvertDType(mask, dtypes.Int32))
		return []*Node{correct, count, predictions}
	})
	var correct, total int
	numSteps := int64(NumPapers-batchSize+1) / batchSize // Each step has batchSize samples.
	numSteps = 64
	predictionsGNN := make([]int32, 0, numSteps*batchSize)
	pBar := progressbar.Default(numSteps, "steps")
	ds.Reset()
	count := 0
	for {
		_, inputs, labels, err := ds.Yield()
		if err == io.EOF || count == int(numSteps) {
			require.NoError(t, pBar.Finish())
			require.NoError(t, pBar.Close())
			break
		}
		require.NoError(t, err, "Dataset.Yield")
		inputs = append(inputs, labels[0])
		var results []*tensors.Tensor
		require.NotPanics(t, func() { results = executor.MustExec(inputs) })
		correct += int(results[0].Value().(int32))
		total += int(results[1].Value().(int32))
		predictionsGNN = append(predictionsGNN, results[2].Value().([]int32)...)
		_ = pBar.Add(1)
		count++
	}
	fmt.Printf("predictionsGNN: %d correct out of %d, %.2f%% accuracy\n%v ...\n",
		correct, total, 100.0*float64(correct)/float64(total), predictionsGNN[:100])
	executor.Finalize()

	// Layer-Wise inference
	modelFn := BuildLayerWiseInferenceModel(strategy, false) // Function that builds the LW inference model.
	numToCompare := len(predictionsGNN)
	executor = context.MustNewExec(backend, ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
		logits := modelFn(ctx, g)
		predictions := ArgMax(logits, -1, dtypes.Int32)
		predictions = Slice(predictions, AxisRange(0, numToCompare))
		return predictions
	})
	var results []*tensors.Tensor
	require.NotPanics(t, func() { results = executor.MustExec() })
	predictionsLW := results[0].Value().([]int32)
	correct = 0
	labels := tensors.MustCopyFlatData[int32](PapersLabels)
	for ii, value := range predictionsLW {
		if value == labels[ii] {
			correct++
		}
	}
	fmt.Printf("\npredictionsLW: %d correct out of %d, %.2f%% accuracy\n%v...\n",
		correct, len(predictionsLW),
		100.0*float64(correct)/float64(len(predictionsLW)),
		predictionsLW[:100])

	// Compare how many are the same.
	matches := 0
	for ii, gnnValue := range predictionsGNN {
		if gnnValue == predictionsLW[ii] {
			matches++
		}
	}
	matchRatio := float64(matches) / float64(len(predictionsGNN))
	fmt.Printf("\n%d matches out of %d (%.2f%%)\n",
		matches, len(predictionsGNN),
		100.0*matchRatio)
	require.Greater(
		t,
		matchRatio,
		0.60,
		"Expect LayerWise inference to match at least 60% of times with sampled graph GNN inference",
	)
}
