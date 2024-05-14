package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
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
		seedsIds := tensor.FromValue([]int32{seedId}) // We take only one seed for testing.
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
			values := state.Value.Local().FlatCopy().([]int32)
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
	ctx.RngStateReset()

	// Standard MAG parameters.
	ctx.SetParams(map[string]any{
		"checkpoint":      "",
		"num_checkpoints": 3,
		"train_steps":     0,
		"plots":           true,

		optimizers.ParamOptimizer:           "adam",
		optimizers.ParamLearningRate:        0.001,
		optimizers.ParamCosineScheduleSteps: 0,
		optimizers.ParamClipStepByValue:     0.0,
		optimizers.ParamAdamEpsilon:         1e-7,
		optimizers.ParamAdamDType:           "",

		layers.ParamL2Regularization: 1e-5,
		layers.ParamDropoutRate:      0.2,
		layers.ParamActivation:       "swish",

		gnn.ParamEdgeDropoutRate:       0.0,
		gnn.ParamNumGraphUpdates:       6, // gnn_num_messages
		gnn.ParamReadoutHiddenLayers:   2,
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

// TestLayerWiseInference uses [flagDataDir] to store downloaded data, which defaults to `~/work/ogbnmag` by
// default.
func TestLayerWiseInference(t *testing.T) {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInference: it requires downloading OGBN-MAG data.")
		return
	}
	fmt.Printf("Creating dataset.\n")
	const batchSize = 1
	// Paper id with the least amount of degrees in its subgraph.
	// seedId := findSmallestDegreeSubgraph(t)
	const seedId = 3162
	seedsIds := tensor.FromValue([]int32{seedId}) // We take only one seed for testing.

	// Create inputs for `seedId`.
	require.NoError(t, Download(*flagDataDir), "Download")
	magSampler, err := NewSampler(*flagDataDir)
	strategy := NewSamplerStrategy(magSampler, batchSize, seedsIds)
	ds := strategy.NewDataset("lwinference_test")
	_, inputs, _, err := ds.Yield()
	require.NoError(t, err, "Dataset.Yield")

	manager := graphtest.BuildTestManager()
	for ctxSourceIdx := 1; ctxSourceIdx < 2; ctxSourceIdx++ {
		// Create context.
		ctx := context.NewContext(manager)
		UploadOgbnMagVariables(ctx)
		if ctxSourceIdx == 0 {
			fmt.Printf("\nRandomly initialized context:\n")
			configureLayerWiseTestContext(ctx)

		} else {
			_, fileName, _, ok := runtime.Caller(0)
			require.True(t, ok, "Failed to get caller information to find out test source directory.")
			baseDir := filepath.Dir(fileName)
			checkpoint, err := checkpoints.Build(ctx).DirFromBase("test_checkpoint", baseDir).
				Immediate().Done()
			fmt.Printf("\nLoaded trained context: %s\n", checkpoint.Dir())
			require.NoError(t, err, "Checkpoint loading.")
			ctx = ctx.Reuse()
		}

		// Execute normal inference model for the inputs.
		executor := context.NewExec(manager, ctx, func(ctx *context.Context, inputs []*Node) *Node {
			predictionsAndMask := MagModelGraph(ctx, strategy, inputs)
			return ConvertType(predictionsAndMask[0], shapes.Float32)
		})
		var results []tensor.Tensor
		require.NotPanics(t, func() { results = executor.Call(inputs) })
		predictionsGNN := results[0]
		fmt.Printf("predictionsGNN:\n%s\n", predictionsGNN)

		// Layer-Wise inference
		modelFn := BuildLayerWiseInferenceModel(strategy, false) // Function that builds the LW inference model.
		executor = context.NewExec(ctx.Manager(), ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
			allPredictions := modelFn(ctx, g)
			return ConvertType(
				Slice(allPredictions, AxisElem(seedId), AxisRange()),
				shapes.Float32)
		})
		require.NotPanics(t, func() { results = executor.Call() })
		predictionsLW := results[0]
		fmt.Printf("\npredictionsLW:\n%s\n", predictionsLW)

		require.True(t,
			slices.DeepSliceCmp(
				predictionsGNN.Local().Value().([][]float32),
				predictionsLW.Local().Value().([][]float32),
				slices.Close[float32]))
	}
}
