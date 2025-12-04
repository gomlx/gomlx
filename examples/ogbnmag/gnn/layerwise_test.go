package gnn

import (
	"fmt"
	"testing"

	samplerPkg "github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// Test sampler constants.
const (
	lwFactor     = 5
	lwNumPapers  = 10
	lwNumAuthors = lwNumPapers * lwFactor
)

// createDenseTestSampler creates a sampler where every paper has exactly 5 author, so the sampled and layer-wise
// inference should be exactly the same.
//
// Optionally, it includes paper citations along an identity node-set for the seeds.
// Each paper cites the next 5 (in a circular form).
func createDenseTestSampler(withCitation bool) *samplerPkg.Sampler {
	sampler := samplerPkg.New()
	sampler.AddNodeType("papers", lwNumPapers)
	sampler.AddNodeType("authors", lwNumAuthors)

	authorWritesPapers := tensors.FromShape(shapes.Make(dtypes.Int32, lwNumAuthors, 2))
	{
		// Each paper is written by 5 authors.
		tensors.MustMutableFlatData[int32](authorWritesPapers, func(authorData []int32) {
			for authorIdx := range int32(lwNumAuthors) {
				paperIdx := authorIdx / 5
				authorData[authorIdx*2] = authorIdx
				authorData[authorIdx*2+1] = paperIdx
			}
		})
	}
	sampler.AddEdgeType("writes", "authors", "papers", authorWritesPapers, false)
	sampler.AddEdgeType("writtenBy", "authors", "papers", authorWritesPapers, true)

	if withCitation {
		paperCitesPaper := tensors.FromShape(shapes.Make(dtypes.Int32, lwNumPapers*lwFactor, 2))
		{
			// Each paper is written by 5 authors.
			tensors.MustMutableFlatData[int32](paperCitesPaper, func(citesData []int32) {
				for citing := range int32(lwNumPapers) {
					for ii := range int32(lwFactor) {
						cited := (citing + ii + 1) % lwNumPapers
						citesData[(citing*lwFactor+ii)*2] = citing
						citesData[(citing*lwFactor+ii)*2+1] = cited
					}
				}
			})
		}
		sampler.AddEdgeType("cites", "papers", "papers", paperCitesPaper, false)
		sampler.AddEdgeType("citedBy", "papers", "papers", paperCitesPaper, true)
	}
	return sampler
}

func createDenseTestStrategy(withCitation bool) (*samplerPkg.Sampler, *samplerPkg.Strategy) {
	s := createDenseTestSampler(withCitation)
	strategy := s.NewStrategy()
	strategy = s.NewStrategy()
	seeds := strategy.NodesFromSet("seeds", "papers", lwNumPapers, nil)
	_ = seeds.FromEdges(
		"authors",
		"writtenBy",
		lwFactor+1,
	) // There are only lwFactor edges, but we sample +1 which means a mask entry set to false.
	if withCitation {
		seedsBase := seeds.IdentitySubRule("seedsBase")
		citations := seeds.FromEdges("citations", "cites", lwFactor)
		citations.UpdateKernelScopeName = seedsBase.UpdateKernelScopeName
	}
	return s, strategy
}

func createDenseTestStateGraphWithMask(
	strategy *samplerPkg.Strategy,
	g *Graph,
	dtype dtypes.DType,
	withCitation bool,
) map[string]*samplerPkg.ValueMask[*Node] {
	graphStates := make(map[string]*samplerPkg.ValueMask[*Node])
	graphStates["seeds"] = &samplerPkg.ValueMask[*Node]{
		Value: IotaFull(g, shapes.Make(dtype, lwNumPapers, 1)),
		Mask:  Ones(g, shapes.Make(dtypes.Bool, lwNumPapers)),
	}

	authorsStates := make([][][]float64, lwNumPapers)
	authorsMask := make([][]bool, lwNumPapers)
	count := 0.0
	for p := range lwNumPapers {
		authorsStates[p] = make([][]float64, lwFactor+1)
		authorsMask[p] = make([]bool, lwFactor+1)
		for a := range lwFactor + 1 {
			if a < lwFactor {
				authorsStates[p][a] = []float64{count}
				authorsMask[p][a] = true
				count++
			} else {
				authorsStates[p][a] = []float64{0}
				authorsMask[p][a] = false
			}
		}
	}
	graphStates["authors"] = &samplerPkg.ValueMask[*Node]{
		Value: ConvertDType(DivScalar(Const(g, authorsStates), 1000.0), dtype),
		Mask:  Const(g, authorsMask),
	}
	if withCitation {
		edges := strategy.ExtractSamplingEdgeIndices()
		indices := Const(g, edges["citations"].TargetIndices)
		indices = InsertAxes(indices, -1)
		citations := Gather(graphStates["seeds"].Value, indices)
		citations = Reshape(citations, lwNumPapers, lwFactor, 1)
		graphStates["citations"] = &samplerPkg.ValueMask[*Node]{
			Value: citations,
			Mask:  Ones(g, shapes.Make(dtypes.Bool, lwNumPapers, lwFactor)),
		}
		graphStates["seedsBase"] = &samplerPkg.ValueMask[*Node]{
			Value: InsertAxes(graphStates["seeds"].Value, -2), // [lwNumPapers, 1, embedding_dim]
			Mask:  InsertAxes(graphStates["seeds"].Mask, -1),  // [lwNumPapers, 1]
		}
	}
	return graphStates
}

func createDenseTestStateGraphLayerWise(
	strategy *samplerPkg.Strategy,
	g *Graph,
	dtype dtypes.DType,
	withCitation bool,
) (
	graphStates map[string]*Node, edges map[string]samplerPkg.EdgePair[*Node]) {
	graphStates = make(map[string]*Node)
	graphStates["seeds"] = IotaFull(g, shapes.Make(dtype, lwNumPapers, 1))
	graphStates["authors"] = DivScalar(IotaFull(g, shapes.Make(dtype, lwNumPapers*lwFactor, 1)), 1000.0)
	if withCitation {
		graphStates["citations"] = graphStates["seeds"]
		graphStates["seedsBase"] = graphStates["seeds"]
	}

	edges = make(map[string]samplerPkg.EdgePair[*Node])
	for name, value := range strategy.ExtractSamplingEdgeIndices() {
		edges[name] = samplerPkg.EdgePair[*Node]{
			SourceIndices: Const(g, value.SourceIndices),
			TargetIndices: Const(g, value.TargetIndices),
		}
	}

	return graphStates, edges
}

func setMinimalTestParams(ctx *context.Context) {
	ctx.SetParams(map[string]any{
		layers.ParamDropoutRate:     0.0,
		activations.ParamActivation: "none", // No activation, to make math simpler.
		layers.ParamNormalization:   "none",

		ParamEdgeDropoutRate:       0.0,
		ParamNumGraphUpdates:       1, // gnn_num_messages
		ParamPoolingType:           "sum",
		ParamUpdateStateType:       "residual",
		ParamUsePathToRootStates:   false,
		ParamGraphUpdateType:       "simultaneous",
		ParamUpdateNumHiddenLayers: 0,
		ParamMessageDim:            1, // 128 or 256 will work better, but takes way more time
		ParamStateDim:              1, // 128 or 256 will work better, but takes way more time
		ParamUseRootAsContext:      false,
	})
}

func setCommonTestParams(ctx *context.Context) {
	ctx.SetParams(map[string]any{
		layers.ParamDropoutRate:     0.0,
		activations.ParamActivation: "swish",
		layers.ParamNormalization:   "layer",

		ParamEdgeDropoutRate:       0.0,
		ParamNumGraphUpdates:       3, // gnn_num_messages
		ParamPoolingType:           "mean|logsum",
		ParamUpdateStateType:       "residual",
		ParamUsePathToRootStates:   false,
		ParamGraphUpdateType:       "simultaneous",
		ParamUpdateNumHiddenLayers: 2,
		ParamMessageDim:            8,
		ParamStateDim:              8,
		ParamUseRootAsContext:      false,
	})
}

// TestLayerWiseInferenceMinimal makes sure sampled and layer-wise inference with manually edited
// weights and minimal configuration get expected results.
func TestLayerWiseInferenceMinimal(t *testing.T) {
	withCitation := false
	manager := graphtest.BuildTestBackend()
	_, strategy := createDenseTestStrategy(withCitation)
	ctx := context.New()
	setMinimalTestParams(ctx)
	// Set weights to fixed values, that makes it easier to interpret:
	{
		ctx := ctx.InAbsPath("/model/graph_update_0/gnn:authors/conv/message/dense")
		_ = ctx.VariableWithValue("weights", tensors.FromValue([][]float32{{1.0}}))
		_ = ctx.VariableWithValue("biases", tensors.FromValue([]float32{0.0}))
	}
	{
		ctx := ctx.InAbsPath("/model/graph_update_0/gnn:seeds/update/dense")
		_ = ctx.VariableWithValue("weights", tensors.FromValue([][]float32{{1000.0}, {1.0}}))
		_ = ctx.VariableWithValue("biases", tensors.FromValue([]float32{0.0}))
	}
	{
		ctx := ctx.InAbsPath("/model/readout/gnn:seeds/dense")
		_ = ctx.VariableWithValue("weights", tensors.FromValue([][]float32{{1.0}}))
		_ = ctx.VariableWithValue("biases", tensors.FromValue([]float32{0.0}))
	}

	// Normal GNN executor.
	execGnn := context.MustNewExec(manager, ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
		graphStates := createDenseTestStateGraphWithMask(strategy, g, dtypes.Float32, withCitation)
		NodePrediction(ctx.In("model"), strategy, graphStates)
		return graphStates["seeds"].Value
	})

	// For each paper: paperIdx (residual connection) + 1000*paperIdx + 0.025*paperIdx + (0+1+2+3+4)/1000
	logits := execGnn.MustExec()[0]
	fmt.Printf("\tGNN seeds states: %s\n", logits)
	//want := [][]float32{{0.010}, {1001.035}, {2002.060}, {3003.085}, {4004.110}, {5005.135}, {6006.160}, {7007.185}, {8008.210}, {9009.235}}
	want := [][]float32{{0.02}, {2002.07}, {4004.12}, {6006.17}, {8008.22}, {10010.27},
		{12012.32}, {14014.37}, {16016.42}, {18018.47}}
	require.Equal(t, want, logits.Value())

	// Uncomment to list variables used in model.
	/*
		ctx.EnumerateVariables(func(v *context.Variable) {
			fmt.Printf("\t%s=%s\n", v.GetParameterName(), v.Value())
		})
	*/

	// Layer-Wise Inference: should return the same values.
	lw, err := LayerWiseGNN(ctx, strategy)
	require.NoError(t, err)
	execLayerWise := context.MustNewExec(manager, ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
		graphStates, edges := createDenseTestStateGraphLayerWise(strategy, g, dtypes.Float32, withCitation)
		lw.NodePrediction(ctx.In("model"), graphStates, edges)
		return graphStates["seeds"]
	})
	logits = execLayerWise.MustExec()[0]
	fmt.Printf("\tLayerWiseGNN seeds states: %s\n", logits)
	require.Equal(t, want, logits.Value())
}

// TestLayerWiseInferenceCommon makes sure sampled and layer-wise inference get the same results under
// common configuration parameters
func TestLayerWiseInferenceCommon(t *testing.T) {
	for _, withCitation := range []bool{false, true} {
		fmt.Printf("\nwithCitation=%v:\n", withCitation)
		manager := graphtest.BuildTestBackend()
		_, strategy := createDenseTestStrategy(withCitation)
		ctx := context.New()
		setCommonTestParams(ctx)

		// Normal GNN executor.
		execGnn := context.MustNewExec(manager, ctx, func(ctx *context.Context, g *Graph) *Node {
			graphStates := createDenseTestStateGraphWithMask(strategy, g, dtypes.Float32, withCitation)
			NodePrediction(ctx, strategy, graphStates)
			return graphStates["seeds"].Value
		})

		sampledLogits := execGnn.MustExec()[0]
		fmt.Printf("\tGNN seeds states: %s\n", sampledLogits.GoStr())

		// Uncomment to list variables used in model.
		/*
			ctx.EnumerateVariables(func(v *context.Variable) {
				fmt.Printf("\t%s=%s\n", v.GetParameterName(), v.Value())
			})
		*/

		// Layer-Wise Inference: should return the same values.
		lw, err := LayerWiseGNN(ctx, strategy)
		require.NoError(t, err)
		execLayerWise := context.MustNewExec(manager, ctx.Reuse(), func(ctx *context.Context, g *Graph) *Node {
			graphStates, edges := createDenseTestStateGraphLayerWise(strategy, g, dtypes.Float32, withCitation)
			lw.NodePrediction(ctx, graphStates, edges)
			return graphStates["seeds"]
		})
		lwLogits := execLayerWise.MustExec()[0]
		fmt.Printf("\tLayerWiseGNN seeds states: %s\n", lwLogits.GoStr())
		require.True(t, sampledLogits.InDelta(lwLogits, 1e-4))
	}
}
