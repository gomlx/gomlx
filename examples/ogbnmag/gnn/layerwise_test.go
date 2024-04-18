package gnn

import (
	"fmt"
	samplerPkg "github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"testing"
)

func createTestSampler() *samplerPkg.Sampler {
	sampler := samplerPkg.New()
	sampler.AddNodeType("papers", 5)
	sampler.AddNodeType("authors", 10)

	authorWritesPapers := tensor.FromValue([][]int32{
		{0, 2}, // Author 0 writes paper 2.
		{3, 2},
		{4, 2},
		{0, 3},
		{0, 4},
		{4, 4},
		{7, 4},
	})
	sampler.AddEdgeType("writes", "authors", "papers", authorWritesPapers, false)
	sampler.AddEdgeType("writtenBy", "authors", "papers", authorWritesPapers, true)

	paperCitesPaper := tensor.FromValue([][]int32{
		{2, 0}, // Paper 2 cites paper 0
		{2, 1},
		{3, 0},
		{3, 2},
		{4, 1},
		{4, 3},
	})
	sampler.AddEdgeType("cites", "papers", "papers", paperCitesPaper, false)
	sampler.AddEdgeType("citedBy", "papers", "papers", paperCitesPaper, true)
	return sampler
}

func createTestStrategy() (*samplerPkg.Sampler, *samplerPkg.Strategy) {
	s := createTestSampler()
	strategy := s.NewStrategy()
	strategy = s.NewStrategy()
	seeds := strategy.NodesFromSet("Seeds", "papers", 2, []int32{2, 3, 4})
	authors := seeds.FromEdges("authors", "writtenBy", 5)
	_ = authors.FromEdges("otherPapers", "writes", 3)
	citations := seeds.FromEdges("citations", "cites", 2)
	citationsAuthors := citations.FromEdges("citationsAuthors", "writtenBy", 1)
	_ = citationsAuthors
	return s, strategy
}

func TestLayerWiseOperations(t *testing.T) {
	sampler := createTestSampler()
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager)

	numPapers := int(sampler.NodeTypesToCount["papers"])
	papersEmbedSize := 10
	edgeType := sampler.EdgeTypes["cites"]
	e := context.NewExec(manager, ctx, func(ctx *context.Context, g *Graph) []*Node {
		sourceState := IotaFull(g, shapes.Make(shapes.F32, numPapers, papersEmbedSize))
		edgesState := gatherToEdgesGraph(ctx, sourceState, edgeType)
		pooled := poolEdgesGraph(ctx, edgesState, edgeType)
		return []*Node{edgesState, pooled}
	})

	var gatheredToEdges, pooledToNodes tensor.Tensor
	require.NotPanics(t, func() {
		res := e.Call()
		gatheredToEdges, pooledToNodes = res[0], res[1]
	})

	{ // Gathered state into edges:
		fmt.Printf("gatheredToEdges=%s\n", gatheredToEdges.Local().GoStr())
		// Based on the edges list, the gathered results should represent the values gathered from the
		// right column below:
		//	paper2 <- paper0
		//	paper2 <- paper1
		//	paper3 <- paper0
		//	paper3 <- paper2
		//	paper4 <- paper1
		//	paper4 <- paper3
		want := [][]float32{
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			{20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
			{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
			{30, 31, 32, 33, 34, 35, 36, 37, 38, 39}}
		require.Equal(t, want, gatheredToEdges.Local().Value())
	}

	{ // Edges state (above) sum-pooled to target nodes.
		fmt.Printf("pooledToNodes=%s\n", pooledToNodes.Local().GoStr())
		want := [][]float32{
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28},      // Mean(Paper0, Paper1), (Paper 0 + Paper 1)
			{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38}, // Mean(Paper0, Paper2), (Paper 0 + Paper 2)
			{20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58}, // Mean(Paper1, Paper3), (Paper 1 + Paper 3),
		}
		require.Equal(t, want, pooledToNodes.Local().Value())
	}
}

func TestTargetIndicesForEdgeType(t *testing.T) {
	sampler := createTestSampler()
	edgeType := sampler.EdgeTypes["cites"]
	indices := targetIndicesForEdgeType(edgeType)
	fmt.Printf("targetIndicesForEdgeType(%q)=%#v\n", edgeType.Name, indices)
	// Based on the edges list, the gathered results should represent the values gathered from the
	// left column below:
	//	paper2 <- paper0
	//	paper2 <- paper1
	//	paper3 <- paper0
	//	paper3 <- paper2
	//	paper4 <- paper1
	//	paper4 <- paper3
	want := []int32{2, 2, 3, 3, 4, 4}
	require.Equal(t, want, indices)
}
