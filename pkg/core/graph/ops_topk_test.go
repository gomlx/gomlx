package graph_test

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/stretchr/testify/require"
)

func TestTopK(t *testing.T) {
	t.Run("TopK basic", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopK basic",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0, 4.0}})
				values, indices := graph.TopK(input, 3)
				return nil, []*graph.Node{values, indices}
			}, []any{
				[][]float32{{5.0, 4.0, 3.0}},
				[][]int32{{1, 4, 2}},
			}, -1)
	})

	t.Run("TopK k=1", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopK k=1",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, -5.0, 3.0, 2.0}})
				values, indices := graph.TopK(input, 1)
				return nil, []*graph.Node{values, indices}
			}, []any{
				[][]float32{{3.0}},
				[][]int32{{2}},
			}, -1)
	})

	t.Run("TopK all elements", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopK all elements",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
				values, indices := graph.TopK(input, 4)
				return nil, []*graph.Node{values, indices}
			}, []any{
				[][]float32{{5.0, 3.0, 2.0, 1.0}},
				[][]int32{{1, 2, 3, 0}},
			}, -1)
	})

	t.Run("TopK multiple batches", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopK multiple batches",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				input := graph.Const(g, [][]float32{
					{1.0, 5.0, 3.0, 2.0},
					{10.0, 2.0, 8.0, 6.0},
				})
				values, indices := graph.TopK(input, 2)
				return nil, []*graph.Node{values, indices}
			}, []any{
				[][]float32{{5.0, 3.0}, {10.0, 8.0}},
				[][]int32{{1, 2}, {0, 2}},
			}, -1)
	})

	t.Run("k=-1 panics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "TestTopKInvalid")
		input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
		require.Panics(t, func() {
			graph.TopK(input, -1)
		})
	})

	t.Run("k too large panics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "TestTopKInvalid")
		input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
		require.Panics(t, func() {
			graph.TopK(input, 10)
		})
	})
}

func TestTopKMask(t *testing.T) {
	t.Run("TopKMask basic", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopKMask basic",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
				mask := graph.TopKMask(input, 2)
				return nil, []*graph.Node{mask}
			}, []any{
				[][]bool{{false, true, true, false}},
			}, -1)
	})

	t.Run("TopKMask k=1", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopKMask k=1",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, -5.0, 3.0, 2.0}})
				mask := graph.TopKMask(input, 1)
				return nil, []*graph.Node{mask}
			}, []any{
				[][]bool{{false, false, true, false}},
			}, -1)
	})

	t.Run("TopKMask all elements", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopKMask all elements",
			func(g *graph.Graph) ([]*graph.Node, []*graph.Node) {
				input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
				mask := graph.TopKMask(input, 4)
				return nil, []*graph.Node{mask}
			}, []any{
				[][]bool{{true, true, true, true}},
			}, -1)
	})

	t.Run("TopKMask multiple batches", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "TopKMask multiple batches",
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				input := graph.Const(g, [][]float32{
					{1.0, 5.0, 3.0, 2.0},
					{10.0, 2.0, 8.0, 6.0},
				})
				mask := graph.TopKMask(input, 2)
				return nil, []*graph.Node{mask}
			}, []any{
				[][]bool{
					{false, true, true, false},
					{true, false, true, false},
				},
			}, -1)
	})

	t.Run("k=-1 panics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "TestTopKMaskInvalid")
		input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
		require.Panics(t, func() {
			graph.TopKMask(input, -1)
		})
	})

	t.Run("k too large panics", func(t *testing.T) {
		backend := graphtest.BuildTestBackend()
		g := graph.NewGraph(backend, "TestTopKMaskInvalid")
		input := graph.Const(g, [][]float32{{1.0, 5.0, 3.0, 2.0}})
		require.Panics(t, func() {
			graph.TopKMask(input, 10)
		})
	})
}
