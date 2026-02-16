package graph_test

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestDot(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("Product", func(t *testing.T) {
		g := NewGraph(backend, t.Name())

		// Shape: [batch=4, dims=3]
		inputs := Const(g, [][]float32{{1.1, 2.2, 3.3}, {11, 22, 33}, {111, 222, 333}, {1111, 2222, 3333}})
		// Layer 0: outputShapes [3, 2], that is the inputNodes have dim=3, and should output dims=2
		w0 := Const(g, [][]float32{{1, 0}, {1, -1}, {-1, 1}})
		// Dot(inputNodes, w0) -> outputShapes [batch=4, dims=2]
		Dot(inputs, w0).Product() // The last node created in the graph is taken as output by default.
		got := compileRunAndTakeFirst(t, g)
		want := tensors.FromValue([][]float32{{0, 1.1}, {0, 11}, {0, 111}, {0, 1111}})
		if !want.InDelta(got, Epsilon) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	})
}

func TestGradientDot(t *testing.T) {
	graphtest.RunTestGraphFn(t, "dot(vector,vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2).Product()
		gradients := Gradient(output, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		float32(24),           // dot product output
		[]float32{3, 3, 3, 3}, // gradient with respect to v1
		[]float32{2, 2, 2, 2}, // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "dot(matrix,vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2).Product()
		// sum := ReduceAllSum(output)
		// gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		// outputs = append([]*Node{output}, gradients...)
		outputs = []*Node{output, v1, v2}
		return
	}, []any{
		[]float32{24, 36},                       // dot product output
		[][]float32{{3, 3, 3, 3}, {3, 3, 3, 3}}, // gradient with respect to v1
		[]float32{5, 5, 5, 5},                   // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "dot(matrix,matrix)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Add(Iota(g, MakeShape(F32, 4, 1), 0), Const(g, float32(1)))
		output := Dot(v1, v2).Product()
		require.NoError(t, output.Shape().CheckDims(2, 1))
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		[][]float32{{20}, {30}},                 // dot product output
		[][]float32{{1, 2, 3, 4}, {1, 2, 3, 4}}, // gradient with respect to v1
		[][]float32{{5}, {5}, {5}, {5}},         // gradient with respect to v2
	}, Epsilon)
}
