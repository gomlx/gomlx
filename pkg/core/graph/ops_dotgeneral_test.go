package graph_test

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

func TestDot(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("Product", func(t *testing) {
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
