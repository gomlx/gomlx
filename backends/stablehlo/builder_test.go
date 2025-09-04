package stablehlo_test

import (
	"testing"

	"github.com/gomlx/gomlx/graph"
	"github.com/stretchr/testify/require"
)

// TestBinaryOp covers the different types of automatic broadcasting for binary operations.
func TestBinaryOp(t *testing.T) {
	// Just return a constant.
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.Add(lhs, rhs)
	})

	runTestCase := func(name string, lhs, rhs, want any) {
		t.Run(name, func(t *testing.T) {
			results, err := exec.CallOrError(lhs, rhs)
			require.NoError(t, err)
			require.Len(t, results, 1)
			require.Equal(t, want, results[0].Value())
		})
	}

	runTestCase("same shape", -2.0, 3.0, 1.0)
	runTestCase("lhs scalar", int32(-2), []int32{1, 5}, []int32{-1, 3})
	runTestCase("rhs scalar", []complex64{1i, 5i}, complex64(-2), []complex64{-2 + 1i, -2 + 5i})
	runTestCase("broadcast lhs to rhs", []int8{-1}, []int8{1, 5}, []int8{0, 4})
	runTestCase("broadcast both sides",
		[][]float64{{-1}, {-2}},
		[][]float64{{1, 5}},
		[][]float64{{0, 4}, {-1, 3}})
}
