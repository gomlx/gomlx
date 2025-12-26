package graph_test

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// TestWhileCountTo10 tests a simple While loop that counts from 0 to 10
func TestWhileCountTo10(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test a loop that computes: counter from 0 to 10
	exec := MustNewExec(backend, func(g *Graph) *Node {
		counter := Scalar(g, dtypes.Int32, int32(0))

		fn := g.StableHLOFunction()
		require.NotNil(t, fn, "StableHLOFunction should not be nil for XLA backend")

		// Condition: counter < 10
		condFn := fn.Closure()
		condInput, err := condFn.Input(xla.ShapeToXLA(counter.Shape()))
		require.NoError(t, err)
		limit, err := condFn.ConstantFromScalar(int32(10))
		require.NoError(t, err)
		cond, err := stablehlo.Compare(condInput, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		require.NoError(t, err)
		err = condFn.Return(cond)
		require.NoError(t, err)

		// Body: counter = counter + 1
		bodyFn := fn.Closure()
		bodyInput, err := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
		require.NoError(t, err)
		one, err := bodyFn.ConstantFromScalar(int32(1))
		require.NoError(t, err)
		next, err := stablehlo.Add(bodyInput, one)
		require.NoError(t, err)
		err = bodyFn.Return(next)
		require.NoError(t, err)

		// Execute While loop
		results := While(condFn, bodyFn, counter)
		require.Len(t, results, 1, "While should return 1 result")

		return results[0]
	})

	result := exec.Call()[0]
	value := result.Value().(int32)
	assert.Equal(t, int32(10), value, "Counter should reach 10")
}

// TestWhileMultipleStates tests While with multiple loop state variables
func TestWhileMultipleStates(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test a loop that computes: counter from 0 to 5, sum = 1+2+3+4+5 = 15
	exec := MustNewExec(backend, func(g *Graph) *Node {
		counter := Scalar(g, dtypes.Int32, int32(0))
		sum := Scalar(g, dtypes.Int32, int32(0))

		fn := g.StableHLOFunction()

		// Condition: counter < 5
		condFn := fn.Closure()
		condCounter, _ := condFn.Input(xla.ShapeToXLA(counter.Shape()))
		condSum, _ := condFn.Input(xla.ShapeToXLA(sum.Shape()))
		_ = condSum // Not used in condition
		limit, _ := condFn.ConstantFromScalar(int32(5))
		cond, _ := stablehlo.Compare(condCounter, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		condFn.Return(cond)

		// Body: counter += 1, sum += counter
		bodyFn := fn.Closure()
		bodyCounter, _ := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
		bodySum, _ := bodyFn.Input(xla.ShapeToXLA(sum.Shape()))
		one, _ := bodyFn.ConstantFromScalar(int32(1))
		nextCounter, _ := stablehlo.Add(bodyCounter, one)
		nextSum, _ := stablehlo.Add(bodySum, nextCounter)
		bodyFn.Return(nextCounter, nextSum)

		results := While(condFn, bodyFn, counter, sum)
		// Return the sum
		return results[1]
	})

	result := exec.Call()[0]
	value := result.Value().(int32)
	// sum = 1 + 2 + 3 + 4 + 5 = 15
	assert.Equal(t, int32(15), value, "Sum should be 15")
}

// TestWhileTensorState tests While with tensor (non-scalar) state
func TestWhileTensorState(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "XLA backend required for While loop tests")

	// Test incrementing a vector [0, 0, 0] to [5, 5, 5]
	exec := MustNewExec(backend, func(g *Graph) *Node {
		vec := Const(g, []int32{0, 0, 0})

		fn := g.StableHLOFunction()

		// Condition: check if first element < 5
		condFn := fn.Closure()
		condVec, _ := condFn.Input(xla.ShapeToXLA(vec.Shape()))
		firstElem, _ := stablehlo.Slice(condVec, []int{0}, []int{1}, []int{1})
		scalar, _ := stablehlo.Reshape(firstElem, stablehloshapes.Make(stablehlotypes.I32))
		limit, _ := condFn.ConstantFromScalar(int32(5))
		cond, _ := stablehlo.Compare(scalar, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
		condFn.Return(cond)

		// Body: add [1, 1, 1] to vector
		bodyFn := fn.Closure()
		bodyVec, _ := bodyFn.Input(xla.ShapeToXLA(vec.Shape()))
		ones, _ := bodyFn.ConstantFromFlatAndDimensions([]int32{1, 1, 1}, 3)
		nextVec, _ := stablehlo.Add(bodyVec, ones)
		bodyFn.Return(nextVec)

		results := While(condFn, bodyFn, vec)
		return results[0]
	})

	result := exec.Call()[0]
	value := result.Value().([]int32)
	assert.Equal(t, []int32{5, 5, 5}, value, "Vector should be [5, 5, 5]")
}
