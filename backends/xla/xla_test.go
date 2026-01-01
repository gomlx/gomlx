package xla_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehlodtypes "github.com/gomlx/go-xla/pkg/types/dtypes"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var backend backends.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, xla.BackendName))
	} else {
		fmt.Printf("\t$%s=%q\n", backends.ConfigEnvVar, os.Getenv(backends.ConfigEnvVar))
	}
	backend = backends.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
	fmt.Printf("\t- Add flag -vmodule=executable=2 to log the StableHLO program being executed.\n")
	for deviceNum := range backends.DeviceNum(backend.NumDevices()) {
		fmt.Printf("\t- Device #%d: %s\n", deviceNum, backend.DeviceDescription(deviceNum))
	}
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestCompileAndRun(t *testing.T) {
	// Just return a constant.
	exec := graph.MustNewExec(backend, func(g *graph.Graph) *graph.Node { return graph.Const(g, float32(-7)) })
	y0 := exec.MustExec()[0]
	assert.Equal(t, float32(-7), y0.Value())
}

// getXLABuilder creates an XLA builder for testing and returns it with type assertion.
func getXLABuilder(t *testing.T, name string) *xla.Builder {
	t.Helper()
	b := backend.Builder(name)
	require.NotNil(t, b)
	xlaBld, ok := b.(*xla.Builder)
	require.True(t, ok, "expected *xla.Builder, got %T", b)
	return xlaBld
}

func TestWhile_CountToTen(t *testing.T) {
	b := getXLABuilder(t, "test_while_count")

	// Create initial state: counter = 0
	counterOp, err := b.Constant([]int32{0})
	require.NoError(t, err)

	// Create condition function: counter < 10
	condFn := b.TestClosure()
	condCounter, err := condFn.Input(stablehloshapes.Make(stablehlodtypes.Int32))
	require.NoError(t, err)
	limit, err := condFn.ConstantFromScalar(int32(10))
	require.NoError(t, err)
	cond, err := stablehlo.Compare(condCounter, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
	require.NoError(t, err)
	err = condFn.Return(cond)
	require.NoError(t, err)

	// Create body function: counter = counter + 1
	bodyFn := b.TestClosure()
	bodyCounter, err := bodyFn.Input(stablehloshapes.Make(stablehlodtypes.Int32))
	require.NoError(t, err)
	one, err := bodyFn.ConstantFromScalar(int32(1))
	require.NoError(t, err)
	nextCounter, err := stablehlo.Add(bodyCounter, one)
	require.NoError(t, err)
	err = bodyFn.Return(nextCounter)
	require.NoError(t, err)

	// Execute while loop
	results, err := b.While(condFn, bodyFn, counterOp)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Compile and run
	exec, err := b.Compile(results, nil)
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify result is 10
	result := make([]int32, 1)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[0], result)
	require.NoError(t, err)
	assert.Equal(t, int32(10), result[0], "While loop should count from 0 to 10")
}

func TestWhile_MultipleStateValues(t *testing.T) {
	b := getXLABuilder(t, "test_while_multiple_states")

	// Create initial states: counter = 0, sum = 0
	counterOp, err := b.Constant([]int32{0})
	require.NoError(t, err)
	sumOp, err := b.Constant([]int32{0})
	require.NoError(t, err)

	counterShape := stablehloshapes.Make(stablehlodtypes.Int32)
	sumShape := stablehloshapes.Make(stablehlodtypes.Int32)

	// Create condition function: counter < 5
	condFn := b.TestClosure()
	condCounter, err := condFn.Input(counterShape)
	require.NoError(t, err)
	_, err = condFn.Input(sumShape) // sum input, not used in condition
	require.NoError(t, err)
	limit, err := condFn.ConstantFromScalar(int32(5))
	require.NoError(t, err)
	cond, err := stablehlo.Compare(condCounter, limit, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
	require.NoError(t, err)
	err = condFn.Return(cond)
	require.NoError(t, err)

	// Create body function: counter += 1, sum += counter
	bodyFn := b.TestClosure()
	bodyCounter, err := bodyFn.Input(counterShape)
	require.NoError(t, err)
	bodySum, err := bodyFn.Input(sumShape)
	require.NoError(t, err)
	one, err := bodyFn.ConstantFromScalar(int32(1))
	require.NoError(t, err)
	nextCounter, err := stablehlo.Add(bodyCounter, one)
	require.NoError(t, err)
	nextSum, err := stablehlo.Add(bodySum, nextCounter)
	require.NoError(t, err)
	err = bodyFn.Return(nextCounter, nextSum)
	require.NoError(t, err)

	// Execute while loop
	results, err := b.While(condFn, bodyFn, counterOp, sumOp)
	require.NoError(t, err)
	require.Len(t, results, 2)

	// Compile and run
	exec, err := b.Compile(results, nil)
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 2)

	// Verify: counter = 5, sum = 1+2+3+4+5 = 15
	counterResult := make([]int32, 1)
	sumResult := make([]int32, 1)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[0], counterResult)
	require.NoError(t, err)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[1], sumResult)
	require.NoError(t, err)
	assert.Equal(t, int32(5), counterResult[0], "Counter should be 5")
	assert.Equal(t, int32(15), sumResult[0], "Sum should be 15 (1+2+3+4+5)")
}

func TestWhile_InvalidConditionFunction(t *testing.T) {
	b := getXLABuilder(t, "test_while_invalid_cond")

	counterOp, err := b.Constant([]int32{0})
	require.NoError(t, err)

	// Pass a string instead of a stablehlo.Function
	_, err = b.While("not a function", nil, counterOp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "condFn must be a *stablehlo.Function")
}

func TestSort_SingleArray(t *testing.T) {
	b := getXLABuilder(t, "test_sort_single")

	// Create input array: [3, 1, 4, 1, 5, 9, 2, 6]
	inputOp, err := b.Constant([]int32{3, 1, 4, 1, 5, 9, 2, 6}, 8)
	require.NoError(t, err)

	inputShape := stablehloshapes.Make(stablehlodtypes.Int32)

	// Create comparator function: lhs < rhs (ascending order)
	compFn := b.TestClosure()
	lhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	rhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	cmp, err := stablehlo.Compare(lhs, rhs, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
	require.NoError(t, err)
	err = compFn.Return(cmp)
	require.NoError(t, err)

	// Execute sort
	results, err := b.Sort(compFn, 0, true, inputOp)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Compile and run
	exec, err := b.Compile(results, nil)
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify sorted result
	result := make([]int32, 8)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[0], result)
	require.NoError(t, err)
	expected := []int32{1, 1, 2, 3, 4, 5, 6, 9}
	assert.Equal(t, expected, result, "Array should be sorted in ascending order")
}

func TestSort_DescendingOrder(t *testing.T) {
	b := getXLABuilder(t, "test_sort_descending")

	// Create input array: [3, 1, 4, 1, 5]
	inputOp, err := b.Constant([]float32{3.0, 1.0, 4.0, 1.0, 5.0}, 5)
	require.NoError(t, err)

	inputShape := stablehloshapes.Make(stablehlodtypes.Float32)

	// Create comparator function: lhs > rhs (descending order)
	compFn := b.TestClosure()
	lhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	rhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	cmp, err := stablehlo.Compare(lhs, rhs, stablehlotypes.CompareGT, stablehlotypes.CompareFloat)
	require.NoError(t, err)
	err = compFn.Return(cmp)
	require.NoError(t, err)

	// Execute sort
	results, err := b.Sort(compFn, 0, true, inputOp)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Compile and run
	exec, err := b.Compile(results, nil)
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify sorted result
	result := make([]float32, 5)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[0], result)
	require.NoError(t, err)
	expected := []float32{5.0, 4.0, 3.0, 1.0, 1.0}
	assert.Equal(t, expected, result, "Array should be sorted in descending order")
}

func TestSort_2DArray(t *testing.T) {
	// Sort 2D array along axis 1 (within each row)
	b := getXLABuilder(t, "test_sort_2d")

	// Create 2x3 array: [[3, 1, 2], [6, 4, 5]]
	inputOp, err := b.Constant([]int32{3, 1, 2, 6, 4, 5}, 2, 3)
	require.NoError(t, err)

	inputShape := stablehloshapes.Make(stablehlodtypes.Int32)

	// Create comparator function: lhs < rhs (ascending order)
	compFn := b.TestClosure()
	lhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	rhs, err := compFn.Input(inputShape)
	require.NoError(t, err)
	cmp, err := stablehlo.Compare(lhs, rhs, stablehlotypes.CompareLT, stablehlotypes.CompareSigned)
	require.NoError(t, err)
	err = compFn.Return(cmp)
	require.NoError(t, err)

	// Sort along axis 1 (within each row)
	results, err := b.Sort(compFn, 1, true, inputOp)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Compile and run
	exec, err := b.Compile(results, nil)
	require.NoError(t, err)

	outputs, err := exec.Execute(nil, nil, 0)
	require.NoError(t, err)
	require.Len(t, outputs, 1)

	// Verify sorted result: each row should be sorted
	// [[1, 2, 3], [4, 5, 6]]
	result := make([]int32, 6)
	err = backend.(*xla.Backend).BufferToFlatData(outputs[0], result)
	require.NoError(t, err)
	expected := []int32{1, 2, 3, 4, 5, 6}
	assert.Equal(t, expected, result, "Each row should be sorted")
}

func TestSort_InvalidComparatorFunction(t *testing.T) {
	b := getXLABuilder(t, "test_sort_invalid_comp")

	inputOp, err := b.Constant([]int32{3, 1, 2}, 3)
	require.NoError(t, err)

	// Pass a number instead of a stablehlo.Function
	_, err = b.Sort(42, 0, true, inputOp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "comparatorFn must be a *stablehlo.Function")
}
