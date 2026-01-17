package context

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReuseContextMultipleExecs verifies that variables persist across multiple NewExec calls
// when using the same reuse context from ctx.Reuse().
func TestReuseContextMultipleExecs(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := New()

	// First execution: Initialize counter to 10 using ctx.Reuse()
	// This creates the variable in the reuse context
	exec1, err := NewExec(backend, ctx.Reuse(), func(testCtx *Context, input *graph.Node) *graph.Node {
		g := input.Graph()

		// Create counter variable with initial value
		// Use Checked(false) to allow creating the variable even if it doesn't exist yet
		initCtx := testCtx.Checked(false)
		counterVar := initCtx.VariableWithShape("counter", shapes.Make(dtypes.Int32, 1))
		counter := counterVar.ValueGraph(g)

		// Set counter to 10
		newCounter := graph.Const(g, []int32{10})
		counterVar.SetValueGraph(newCounter)

		return counter
	})
	require.NoError(t, err)

	results1 := exec1.MustExec(int32(0))
	counter1 := results1[0].Value().([]int32)[0]
	t.Logf("After exec1 (init): counter = %d", counter1)

	// Second execution: Increment counter using ctx.Reuse() AGAIN
	// This should access the SAME variable
	exec2, err := NewExec(backend, ctx.Reuse(), func(testCtx *Context, input *graph.Node) *graph.Node {
		g := input.Graph()

		// Get counter variable (should be 10)
		counter := testCtx.VariableWithShape("counter", shapes.Make(dtypes.Int32, 1)).ValueGraph(g)

		// Increment by 5
		newCounter := graph.AddScalar(counter, 5.0)
		newCounter = graph.ConvertDType(newCounter, dtypes.Int32)
		testCtx.VariableWithShape("counter", newCounter.Shape()).SetValueGraph(newCounter)

		return counter // Return OLD value before update
	})
	require.NoError(t, err)

	results2 := exec2.MustExec(int32(0))
	counter2 := results2[0].Value().([]int32)[0]
	t.Logf("After exec2 (increment): counter = %d (returned old value)", counter2)
	assert.Equal(t, int32(10), counter2, "Should return value BEFORE increment")

	// Third execution: Read counter (should be 15) using ctx.Reuse() AGAIN
	exec3, err := NewExec(backend, ctx.Reuse(), func(testCtx *Context, input *graph.Node) *graph.Node {
		g := input.Graph()

		// Get counter variable (should be 15 now)
		counter := testCtx.VariableWithShape("counter", shapes.Make(dtypes.Int32, 1)).ValueGraph(g)

		return counter
	})
	require.NoError(t, err)

	results3 := exec3.MustExec(int32(0))
	counter3 := results3[0].Value().([]int32)[0]
	t.Logf("After exec3 (read): counter = %d", counter3)
	assert.Equal(t, int32(15), counter3, "Counter should have persisted the increment")
}

// TestReuseContextSeparateCalls verifies what happens when calling ctx.Reuse() separately
func TestReuseContextSeparateCalls(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := New()

	// First execution with ctx.Reuse()
	exec1, err := NewExec(backend, ctx.Reuse(), func(testCtx *Context, input *graph.Node) *graph.Node {
		g := input.Graph()

		// First call to VariableWithShape - create/get the variable
		counter := testCtx.Checked(false).VariableWithShape("counter", shapes.Make(dtypes.Int32, 1)).ValueGraph(g)

		// Second call to VariableWithShape on same variable - test it works
		newCounter := graph.Const(g, []int32{10})
		testCtx.VariableWithShape("counter", newCounter.Shape()).SetValueGraph(newCounter)

		return counter
	})
	require.NoError(t, err)

	results1 := exec1.MustExec(int32(0))
	counter1 := results1[0].Value().([]int32)[0]
	t.Logf("After exec1: counter = %d", counter1)

	// Second execution with ANOTHER ctx.Reuse() call
	exec2, err := NewExec(backend, ctx.Reuse(), func(testCtx *Context, input *graph.Node) *graph.Node {
		g := input.Graph()

		// Try to get counter variable
		counter := testCtx.VariableWithShape("counter", shapes.Make(dtypes.Int32, 1)).ValueGraph(g)

		return counter
	})
	require.NoError(t, err)

	results2 := exec2.MustExec(int32(0))
	counter2 := results2[0].Value().([]int32)[0]
	t.Logf("After exec2 (separate Reuse()): counter = %d", counter2)

	// Check if it's the same value (should be 10 if ctx.Reuse() returns same context)
	// or 0 if it's a different context
	assert.Equal(t, int32(10), counter2, "ctx.Reuse() should return the same reuse context each time")
}
