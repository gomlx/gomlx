package models

import (
	"fmt"
	"runtime"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type biasModel struct {
	Bias float64
}

func (b *biasModel) AddBias(x *graph.Node) *graph.Node {
	return graph.AddScalar(x, b.Bias)
}

func TestExec(t *testing.T) {
	//t.Parallel()
	model := &biasModel{Bias: 7}
	runTestModel(t, "NoVariables-int32", model.AddBias, []any{int32(4)}, []any{int32(11)}, -1)
	runTestModel(t, "NoVariables-float32", model.AddBias, []any{float32(6)}, []any{float32(13)}, -1)

	// Experiment with variables: our model is simply a variable "counter":
	// - Variable as input:
	counter, err := VariableWithValue("counter", int32(5))
	require.NoError(t, err)
	runTestModel(t, "VariableIn", func(g *graph.Graph) *graph.Node {
		return graph.AddScalar(counter.ValueGraph(g), 1)
	}, nil, []any{int32(6)}, -1)

	// - Variable output only:
	runTestModel(t, "VariableOut", func(newValue *graph.Node) {
		counter.SetValueGraph(newValue)
	}, []any{int32(13)}, nil, -1)
	require.Equal(t, int32(13), tensors.ToScalar[int32](counter.Value()))

	// - Variable input and input:
	runTestModel(t, "VariableInOut", func(g *graph.Graph) *graph.Node {
		// Increment counter:
		counter.SetValueGraph(graph.AddScalar(counter.ValueGraph(g), 1))
		return counter.ValueGraph(g)
	}, nil, []any{int32(14)}, -1)
	require.Equal(t, int32(14), tensors.ToScalar[int32](counter.Value()))
}

func TestExecCleanup(t *testing.T) {
	// DO NOT USE t.Parallel() here!
	for _ = range 3 {
		runtime.GC()
		runtime.Gosched()
	}
	countGraphs := len(graphToExec)
	fmt.Printf("(before test) countGraphs: %d\n", countGraphs)

	model := &biasModel{Bias: 7}
	runTestModel(t, "NoVariables-float32", model.AddBias, []any{float32(6)}, []any{float32(13)}, -1)

	for _ = range 10 {
		runtime.GC()
		runtime.Gosched()
	}
	newCountGraphs := len(graphToExec)
	fmt.Printf("(after test) countGraphs:  %d\n", newCountGraphs)

	assert.Equal(t, countGraphs, newCountGraphs)
}
