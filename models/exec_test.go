package models

import (
	"fmt"
	"runtime"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/graph"
	"github.com/stretchr/testify/assert"
)

type biasModel struct {
	Bias float64
}

func (b *biasModel) AddBias(x *graph.Node) *graph.Node {
	return graph.AddScalar(x, b.Bias)
}

func TestExec(t *testing.T) {
	t.Parallel()
	model := &biasModel{Bias: 7}
	runTestModel(t, "NoVariables-int32", model.AddBias, []any{int32(4)}, []any{int32(11)}, -1)
	runTestModel(t, "NoVariables-float32", model.AddBias, []any{float32(6)}, []any{float32(13)}, -1)
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
