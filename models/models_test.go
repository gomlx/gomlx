package models

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/models/builderiface"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/stretchr/testify/require"
)

// runTestModel runs a test for a model and checks that the outputs match the wanted values.
func runTestModel[B builderiface.FnSet](t *testing.T, testName string, buildFn B, inputs []any, want []any, delta float64) {
	backend := graphtest.BuildTestBackend()
	t.Run(testName, func(t *testing.T) {
		// Convert the want values to tensors.
		wantTensors := xslices.Map(want, func(value any) *tensors.Tensor {
			if s, ok := value.(shapes.Shape); ok {
				return tensors.FromShape(s)
			}
			return tensors.FromAnyValue(value)
		})

		// Convert inputs to tensors.
		for i, input := range inputs {
			t := tensors.FromAnyValue(input)
			fmt.Printf("\tInputs[%d]:  %s\n", i, t.GoStr())
			inputs[i] = t
		}

		// Build and compile model.
		e, err := NewExec(backend, buildFn)
		require.NoError(t, err, "while building/compiling model")

		// Execute model, print outputs.
		outputs, err := e.Exec(inputs...)
		require.NoError(t, err, "while executing model")
		if len(inputs) > 0 {
			fmt.Printf("\t======\n")
		}
		for i, output := range outputs {
			fmt.Printf("\tOutputs[%d]: %s\n", i, output.GoStr())
		}

		// Check outputs.
		if len(want) != len(outputs) {
			t.Fatalf("%s: number of outputs (%d) doesn't match number of wanted outputs (%d)",
				testName, len(outputs), len(wantTensors))
		}
		for ii, output := range outputs {
			require.Truef(t, wantTensors[ii].InDelta(output, delta), "%s: output #%d doesn't match wanted value %s",
				testName, ii, wantTensors[ii].GoStr())
		}
	})
}

// TestExample in the package documentation.
//
// If this breaks, please update the documentation.
func TestExample(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	myModel := &struct {
		counter *Variable
	}{
		counter: must.M1(VariableWithValue("counter", int32(0))),
	}
	incFn := func(g *graph.Graph) *graph.Node {
		currentValue := myModel.counter.ValueGraph(g)
		nextValue := graph.AddScalar(currentValue, 1)
		myModel.counter.SetValueGraph(nextValue) // Updates the counter.
		return currentValue
	}
	incExec := must.M1(NewExec(backend, incFn)) // Executor that increments the counter.
	got := incExec.Call1()
	require.Equal(t, int32(0), tensors.ToScalar[int32](got))
	got = incExec.Call1()
	require.Equal(t, int32(1), tensors.ToScalar[int32](got))
	require.Equal(t, int32(2), tensors.ToScalar[int32](myModel.counter.Value()))
}

func TestIterVariables(t *testing.T) {
	type SubStruct struct {
		V1 *Variable
		V2 *Variable
	}
	type TestStruct struct {
		StringMap map[string]*Variable
		IntMap    map[int]*Variable
		Array     [2]*Variable
		Slice     []*Variable
		Sub       *SubStruct
	}

	// Create variables and structure.
	v1 := &Variable{}
	v2 := &Variable{}
	v3 := &Variable{}
	v4 := &Variable{}
	v5 := &Variable{}
	v6 := &Variable{}
	v7 := &Variable{}
	v8 := &Variable{}

	test := &TestStruct{
		StringMap: map[string]*Variable{"a": v1, "b": v2},
		IntMap:    map[int]*Variable{1: v3, 2: v4},
		Array:     [2]*Variable{v5, v6},
		Slice:     []*Variable{v7},
		Sub:       &SubStruct{V1: v8},
	}

	// Collect all paths and variables.
	var got []PathAndVariable
	for pv := range IterVariables(test) {
		got = append(got, pv)
	}

	// Expected paths and variables -- sorted in the expected order.
	want := []PathAndVariable{
		{Path: "StringMap[a]", Variable: v1},
		{Path: "StringMap[b]", Variable: v2},
		{Path: "IntMap[1]", Variable: v3},
		{Path: "IntMap[2]", Variable: v4},
		{Path: "Array[0]", Variable: v5},
		{Path: "Array[1]", Variable: v6},
		{Path: "Slice[0]", Variable: v7},
		{Path: "Sub.V1", Variable: v8},
	}

	require.Equal(t, len(want), len(got), "different number of variables found")
	for ii := range want {
		require.Equal(t, want[ii].Path, got[ii].Path,
			"path at position %d different: want=%q got=%q", ii, want[ii].Path, got[ii].Path)
		require.Equal(t, want[ii].Variable, got[ii].Variable,
			"variable at position %d different", ii)
	}
}
