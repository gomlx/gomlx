package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/stretchr/testify/require"
)

// runTestModel runs a test for a model and checks that the outputs match the wanted values.
func runTestModel[B BuilderFnSet](t *testing.T, testName string, buildFn B, inputs []any, want []any, delta float64) {
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

func TestSaveLoad(t *testing.T) {
	type NestedStruct struct {
		Scalar *Variable
		Vector *Variable
	}
	type ModelStruct struct {
		Items        map[string]*NestedStruct
		LearningRate float64
	}

	// Create model
	model := &ModelStruct{
		Items: map[string]*NestedStruct{
			"0": {
				Scalar: must.M1(VariableWithValue("scalar1", float32(1.0))),
				Vector: must.M1(VariableWithValue("vector1", []float32{1.0, 2.0, 3.0})),
			},
			"1": {
				Scalar: must.M1(VariableWithValue("scalar2", float32(2.0))),
				Vector: must.M1(VariableWithValue("vector2", []float32{4.0, 5.0, 6.0})),
			},
		},
		LearningRate: 1e-3,
	}

	// Save model
	tmpDir := t.TempDir()
	tmpBase := path.Join(tmpDir, "model")
	fmt.Printf("- Saving model to %s.{bin,json}\n", tmpBase)
	err := Save(model, tmpBase)
	require.NoError(t, err)

	jsonPath := tmpBase + ".jsonl"
	jsonContents, err := os.ReadFile(jsonPath)
	var indentedJsonContents bytes.Buffer
	require.NoError(t, err)
	fmt.Printf("- Contents of the %s:\n", jsonPath)
	for ii, part := range strings.Split(string(jsonContents), "\n") {
		if part == "" {
			continue
		}
		if ii > 0 {
			indentedJsonContents.WriteString("\n\n  ")
		}
		err = json.Indent(&indentedJsonContents, []byte(part), "  ", "  ")
	}
	require.NoErrorf(t, err, "Full error: %+v\n", err)
	fmt.Printf("  %s\n", indentedJsonContents.String())

	// Load model
	loadedModel := &ModelStruct{}
	err = Load(loadedModel, tmpBase)
	require.NoError(t, err)

	// Verify values
	require.Equal(t, model.LearningRate, loadedModel.LearningRate)
	require.Equal(t, len(model.Items), len(loadedModel.Items))
	for key, item := range model.Items {
		require.Equal(t,
			tensors.ToScalar[float32](item.Scalar.Value()),
			tensors.ToScalar[float32](loadedModel.Items[key].Scalar.Value()))
		require.True(t,
			item.Vector.Value().Equal(
				loadedModel.Items[key].Vector.Value()))
	}
}
