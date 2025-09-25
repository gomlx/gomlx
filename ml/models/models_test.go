package models

import (
	"fmt"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/stretchr/testify/require"
)

func runTestModel[M any](t *testing.T, testName string, model *M, buildFn any, inputs []any, want []any, delta float64) {
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
		e, err := NewExec(backend, model, buildFn)
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

func TestExec(t *testing.T) {
	type biasModel struct {
		Bias float64
	}
	addBiasFn := func(m *biasModel, x *graph.Node) *graph.Node {
		return graph.AddScalar(x, m.Bias)
	}
	model := &biasModel{Bias: 7}
	runTestModel(t, "NoVariables-int32", model, addBiasFn, []any{int32(4)}, []any{int32(11)}, -1)
	runTestModel(t, "NoVariables-float32", model, addBiasFn, []any{float32(6)}, []any{float32(13)}, -1)
}
