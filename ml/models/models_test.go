package models

import (
	"fmt"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/models/builderif"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/stretchr/testify/require"
)

func runTestModel[B builderif.BuilderIf](t *testing.T, testName string, buildFn B, inputs []any, want []any, delta float64) {
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

type biasModel struct {
	Bias float64
}

func (b *biasModel) AddBias(x *graph.Node) *graph.Node {
	return graph.AddScalar(x, b.Bias)
}

func TestExec(t *testing.T) {
	model := &biasModel{Bias: 7}
	runTestModel(t, "NoVariables-int32", model.AddBias, []any{int32(4)}, []any{int32(11)}, -1)
	runTestModel(t, "NoVariables-float32", model.AddBias, []any{float32(6)}, []any{float32(13)}, -1)
}
