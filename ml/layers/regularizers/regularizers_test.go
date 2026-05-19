// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package regularizers

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/modeltest"
	"github.com/gomlx/gomlx/ml/train"
)

func TestConstantL1(t *testing.T) {
	modeltest.RunTestGraphFn(t, "ConstantL1 regularizer", func(scope *model.Scope, g *Graph) (inputs, outputs []*Node) {
		wVar := scope.VariableWithValue("w", [][]float32{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}})
		w := wVar.NodeValue(g)
		ConstantL1(0.1)(scope, g, wVar)
		loss := train.GetLosses(scope, g)
		inputs = []*Node{w}
		outputs = []*Node{loss, Gradient(loss, w)[0]}
		return
	}, []any{
		float32(8 * 0.1), // Total regularization loss.
		[][]float32{{-0.1, 0, 0, 0, 0.1}, {-0.1, 0, 0, 0, 0.1}}, // Gradient.
	}, 1e-4)
}
