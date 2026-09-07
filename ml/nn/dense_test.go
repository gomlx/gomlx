// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/xslices"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/support/testutil"
)

func TestDense(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		// 1. Dense with ActivationNone and bias
		graphtest.RunTestGraphFnWithBackend(t, "DenseNoActivationWithBias", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, [][]float32{{1, 2}})
				w := Const(g, [][]float32{{1, 0}, {0, 1}})
				bias := Const(g, []float32{3, 4})
				y := nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs)
				loss := ReduceAllSum(y)
				grads := Gradient(loss, x, w, bias)
				return []*Node{x, w, bias}, append([]*Node{loss}, grads...)
			}, []any{
				float32(10.0),
				[][]float32{{1, 1}},
				[][]float32{{1, 1}, {2, 2}},
				[]float32{1, 1},
			}, xslices.Epsilon)

		// 2. Dense with ActivationRelu (VJPRequiresInput == false)
		graphtest.RunTestGraphFnWithBackend(t, "DenseRelu", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, [][]float32{{1, -2}})
				w := Const(g, [][]float32{{2, 0}, {0, 3}})
				bias := Const(g, []float32{-1, 1})
				// x @ w + bias = [1*2 - 1, -2*3 + 1] = [1, -5]
				// relu([1, -5]) = [1, 0]
				// loss = 1.0
				// d(loss)/d(output) = [1, 0] (since second element <= 0)
				// d(loss)/dx = [1*2, 0*3] = [2, 0]
				// d(loss)/dw = [[1*1, 0], [-2*1, 0]] = [[1, 0], [-2, 0]]
				// d(loss)/dbias = [1, 0]
				y := nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs, activation.TypeRelu)
				loss := ReduceAllSum(y)
				grads := Gradient(loss, x, w, bias)
				return []*Node{x, w, bias}, append([]*Node{loss}, grads...)
			}, []any{
				float32(1.0),
				[][]float32{{2, 0}},
				[][]float32{{1, 0}, {-2, 0}},
				[]float32{1, 0},
			}, xslices.Epsilon)

		// 3. Dense with ActivationGelu (VJPRequiresInput == true)
		graphtest.RunTestGraphFnWithBackend(t, "DenseGelu", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, [][]float32{{1, 0}})
				w := Const(g, [][]float32{{1, 0}, {0, 1}})
				bias := Const(g, []float32{0, 0})
				// x @ w + bias = [1, 0]
				// Gelu([1, 0]) ≈ [0.8413447, 0]
				y := nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs, activation.TypeGelu)
				loss := ReduceAllSum(y)
				grads := Gradient(loss, x, w, bias)
				return []*Node{x, w, bias}, append([]*Node{loss}, grads...)
			}, []any{
				float32(0.8413447),
				[][]float32{{1.0833155, 0.5}},
				[][]float32{{1.0833155, 0.5}, {0, 0}},
				[]float32{1.0833155, 0.5},
			}, 1e-4)

		// 4. Dense without bias (bias == nil)
		graphtest.RunTestGraphFnWithBackend(t, "DenseNoBias", backend,
			func(g *Graph) (inputs, outputs []*Node) {
				x := Const(g, [][]float32{{1, 2}})
				w := Const(g, [][]float32{{1, 0}, {0, 1}})
				y := nn.Dense(x, w, nil, compute.DenseLayoutInputOutputs)
				loss := ReduceAllSum(y)
				grads := Gradient(loss, x, w)
				return []*Node{x, w}, append([]*Node{loss}, grads...)
			}, []any{
				float32(3.0),
				[][]float32{{1, 1}},
				[][]float32{{1, 1}, {2, 2}},
			}, xslices.Epsilon)
	})
}
