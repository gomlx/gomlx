// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package onnx_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	onnxbackend "github.com/gomlx/compute-onnx"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/onnx"
	"github.com/stretchr/testify/require"
)

func TestSaveAfterExecutionAndWeightUpdates(t *testing.T) {
	for _, warm := range []bool{false, true} {
		for _, keep := range []bool{false, true} {
			for _, named := range []bool{false, true} {
				t.Run(fmt.Sprintf("warm=%t/keep=%t/named=%t", warm, keep, named), func(t *testing.T) {
					backend, err := compute.NewWithConfig("onnx:cpu,intra_op_num_threads=1")
					require.NoError(t, err)
					defer backend.Finalize()
					backend.(*onnxbackend.Backend).SetKeepModelProto(keep)
					store := model.NewStore()
					defer store.Finalize()
					weights, err := store.CreateVariable("/weights", tensors.FromValue([][]float32{{2, 3}, {5, 7}, {11, 13}}))
					require.NoError(t, err)
					exec, err := model.NewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
						return MatMul(x, scope.GetVariable("weights").NodeValue(x.Graph()))
					})
					require.NoError(t, err)
					defer exec.Finalize()
					input := [][]float32{{1, 2, 3}}
					if warm {
						values, err := exec.Exec(input)
						require.NoError(t, err)
						require.Equal(t, [][]float32{{45, 56}}, values[0].Value())
						require.NoError(t, values[0].FinalizeAll())
					}
					var inputNames, outputNames []string
					if named {
						inputNames, outputNames = []string{"x"}, []string{"y"}
					}
					for _, multiplier := range []float32{1, 2} {
						require.NoError(t, weights.SetValue(tensors.FromValue([][]float32{{2 * multiplier, 3 * multiplier}, {5 * multiplier, 7 * multiplier}, {11 * multiplier, 13 * multiplier}})))
						var saved bytes.Buffer
						err = onnx.Save(backend, exec, &saved, []shapes.Shape{shapes.Make(dtypes.Float32, 1, 3)}, inputNames, outputNames)
						require.NoError(t, err)
						require.Equal(t, keep, backend.(*onnxbackend.Backend).KeepModelProto())
						require.False(t, store.VariablesAsConst())
						var repeated bytes.Buffer
						err = onnx.Save(backend, exec, &repeated, []shapes.Shape{shapes.Make(dtypes.Float32, 1, 3)}, inputNames, outputNames)
						require.NoError(t, err)
						loaded, err := onnx.Load(backend, bytes.NewReader(repeated.Bytes()))
						require.NoError(t, err)
						actual, err := loaded.Call(input)
						require.NoError(t, err)
						require.Equal(t, [][]float32{{45 * multiplier, 56 * multiplier}}, actual[0].Value())
						require.NoError(t, actual[0].FinalizeAll())
						loaded.Finalize()
						current, err := exec.Exec(input)
						require.NoError(t, err)
						require.Equal(t, [][]float32{{45 * multiplier, 56 * multiplier}}, current[0].Value())
						require.NoError(t, current[0].FinalizeAll())
					}
				})
			}
		}
	}
}
