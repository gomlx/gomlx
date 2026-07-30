// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package onnx_test

import (
	"path/filepath"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/onnx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSaveAndLoadONNX(t *testing.T) {
	backend, err := compute.NewWithConfig("onnxruntime")
	require.NoError(t, err)

	// Create a model with a linear layer initialized with prime numbers.
	store := model.NewStore()
	scope := store.RootScope()

	// Weight [3, 2] initialized with primes: 2, 3, 5, 7, 11, 13
	// Bias [2] initialized with primes: 17, 19
	wVar := scope.At("dense").VariableWithShape("weights", shapes.Make(dtypes.Float32, 3, 2))
	bVar := scope.At("dense").VariableWithShape("biases", shapes.Make(dtypes.Float32, 2))
	require.NoError(t, wVar.SetValue(tensors.FromValue([][]float32{{2, 3}, {5, 7}, {11, 13}})))
	require.NoError(t, bVar.SetValue(tensors.FromValue([]float32{17, 19})))

	modelFn := func(scope *model.Scope, input *Node) *Node {
		return layers.Dense(scope, input, true, 2)
	}

	exec, err := model.NewExec(backend, store, modelFn)
	require.NoError(t, err)

	inputShape := shapes.Make(dtypes.Float32, 1, 3)
	inputNames := []string{"x"}
	outputNames := []string{"y"}

	tmpDir := t.TempDir()
	onnxPath := filepath.Join(tmpDir, "linear_model.onnx")

	// Save the model to ONNX file.
	err = onnx.SaveToFile(backend, exec, onnxPath, []shapes.Shape{inputShape}, inputNames, outputNames)
	require.NoError(t, err)

	// Load the model back using onnx.LoadFromFile.
	loadedExec, err := onnx.LoadFromFile(backend, onnxPath)
	require.NoError(t, err)
	defer loadedExec.Finalize()

	// Run GoMLX exec directly to get expected results.
	inputVal := [][]float32{{1.0, 2.0, 3.0}}
	expectedOutputs, err := exec.Exec(inputVal)
	require.NoError(t, err)
	require.Len(t, expectedOutputs, 1)

	// Run loaded ONNX model.
	actualOutputs, err := loadedExec.Call(inputVal)
	require.NoError(t, err)
	require.Len(t, actualOutputs, 1)

	// Compare outputs.
	assert.InDeltaSlice(t, expectedOutputs[0].Value().([][]float32)[0], actualOutputs[0].Value().([][]float32)[0], 1e-5)
}
