// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

var flagSaveONNX = flag.String("save_onnx", "", "Save model to ONNX format.")

func handleSaveONNX(backend compute.Backend, store *model.Store) (bool, error) {
	if *flagSaveONNX == "" {
		return false, nil
	}

	if !backend.Capabilities().DynamicAxes {
		return true, errors.Errorf("backend %s does not support DynamicAxes", backend.Name())
	}

	// Create inference model.Exec wrapping Model graph function.
	// Model takes: scope, categorical, continuous, weights
	exec, err := model.NewExec(backend, store, Model)
	if err != nil {
		return true, errors.Wrap(err, "failed to create model.Exec for ONNX export")
	}

	// Configure dynamic batch axis for the 3 input parameters:
	// categorical: [batch, len(VocabulariesFeatures)] -> ["batch", ""]
	// continuous: [batch, len(Quantiles)] -> ["batch", ""]
	// weights: [batch, 1] -> ["batch", ""]
	exec.WithDynamicAxes(
		[]string{"batch", ""},
		[]string{"batch", ""},
		[]string{"batch", ""},
	)

	// Create zero inputs for testing with batch size 1.
	numCat := len(adult.Data.VocabulariesFeatures)
	numCont := len(adult.Data.Quantiles)

	catInput := tensors.FromShape(shapes.Make(dtypes.Int64, 1, numCat))
	contInput := tensors.FromShape(shapes.Make(ModelDType, 1, numCont))
	weightsInput := tensors.FromShape(shapes.Make(ModelDType, 1, 1))

	// Call inference model.Exec to test dynamic shape handling.
	out, err := exec.Call(catInput, contInput, weightsInput)
	if err != nil {
		return true, errors.Wrap(err, "failed to execute model with dynamic batch shape")
	}

	fmt.Printf("Successfully executed dynamic shape model. Output shape: %s, value: %v\n", out[0].Shape(), out[0].Value())

	return true, nil
}
