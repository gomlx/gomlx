// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package main

import (
	"flag"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/examples/adult"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/onnx"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
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

	// Feature counts.
	numCategoricalFeatures := len(adult.Data.VocabulariesFeatures)
	numContinuousFeatures := len(adult.Data.Quantiles)

	// Test: call inference model.Exec to test dynamic shape handling.
	if klog.V(1).Enabled() {
		// Create zero inputs for testing with batch size 1.
		catInput := tensors.FromShape(shapes.Make(dtypes.Int64, 1, numCategoricalFeatures))
		contInput := tensors.FromShape(shapes.Make(ModelDType, 1, numContinuousFeatures))
		weightInput := tensors.FromShape(shapes.Make(ModelDType, 1, 1))

		klog.Infof("Testing dynamic batch shape with batch_size=1, inputs:")
		klog.Infof("- Categorical: %s", catInput.Shape())
		klog.Infof("- Continuous: %s", contInput.Shape())
		klog.Infof("- Weight: %s", weightInput.Shape())
		outputs, err := exec.Call(catInput, contInput, weightInput)
		if err != nil {
			return true, errors.Wrap(err, "failed to execute model with dynamic batch shape")
		}
		klog.Infof("Output: shape=%s, value=%s\n", outputs[0].Shape(), outputs[0])
	}

	// Input shapes with dynamic batch size.
	categoricalShape := shapes.MakeDynamic(dtypes.Int64, []int{shapes.DynamicDim, numCategoricalFeatures}, []string{"batch", ""})
	continuousShape := shapes.MakeDynamic(ModelDType, []int{shapes.DynamicDim, numContinuousFeatures}, []string{"batch", ""})
	weightShape := shapes.MakeDynamic(ModelDType, []int{shapes.DynamicDim, 1}, []string{"batch", ""})
	err = onnx.SaveToFile(backend, exec, *flagSaveONNX,
		[]shapes.Shape{categoricalShape, continuousShape, weightShape},
		[]string{"categorical", "continuous", "weight"},
		[]string{"is_>50K_logit"})
	if err != nil {
		return true, err
	}

	return true, nil
}
