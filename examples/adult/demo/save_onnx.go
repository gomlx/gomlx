// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package main

import (
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

// saveONNX saves the model with the weights in store to onnxPath.
func saveONNX(backend compute.Backend, store *model.Store, onnxPath string) error {
	if !backend.Capabilities().DynamicAxes {
		return errors.Errorf("backend %s does not support DynamicAxes", backend.Name())
	}

	type keepModelProtoSetter interface {
		SetKeepModelProto(keep bool)
	}
	if setter, ok := backend.(keepModelProtoSetter); ok {
		setter.SetKeepModelProto(true)
	}

	// Create inference model.Exec wrapping Model graph function.
	// Model takes: scope, categorical, continuous, weights
	exec, err := model.NewExec(backend, store, Model)
	if err != nil {
		return errors.WithMessagef(err, "failed to create inference model.Exec for ONNX export")
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

	if onnx.IsONNX(backend) {
		// Input shapes with dynamic batch size.
		categoricalShape := shapes.MakeDynamic(dtypes.Int64, []int{shapes.DynamicDim, numCategoricalFeatures}, []string{"batch", ""})
		continuousShape := shapes.MakeDynamic(ModelDType, []int{shapes.DynamicDim, numContinuousFeatures}, []string{"batch", ""})
		weightShape := shapes.MakeDynamic(ModelDType, []int{shapes.DynamicDim, 1}, []string{"batch", ""})
		err = onnx.SaveToFile(backend, exec, onnxPath,
			[]shapes.Shape{categoricalShape, continuousShape, weightShape},
			[]string{"categorical", "continuous", "weight"},
			[]string{"is_>50K_logit"})
		if err != nil {
			return err
		}
	}

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
			return errors.WithMessagef(err, "failed to execute model with dynamic batch shape")
		}
		klog.Infof("Output: shape=%s, value=%s\n", outputs[0].Shape(), outputs[0])
	}

	// If it was not an ONNX backend, we cannot save an ONNX model.
	if !onnx.IsONNX(backend) {
		return errors.Errorf("backend %q (%T) is not an \"onnx\" backend (*onnxbackend.Backend) required to generate a .onnx model", backend.Name(), backend)
	}
	return nil
}
