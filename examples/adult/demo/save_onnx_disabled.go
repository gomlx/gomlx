// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !onnx

package main

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// saveONNX saves the model with the weights in store to onnxPath.
// Not implemented if -tags=onnx is not set.
func saveONNX(backend compute.Backend, store *model.Store, onnxPath string) error {
	return errors.Errorf("saving to ONNX requires building with -tags=onnx")
}
