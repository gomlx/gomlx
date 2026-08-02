// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !onnx

package main

import (
	"flag"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

var flagSaveONNX = flag.String("save_onnx", "", "Save model to ONNX format (requires -tags=onnx build tag).")

func handleSaveONNX(backend compute.Backend, store *model.Store) (bool, error) {
	if *flagSaveONNX != "" {
		return true, errors.Errorf("saving to ONNX requires building with -tags=onnx")
	}
	return false, nil
}
