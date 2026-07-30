// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !onnx

package onnx

import (
	"io"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// ErrNotImplemented is returned when the package is built without the onnx build tag.
var ErrNotImplemented = errors.Errorf("ONNX support disabled: build with -tags=onnx")

// Save returns ErrNotImplemented when -tags=onnx is not set.
func Save(backend compute.Backend, exec *model.Exec, w io.Writer, inputShapes []shapes.Shape, inputNames, outputNames []string) error {
	return ErrNotImplemented
}

// SaveToFile returns ErrNotImplemented when -tags=onnx is not set.
func SaveToFile(backend compute.Backend, exec *model.Exec, filePath string, inputShapes []shapes.Shape, inputNames, outputNames []string) error {
	return ErrNotImplemented
}

// Executable represents a loaded ONNX model executable when ONNX support is disabled.
type Executable struct{}

// Load returns ErrNotImplemented when -tags=onnx is not set.
func Load(backend compute.Backend, r io.Reader) (*Executable, error) {
	return nil, ErrNotImplemented
}

// LoadFromFile returns ErrNotImplemented when -tags=onnx is not set.
func LoadFromFile(backend compute.Backend, filePath string) (*Executable, error) {
	return nil, ErrNotImplemented
}

// Call returns ErrNotImplemented when -tags=onnx is not set.
func (e *Executable) Call(inputs ...any) ([]*tensors.Tensor, error) {
	return nil, ErrNotImplemented
}

// Finalize is a no-op when ONNX support is disabled.
func (e *Executable) Finalize() {}
