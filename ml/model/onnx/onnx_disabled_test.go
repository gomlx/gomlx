// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !onnx

package onnx_test

import (
	"bytes"
	"testing"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/ml/model/onnx"
	"github.com/stretchr/testify/assert"
)

func TestONNXDisabled(t *testing.T) {
	err := onnx.Save(nil, nil, &bytes.Buffer{}, []shapes.Shape{}, nil, nil)
	assert.ErrorIs(t, err, onnx.ErrNotImplemented)

	err = onnx.SaveToFile(nil, nil, "model.onnx", []shapes.Shape{}, nil, nil)
	assert.ErrorIs(t, err, onnx.ErrNotImplemented)

	_, err = onnx.Load(nil, &bytes.Buffer{})
	assert.ErrorIs(t, err, onnx.ErrNotImplemented)

	_, err = onnx.LoadFromFile(nil, "model.onnx")
	assert.ErrorIs(t, err, onnx.ErrNotImplemented)

	exec := &onnx.Executable{}
	_, err = exec.Call(nil)
	assert.ErrorIs(t, err, onnx.ErrNotImplemented)
}
