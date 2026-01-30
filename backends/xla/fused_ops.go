// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// Fused ops are not supported by the XLA backend — they are decomposed into
// primitives at the graph layer. These stubs satisfy the StandardOps interface.

func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	return nil, errors.New("FusedSoftmax not implemented in XLA backend")
}

func (f *Function) FusedGelu(x backends.Value, mode string) (backends.Value, error) {
	return nil, errors.New("FusedGelu not implemented in XLA backend")
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	return nil, errors.New("FusedLayerNorm not implemented in XLA backend")
}

func (f *Function) FusedDense(x, weight, bias backends.Value) (backends.Value, error) {
	return nil, errors.New("FusedDense not implemented in XLA backend")
}

func (f *Function) FusedDenseActivation(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.New("FusedDenseActivation not implemented in XLA backend")
}
