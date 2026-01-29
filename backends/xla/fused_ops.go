// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// Fused ops are not supported by the XLA backend — they are decomposed into
// primitives at the graph layer. These stubs satisfy the StandardOps interface.

func (f *Function) Softmax(x backends.Value, axis int) (backends.Value, error) {
	return nil, errors.New("Softmax not implemented in XLA backend")
}

func (f *Function) Gelu(x backends.Value, mode string) (backends.Value, error) {
	return nil, errors.New("Gelu not implemented in XLA backend")
}

func (f *Function) LayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	return nil, errors.New("LayerNorm not implemented in XLA backend")
}

func (f *Function) Linear(x, weight, bias backends.Value) (backends.Value, error) {
	return nil, errors.New("Linear not implemented in XLA backend")
}

func (f *Function) LinearActivation(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.New("LinearActivation not implemented in XLA backend")
}
