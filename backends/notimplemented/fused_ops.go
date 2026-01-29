// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package notimplemented

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// Compile-time check that Function implements FusedOps.
var _ backends.FusedOps = Function{}

// Softmax returns NotImplementedError.
func (f Function) Softmax(x backends.Value, axis int) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Softmax()")
}

// LayerNorm returns NotImplementedError.
func (f Function) LayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in LayerNorm()")
}

// Einsum returns NotImplementedError.
func (f Function) Einsum(equation string, operands ...backends.Value) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Einsum()")
}

// Gelu returns NotImplementedError.
func (f Function) Gelu(x backends.Value, mode string) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Gelu()")
}

// Linear returns NotImplementedError.
func (f Function) Linear(x, weight, bias backends.Value) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Linear()")
}

// LinearActivation returns NotImplementedError.
func (f Function) LinearActivation(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in LinearActivation()")
}

// ScaledDotProductAttention returns NotImplementedError.
func (f Function) ScaledDotProductAttention(q, k, v, mask backends.Value, scale float64) (backends.Value, error) {
	return nil, errors.Wrapf(NotImplementedError, "in ScaledDotProductAttention()")
}
