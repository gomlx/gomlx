// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// Fused ops are not supported by the XLA backend — they are decomposed into
// primitives at the graph layer. These stubs satisfy the FusedOps interface.

func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedSoftmax not implemented in XLA backend")
}

func (f *Function) FusedGelu(x backends.Value, exact bool) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedGelu not implemented in XLA backend")
}

func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedLayerNorm not implemented in XLA backend")
}

func (f *Function) FusedDense(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedDense not implemented in XLA backend")
}

func (f *Function) FusedScaledDotProductAttention(query, key, value, mask backends.Value, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedScaledDotProductAttention not implemented in XLA backend")
}

func (f *Function) FusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV backends.Value, queryDim, keyValueDim int) (query, key, value backends.Value, err error) {
	err = errors.Wrapf(backends.ErrNotImplemented, "FusedAttentionQKVProjection not implemented in XLA backend")
	return
}

func (f *Function) FusedQuantizedDense(x, packedWeights, scales, bias backends.Value, quantFormat backends.QuantFormat, groupSize int, outFeatures int, activation backends.ActivationType) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedDense not implemented in XLA backend")
}

func (f *Function) FusedQuantizedScaledDotProductAttention(query, key, value, mask backends.Value, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedScaledDotProductAttention not implemented in XLA backend")
}
