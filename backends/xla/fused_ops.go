// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/compute"
	"github.com/pkg/errors"
)

// Fused ops are not supported by the XLA backend — they are decomposed into
// primitives at the graph layer. These stubs satisfy the FusedOps interface.

func (f *Function) FusedSoftmax(x compute.Value, axis int) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedSoftmax not implemented in XLA backend")
}

func (f *Function) FusedGelu(x compute.Value, exact bool) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedGelu not implemented in XLA backend")
}

func (f *Function) FusedLayerNorm(x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedLayerNorm not implemented in XLA backend")
}

func (f *Function) FusedDense(x, weight, bias compute.Value, activation compute.ActivationType) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense not implemented in XLA backend")
}

func (f *Function) FusedScaledDotProductAttention(query, key, value, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedScaledDotProductAttention not implemented in XLA backend")
}

func (f *Function) FusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV compute.Value, queryDim, keyValueDim int) (query, key, value compute.Value, err error) {
	err = errors.Wrapf(compute.ErrNotImplemented, "FusedAttentionQKVProjection not implemented in XLA backend")
	return
}

func (f *Function) FusedQuantizedDense(x, weights, bias compute.Value, weightsQuantization *compute.Quantization, activation compute.ActivationType) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense not implemented in XLA backend")
}

func (f *Function) QuantizedEmbeddingLookup(data, indices compute.Value, dataQuantization *compute.Quantization) (compute.Value, error) {
	return nil, errors.Wrapf(compute.ErrNotImplemented, "QuantizedEmbeddingLookup not implemented in XLA backend")
}
