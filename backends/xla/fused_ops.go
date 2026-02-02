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

func (f *Function) FusedMultiHeadSDPA(q, k, v, mask backends.Value, numHeads, numKVHeads int, scale float64, causal bool) (backends.Value, error) {
	return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedMultiHeadSDPA not implemented in XLA backend")
}

func (f *Function) FusedQKVDense(x, wQKV, biasQ, biasK, biasV backends.Value, qDim, kvDim int) (q, k, v backends.Value, err error) {
	err = errors.Wrapf(backends.ErrNotImplemented, "FusedQKVDense not implemented in XLA backend")
	return
}
