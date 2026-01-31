// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// Dense performs a dense (linear) transformation with optional activation:
//   y = activation(x @ weight + bias)
//
// weight has shape [in_features, out_features...]. bias is optional (nil means no
// bias). x's last axis contracts with weight's first axis.
//
// activation is optional; if omitted or activations.TypeNone, no activation is
// applied.
//
// If the backend supports fused Dense (backends.OpTypeFusedDense), the optimized
// native implementation is used; otherwise the operation is decomposed into
// Einsum + Add + activation.
func Dense(x, weight, bias *Node, activation ...activations.Type) *Node {
	act := activations.TypeNone
	if len(activation) > 0 {
		act = activation[0]
	}

	backendAct := activationTypeToBackend(act)

	if x.Graph().Backend().Capabilities().Operations[backends.OpTypeFusedDense] {
		return FusedDense(x, weight, bias, backendAct)
	}

	// Decomposed: contract x's last axis with weight's first axis,
	// producing x @ weight.
	xShape := x.Shape()
	wShape := weight.Shape()
	xRank := xShape.Rank()
	wRank := wShape.Rank()

	// Flatten x to 2D [batchSize, inFeatures] if needed.
	inFeatures := xShape.Dimensions[xRank-1]
	xBatchSize := xShape.Size() / inFeatures
	x2d := x
	if xRank > 2 {
		x2d = Reshape(x, xBatchSize, inFeatures)
	}

	// Flatten weight to 2D [inFeatures, outFeatures] if needed.
	outFeaturesFlat := wShape.Size() / wShape.Dimensions[0]
	w2d := weight
	if wRank > 2 {
		w2d = Reshape(weight, wShape.Dimensions[0], outFeaturesFlat)
	}

	y2d := Dot(x2d, w2d)

	// Reshape output to [x_batch_dims..., weight_out_dims...] if needed.
	var y *Node
	if xRank <= 2 && wRank <= 2 {
		y = y2d
	} else {
		outDims := make([]int, xRank-1+wRank-1)
		copy(outDims, xShape.Dimensions[:xRank-1])
		copy(outDims[xRank-1:], wShape.Dimensions[1:])
		y = Reshape(y2d, outDims...)
	}
	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	if act != activations.TypeNone {
		y = activations.Apply(act, y)
	}
	return y
}

// activationTypeToBackend converts activations.Type to backends.ActivationType.
func activationTypeToBackend(act activations.Type) backends.ActivationType {
	switch act {
	case activations.TypeNone:
		return backends.ActivationNone
	case activations.TypeGelu, activations.TypeGeluApprox:
		return backends.ActivationGelu
	case activations.TypeRelu:
		return backends.ActivationRelu
	case activations.TypeSwish, activations.TypeSilu:
		return backends.ActivationSilu
	case activations.TypeTanh:
		return backends.ActivationTanh
	default:
		return backends.ActivationNone
	}
}
