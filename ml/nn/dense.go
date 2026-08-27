// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// Dense performs a dense (linear) transformation with optional activation:
//
//	y = activation(x @ weight + bias) (if weightLayout is DenseLayoutInputOutputs)
//	y = activation(x @ weight^T + bias) (if weightLayout is DenseLayoutOutputsInput)
//
// weight has shape [in_features, out_features...] for DenseLayoutInputOutputs, or
// [out_features..., in_features] for DenseLayoutOutputsInput.
// bias is optional (nil means no bias).
//
// optionalActivation is optional; if omitted or activations.TypeNone, no activation is
// applied.
//
// If the backend supports fused Dense, the optimized native implementation is
// used; otherwise the operation is decomposed into primitives. Fallback is
// handled automatically via InternalFusedOpCaller.
func Dense(x, weight, bias *Node, weightLayout compute.DenseLayout, optionalActivation ...activation.Type) *Node {
	act := activation.TypeNone
	if len(optionalActivation) > 0 {
		if len(optionalActivation) > 1 {
			Panicf("nn.Dense() can only take one optional activation, got %v", optionalActivation)
		}
		act = optionalActivation[0]
	}

	decomposed := func() *Node {
		return denseDecomposed(x, weight, bias, weightLayout, act)
	}

	denseCfg := compute.DenseConfig{
		Activation:   act.ToBackend(),
		WeightLayout: weightLayout,
	}
	res, _ := InternalFusedOpCaller(
		func() *Node { return BackendFusedDense(x, weight, bias, denseCfg) },
		decomposed,
		true,
	)
	return res
}

// denseDecomposed implements Dense using primitive graph ops.
func denseDecomposed(x, weight, bias *Node, weightLayout compute.DenseLayout, act activation.Type) *Node {
	xShape := x.Shape()
	wShape := weight.Shape()
	xRank := xShape.Rank()
	wRank := wShape.Rank()

	// Flatten x to 2D [batchSize, inFeatures] if needed.
	inFeatures := xShape.Dimensions[xRank-1]
	x2d := x
	if xRank > 2 {
		g := x.Graph()
		x2d = DynamicReshape(x, NamedInferredDim(g.UniqueName("dense_x_batch")), StaticDim(inFeatures))
	}

	var y2d *Node
	var outDims []int

	switch weightLayout {
	case compute.DenseLayoutInputOutputs:
		// Weight shape: [inFeatures, outFeatures...]
		outFeaturesFlat := wShape.Size() / wShape.Dimensions[0]
		w2d := weight
		if wRank > 2 {
			w2d = Reshape(weight, wShape.Dimensions[0], outFeaturesFlat)
		}
		y2d = DotProduct(x2d, w2d)
		outDims = make([]int, xRank-1+wRank-1)
		copy(outDims, xShape.Dimensions[:xRank-1])
		copy(outDims[xRank-1:], wShape.Dimensions[1:])

	case compute.DenseLayoutOutputsInput:
		// Weight shape: [outFeatures..., inFeatures]
		weightLastAxis := wRank - 1
		outFeaturesFlat := wShape.Size() / wShape.Dimensions[weightLastAxis]
		w2d := weight
		if wRank > 2 {
			w2d = Reshape(weight, outFeaturesFlat, wShape.Dimensions[weightLastAxis])
		}
		y2d = Dot(x2d, w2d).General([]int{1}, nil, []int{1}, nil)
		outDims = make([]int, xRank-1+wRank-1)
		copy(outDims, xShape.Dimensions[:xRank-1])
		copy(outDims[xRank-1:], wShape.Dimensions[:weightLastAxis])

	default:
		Panicf("nn.Dense(): unknown WeightLayout %v", weightLayout)
	}

	// Reshape output to [x_batch_dims..., weight_out_dims...] if needed.
	var y *Node
	if xRank <= 2 && wRank <= 2 {
		y = y2d
	} else {
		// Build output shape: x batch dimensions + weight output dimensions
		outAxisSpecs := make([]DimensionSpec, len(outDims))
		for i := range xRank - 1 {
			dim := xShape.Dimensions[i]
			name := xShape.AxisName(i)
			if dim >= 0 {
				outAxisSpecs[i] = StaticDim(dim)
			} else if name != "" {
				outAxisSpecs[i] = NamedDynamicDim(name, DynamicDimensionSize(x, i))
			} else {
				outAxisSpecs[i] = DynamicDim(DynamicDimensionSize(x, i))
			}
		}
		offset := xRank - 1
		for j := range wRank - 1 {
			wAxis := j + 1
			if weightLayout == compute.DenseLayoutOutputsInput {
				wAxis = j
			}
			dim := wShape.Dimensions[wAxis]
			name := wShape.AxisName(wAxis)
			if dim >= 0 {
				outAxisSpecs[offset+j] = StaticDim(dim)
			} else if name != "" {
				outAxisSpecs[offset+j] = NamedDynamicDim(name, DynamicDimensionSize(weight, wAxis))
			} else {
				outAxisSpecs[offset+j] = DynamicDim(DynamicDimensionSize(weight, wAxis))
			}
		}
		y = DynamicReshape(y2d, outAxisSpecs...)
	}
	if bias != nil {
		y = Add(y, ExpandLeftToRank(bias, y.Rank()))
	}
	if act != activation.TypeNone {
		y = activation.Apply(act, y)
	}
	return y
}
