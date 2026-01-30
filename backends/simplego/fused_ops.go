// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Node data types for fused ops.

type nodeFusedSoftmax struct {
	axis int
}

func (d *nodeFusedSoftmax) EqualNodeData(other nodeDataComparable) bool {
	return d.axis == other.(*nodeFusedSoftmax).axis
}

type nodeFusedLayerNorm struct {
	axes    []int
	epsilon float64
}

func (d *nodeFusedLayerNorm) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedLayerNorm)
	if d.epsilon != o.epsilon || len(d.axes) != len(o.axes) {
		return false
	}
	for i, a := range d.axes {
		if a != o.axes[i] {
			return false
		}
	}
	return true
}

type nodeFusedGelu struct {
	mode string
}

func (d *nodeFusedGelu) EqualNodeData(other nodeDataComparable) bool {
	return d.mode == other.(*nodeFusedGelu).mode
}

type nodeFusedDenseActivation struct {
	activation backends.ActivationType
}

func (d *nodeFusedDenseActivation) EqualNodeData(other nodeDataComparable) bool {
	return d.activation == other.(*nodeFusedDenseActivation).activation
}

// Softmax computes softmax along the specified axis.
func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("Softmax", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	// Normalize negative axis.
	rank := xNode.shape.Rank()
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("Softmax: axis %d out of range for rank %d", axis, rank)
	}

	// Output shape is the same as input shape.
	data := &nodeFusedSoftmax{axis: axis}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedSoftmax, xNode.shape.Clone(), []*Node{xNode}, data)
	return node, nil
}

// LayerNorm applies layer normalization.
func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	values := []backends.Value{x}
	if gamma != nil {
		values = append(values, gamma)
	}
	if beta != nil {
		values = append(values, beta)
	}
	inputs, err := f.verifyAndCastValues("LayerNorm", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	// Normalize negative axes.
	rank := xNode.shape.Rank()
	normalizedAxes := make([]int, len(axes))
	for i, a := range axes {
		if a < 0 {
			a += rank
		}
		if a < 0 || a >= rank {
			return nil, errors.Errorf("LayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normalizedAxes[i] = a
	}

	data := &nodeFusedLayerNorm{axes: normalizedAxes, epsilon: epsilon}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedLayerNorm, xNode.shape.Clone(), inputs, data)
	return node, nil
}

// Gelu computes Gaussian Error Linear Unit activation.
func (f *Function) FusedGelu(x backends.Value, mode string) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("Gelu", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	data := &nodeFusedGelu{mode: mode}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedGelu, xNode.shape.Clone(), []*Node{xNode}, data)
	return node, nil
}

// Dense performs fused matmul + bias: y = x @ W + bias.
// x: [..., in_features], weight: [in_features, out_features...], bias: [out_features...] (optional)
func (f *Function) FusedDense(x, weight, bias backends.Value) (backends.Value, error) {
	values := []backends.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("Dense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	// Output shape: x's batch dims + weight's remaining dims (out_features...).
	// x: [..., in_features], weight: [in_features, out_features...]
	// output: [..., out_features...]
	if xNode.shape.Rank() < 1 || wNode.shape.Rank() < 2 {
		return nil, errors.Errorf("Dense: x must have rank >= 1 (got %d), weight must have rank >= 2 (got %d)",
			xNode.shape.Rank(), wNode.shape.Rank())
	}
	inFeatures := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if inFeatures != wNode.shape.Dimensions[0] {
		return nil, errors.Errorf("Dense: x's last dim (%d) must match weight's first dim (%d)",
			inFeatures, wNode.shape.Dimensions[0])
	}

	outDims := make([]int, xNode.shape.Rank()-1+wNode.shape.Rank()-1)
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	copy(outDims[xNode.shape.Rank()-1:], wNode.shape.Dimensions[1:])
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	node, _ := f.getOrCreateNode(backends.OpTypeFusedDense, outShape, inputs, nil)
	return node, nil
}

// DenseActivation performs Dense followed by activation in one op.
func (f *Function) FusedDenseActivation(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	values := []backends.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("DenseActivation", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.shape.Rank() < 1 || wNode.shape.Rank() < 2 {
		return nil, errors.Errorf("DenseActivation: x must have rank >= 1 (got %d), weight must have rank >= 2 (got %d)",
			xNode.shape.Rank(), wNode.shape.Rank())
	}
	inFeatures := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if inFeatures != wNode.shape.Dimensions[0] {
		return nil, errors.Errorf("DenseActivation: x's last dim (%d) must match weight's first dim (%d)",
			inFeatures, wNode.shape.Dimensions[0])
	}

	outDims := make([]int, xNode.shape.Rank()-1+wNode.shape.Rank()-1)
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	copy(outDims[xNode.shape.Rank()-1:], wNode.shape.Dimensions[1:])
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	data := &nodeFusedDenseActivation{activation: activation}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedDenseActivation, outShape, inputs, data)
	return node, nil
}
