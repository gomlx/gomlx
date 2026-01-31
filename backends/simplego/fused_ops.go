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
	exact bool
}

func (d *nodeFusedGelu) EqualNodeData(other nodeDataComparable) bool {
	return d.exact == other.(*nodeFusedGelu).exact
}

type nodeFusedDense struct {
	activation backends.ActivationType
}

func (d *nodeFusedDense) EqualNodeData(other nodeDataComparable) bool {
	return d.activation == other.(*nodeFusedDense).activation
}

// FusedSoftmax computes softmax along the specified axis.
// The axis must be non-negative (the caller normalizes negative indices).
func (f *Function) FusedSoftmax(x backends.Value, axis int) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("FusedSoftmax", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	rank := xNode.shape.Rank()
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("FusedSoftmax: axis %d out of range for rank %d", axis, rank)
	}

	data := &nodeFusedSoftmax{axis: axis}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedSoftmax, xNode.shape.Clone(), []*Node{xNode}, data)
	return node, nil
}

// FusedLayerNorm applies layer normalization.
func (f *Function) FusedLayerNorm(x backends.Value, axes []int, epsilon float64, gamma, beta backends.Value) (backends.Value, error) {
	values := []backends.Value{x}
	if gamma != nil {
		values = append(values, gamma)
	}
	if beta != nil {
		values = append(values, beta)
	}
	inputs, err := f.verifyAndCastValues("FusedLayerNorm", values...)
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
			return nil, errors.Errorf("FusedLayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normalizedAxes[i] = a
	}

	data := &nodeFusedLayerNorm{axes: normalizedAxes, epsilon: epsilon}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedLayerNorm, xNode.shape.Clone(), inputs, data)
	return node, nil
}

// FusedGelu computes Gaussian Error Linear Unit activation.
// If exact is true, uses the exact GELU (erf); otherwise uses the tanh approximation.
func (f *Function) FusedGelu(x backends.Value, exact bool) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues("FusedGelu", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	data := &nodeFusedGelu{exact: exact}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedGelu, xNode.shape.Clone(), []*Node{xNode}, data)
	return node, nil
}

// FusedDense performs fused matmul + optional bias + optional activation:
//
//	y = activation(x @ W + bias)
//
// The matmul is delegated to DotGeneral (which selects the optimal execution
// path at build time). FusedDense then adds bias and applies activation on top
// of the DotGeneral result.
func (f *Function) FusedDense(x, weight, bias backends.Value, activation backends.ActivationType) (backends.Value, error) {
	values := []backends.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("FusedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.shape.Rank() < 1 || wNode.shape.Rank() < 2 {
		return nil, errors.Errorf("FusedDense: x must have rank >= 1 (got %d), weight must have rank >= 2 (got %d)",
			xNode.shape.Rank(), wNode.shape.Rank())
	}
	inFeatures := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if inFeatures != wNode.shape.Dimensions[0] {
		return nil, errors.Errorf("FusedDense: x's last dim (%d) must match weight's first dim (%d)",
			inFeatures, wNode.shape.Dimensions[0])
	}

	outDims := make([]int, xNode.shape.Rank()-1+wNode.shape.Rank()-1)
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	copy(outDims[xNode.shape.Rank()-1:], wNode.shape.Dimensions[1:])
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	// Build DotGeneral sub-node for the matmul: contract x's last axis with weight's first.
	dotResult, err := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{0}, nil)
	if err != nil {
		return nil, errors.WithMessagef(err, "FusedDense: DotGeneral")
	}
	dotNode := dotResult.(*Node)

	// FusedDense inputs: [dotResult, bias?]. The matmul is already computed by DotGeneral.
	fusedInputs := []*Node{dotNode}
	if len(inputs) > 2 {
		fusedInputs = append(fusedInputs, inputs[2])
	}

	data := &nodeFusedDense{activation: activation}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedDense, outShape, fusedInputs, data)
	return node, nil
}
