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

type nodeFusedScaledDotProductAttention struct {
	numHeads   int
	numKVHeads int
	axesLayout backends.AxesLayout
	scale      float64
	causal     bool
}

func (d *nodeFusedScaledDotProductAttention) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal
}

// nodeFusedAttentionQKVProjection stores parameters for the fused QKV projection.
// It does not implement nodeDataComparable because multi-output nodes are not
// de-duplicated (see newMultiOutputsNode).
type nodeFusedAttentionQKVProjection struct {
	qDim     int
	kvDim    int
	hasBiasQ bool
	hasBiasK bool
	hasBiasV bool
}

type nodeFusedQuantizedDense struct {
	quantFormat backends.QuantFormat
	groupSize   int
	outFeatures int
	activation  backends.ActivationType
}

func (d *nodeFusedQuantizedDense) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDense)
	return d.quantFormat == o.quantFormat && d.groupSize == o.groupSize &&
		d.outFeatures == o.outFeatures && d.activation == o.activation
}

type nodeFusedQuantizedScaledDotProductAttention struct {
	numHeads   int
	numKVHeads int
	axesLayout backends.AxesLayout
	scale      float64
	causal     bool
}

func (d *nodeFusedQuantizedScaledDotProductAttention) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal
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
	dotResult, err := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{0}, nil, backends.DotGeneralConfig{})
	if err != nil {
		return nil, errors.WithMessagef(err, "FusedDense: DotGeneral")
	}
	dotNode := dotResult.(*Node)

	// FusedDense inputs: [dotResult, x, weight, bias?].
	// The matmul is already computed by the DotGeneral sub-node (inputs[0]).
	// x and weight are included so that SIMD-accelerated executors (highway) can
	// redo the fused matmul+bias+activation from scratch.
	fusedInputs := []*Node{dotNode, xNode, wNode}
	if len(inputs) > 2 {
		fusedInputs = append(fusedInputs, inputs[2])
	}

	data := &nodeFusedDense{activation: activation}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedDense, outShape, fusedInputs, data)
	return node, nil
}

// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
// Both AxesLayoutBHSD and AxesLayoutBSHD are supported; the executor transposes
// BSHD inputs to BHSD internally.
func (f *Function) FusedScaledDotProductAttention(query, key, value, mask backends.Value, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) (backends.Value, error) {
	values := []backends.Value{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}
	inputs, err := f.verifyAndCastValues("FusedScaledDotProductAttention", values...)
	if err != nil {
		return nil, err
	}
	qNode := inputs[0]

	// Validate shapes: query [batch, numHeads, seqLen, headDim]
	if qNode.shape.Rank() != 4 {
		return nil, errors.Errorf("FusedScaledDotProductAttention: query must have rank 4, got %d", qNode.shape.Rank())
	}
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, errors.Errorf("FusedScaledDotProductAttention: numHeads (%d) must be positive and divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}

	// Output shape is the same as query.
	data := &nodeFusedScaledDotProductAttention{numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout, scale: scale, causal: causal}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedScaledDotProductAttention, qNode.shape.Clone(), inputs, data)
	return node, nil
}

// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
//
// Unlike FusedDense, this does not create a DotGeneral sub-node — the quantized matmul
// is fundamentally different (mixed-dtype with per-group scales). The inputs to the
// executor are [x, packedWeights, scales, bias?] directly.
func (f *Function) FusedQuantizedDense(x, packedWeights, scales, bias backends.Value,
	quantFormat backends.QuantFormat, groupSize int, outFeatures int,
	activation backends.ActivationType) (backends.Value, error) {

	values := []backends.Value{x, packedWeights, scales}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("FusedQuantizedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]
	sNode := inputs[2]

	// Validate x shape: [batch..., K]
	if xNode.shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense: x must have rank >= 1, got %d", xNode.shape.Rank())
	}
	K := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	N := outFeatures

	// Validate packed weight shape based on format.
	switch quantFormat {
	case backends.QuantNF4, backends.QuantInt4:
		// Expected: [K, N/2] uint8
		packedN := (N + 1) / 2
		if wNode.shape.Rank() != 2 || wNode.shape.Dimensions[0] != K || wNode.shape.Dimensions[1] != packedN {
			return nil, errors.Errorf("FusedQuantizedDense: %s packed weights must be [%d, %d], got %v",
				quantFormat, K, packedN, wNode.shape.Dimensions)
		}
	case backends.QuantInt8:
		// Expected: [K, N] int8
		if wNode.shape.Rank() != 2 || wNode.shape.Dimensions[0] != K || wNode.shape.Dimensions[1] != N {
			return nil, errors.Errorf("FusedQuantizedDense: Int8 weights must be [%d, %d], got %v",
				K, N, wNode.shape.Dimensions)
		}
	default:
		return nil, errors.Errorf("FusedQuantizedDense: unknown quant format %d", quantFormat)
	}

	// Validate scales shape: [K, numGroups]
	numGroups := (N + groupSize - 1) / groupSize
	if sNode.shape.Rank() != 2 || sNode.shape.Dimensions[0] != K || sNode.shape.Dimensions[1] != numGroups {
		return nil, errors.Errorf("FusedQuantizedDense: scales must be [%d, %d], got %v",
			K, numGroups, sNode.shape.Dimensions)
	}

	// Output shape: [batch..., N]
	outDims := make([]int, xNode.shape.Rank())
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	outDims[xNode.shape.Rank()-1] = N
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	data := &nodeFusedQuantizedDense{
		quantFormat: quantFormat,
		groupSize:   groupSize,
		outFeatures: outFeatures,
		activation:  activation,
	}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedQuantizedDense, outShape, inputs, data)
	return node, nil
}

// FusedQuantizedScaledDotProductAttention computes multi-head SDPA using int8×int8
// matmuls for Q@K^T and attn@V. Inputs are float32; quantization is internal.
func (f *Function) FusedQuantizedScaledDotProductAttention(query, key, value, mask backends.Value, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool) (backends.Value, error) {
	values := []backends.Value{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}
	inputs, err := f.verifyAndCastValues("FusedQuantizedScaledDotProductAttention", values...)
	if err != nil {
		return nil, err
	}
	qNode := inputs[0]

	if qNode.shape.Rank() != 4 {
		return nil, errors.Errorf("FusedQuantizedScaledDotProductAttention: query must have rank 4, got %d", qNode.shape.Rank())
	}
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, errors.Errorf("FusedQuantizedScaledDotProductAttention: numHeads (%d) must be positive and divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}

	data := &nodeFusedQuantizedScaledDotProductAttention{numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout, scale: scale, causal: causal}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedQuantizedScaledDotProductAttention, qNode.shape.Clone(), inputs, data)
	return node, nil
}

// FusedAttentionQKVProjection performs fused Query-Key-Value projection.
//
// The matmul (x @ wQKV) is delegated to DotGeneral, which selects the optimal
// execution path (blocked, packgemm, highway, etc.) at build time. The fused
// executor then splits the result into Q/K/V and adds biases.
func (f *Function) FusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV backends.Value, queryDim, keyValueDim int) (queryOut, keyOut, valueOut backends.Value, err error) {
	values := []backends.Value{x, wQKV}
	if biasQ != nil {
		values = append(values, biasQ)
	}
	if biasK != nil {
		values = append(values, biasK)
	}
	if biasV != nil {
		values = append(values, biasV)
	}
	inputs, err := f.verifyAndCastValues("AttentionQKVProjection", values...)
	if err != nil {
		return nil, nil, nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.shape.Rank() < 1 {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: x must have rank >= 1, got %d", xNode.shape.Rank())
	}

	batchDims := xNode.shape.Dimensions[:xNode.shape.Rank()-1]
	qDims := make([]int, len(batchDims)+1)
	copy(qDims, batchDims)
	qDims[len(batchDims)] = queryDim
	kvDims := make([]int, len(batchDims)+1)
	copy(kvDims, batchDims)
	kvDims[len(batchDims)] = keyValueDim

	qShape := shapes.Make(xNode.shape.DType, qDims...)
	kShape := shapes.Make(xNode.shape.DType, kvDims...)
	vShape := shapes.Make(xNode.shape.DType, kvDims...)

	// Build DotGeneral sub-node for the matmul: x @ wQKV.
	// This delegates to the optimized matmul infrastructure (blocked, packgemm, highway, etc.).
	dotResult, dotErr := f.DotGeneral(xNode, []int{xNode.shape.Rank() - 1}, nil, wNode, []int{0}, nil, backends.DotGeneralConfig{})
	if dotErr != nil {
		return nil, nil, nil, errors.WithMessagef(dotErr, "FusedAttentionQKVProjection: DotGeneral")
	}
	dotNode := dotResult.(*Node)

	// FusedAttentionQKVProjection inputs: [dotResult, biasQ?, biasK?, biasV?].
	// The matmul is already computed by the DotGeneral sub-node (inputs[0]).
	fusedInputs := []*Node{dotNode}
	if biasQ != nil {
		fusedInputs = append(fusedInputs, inputs[2])
	}
	biasIdx := 2
	if biasQ != nil {
		biasIdx++
	}
	if biasK != nil {
		fusedInputs = append(fusedInputs, inputs[biasIdx])
		biasIdx++
	}
	if biasV != nil {
		fusedInputs = append(fusedInputs, inputs[biasIdx])
	}

	data := &nodeFusedAttentionQKVProjection{qDim: queryDim, kvDim: keyValueDim, hasBiasQ: biasQ != nil, hasBiasK: biasK != nil, hasBiasV: biasV != nil}
	node := f.newMultiOutputsNode(backends.OpTypeFusedAttentionQKVProjection, []shapes.Shape{qShape, kShape, vShape}, fusedInputs...)
	node.data = data
	queryOut = node.multiOutputsNodes[0]
	keyOut = node.multiOutputsNodes[1]
	valueOut = node.multiOutputsNodes[2]
	return
}
