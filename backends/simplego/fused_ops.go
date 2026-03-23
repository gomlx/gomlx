// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
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
	options    *backends.ScaledDotProductAttentionConfig
}

func (d *nodeFusedScaledDotProductAttention) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal &&
		d.equalOptions(o)
}

func (d *nodeFusedScaledDotProductAttention) equalOptions(o *nodeFusedScaledDotProductAttention) bool {
	if d.options == nil && o.options == nil {
		return true
	}
	if d.options == nil || o.options == nil {
		return false
	}
	return *d.options == *o.options
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

// nodeFusedQuantizedDense stores parameters for the quantized dense op.
//
// For the QuantGGML scheme, weights are stored in native GGML block format
// (see exec_fused_quantized_ggml.go for format details). The weight tensor is
// [N, bytesPerRow] Uint8, where N is the output-features dimension and
// bytesPerRow = (K / valuesPerBlock) * bytesPerBlock. K (input-features) is
// derived from bytesPerRow at build time via deriveGGMLK.
type nodeFusedQuantizedDense struct {
	scheme       backends.QuantizationScheme
	blockAxis    int // Always 1 (output-features axis); validated in builder. Stored for EqualNodeData.
	blockSize    int
	activation   backends.ActivationType
	hasZeroPoint bool
	hasBias      bool
	ggmlType     backends.GGMLQuantType // Only used when scheme == QuantGGML.
	ggmlN        int                    // Output features (rows in GGML layout). Only for QuantGGML.
	ggmlK        int                    // Input features (logical columns). Only for QuantGGML.
}

func (d *nodeFusedQuantizedDense) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDense)
	return d.scheme == o.scheme && d.blockAxis == o.blockAxis &&
		d.blockSize == o.blockSize && d.activation == o.activation &&
		d.hasZeroPoint == o.hasZeroPoint && d.hasBias == o.hasBias &&
		d.ggmlType == o.ggmlType && d.ggmlN == o.ggmlN && d.ggmlK == o.ggmlK
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
func (f *Function) FusedScaledDotProductAttention(query, key, value, mask backends.Value, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool, options *backends.ScaledDotProductAttentionConfig) (backends.Value, error) {
	return f.buildSDPANode(backends.OpTypeFusedScaledDotProductAttention, "FusedScaledDotProductAttention",
		query, key, value, mask, numHeads, numKVHeads, axesLayout, scale, causal, options)
}

// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
//
// Unlike FusedDense, this does not create a DotGeneral sub-node — the quantized matmul
// is fundamentally different (mixed-dtype with per-group scales). The inputs to the
// executor are [x, weights, scales, zeroPoints?, bias?] directly.
//
// Weights should have their dtype set to reflect the actual storage type (e.g. Int4, Int8).
// For sub-byte types, the caller should Bitcast packed byte data to the correct dtype.
func (f *Function) FusedQuantizedDense(x, weights, bias backends.Value,
	weightsQuantization *backends.Quantization,
	activation backends.ActivationType) (backends.Value, error) {

	scheme := weightsQuantization.Scheme

	// GGML weights have scales embedded in their native block format.
	// The weight layout is [N, bytesPerRow] Uint8 instead of [K, N].
	if scheme == backends.QuantGGML {
		return f.fusedQuantizedDenseGGML(x, weights, bias, weightsQuantization, activation)
	}

	scales := weightsQuantization.Scale
	zeroPoints := weightsQuantization.ZeroPoint
	blockAxis := weightsQuantization.BlockAxis
	blockSize := weightsQuantization.BlockSize

	values := []backends.Value{x, weights, scales}
	if zeroPoints != nil {
		values = append(values, zeroPoints)
	}
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

	// Validate x dtype: only float32 is supported.
	if xNode.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("FusedQuantizedDense: x must be float32, got %s", xNode.shape.DType)
	}

	// Validate x shape: [batch..., K]
	if xNode.shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense: x must have rank >= 1, got %d", xNode.shape.Rank())
	}
	K := xNode.shape.Dimensions[xNode.shape.Rank()-1]

	// Derive N from weights shape. The weights dtype reflects the storage type.
	if wNode.shape.Rank() != 2 || wNode.shape.Dimensions[0] != K {
		return nil, errors.Errorf("FusedQuantizedDense: weights must be [%d, N], got %v", K, wNode.shape.Dimensions)
	}
	N := wNode.shape.Dimensions[1]

	// Validate scales shape: [K, numBlocks]
	numBlocks := (N + blockSize - 1) / blockSize
	if sNode.shape.Rank() != 2 || sNode.shape.Dimensions[0] != K || sNode.shape.Dimensions[1] != numBlocks {
		return nil, errors.Errorf("FusedQuantizedDense: scales must be [%d, %d], got %v",
			K, numBlocks, sNode.shape.Dimensions)
	}

	// Output shape: [batch..., N]
	outDims := make([]int, xNode.shape.Rank())
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	outDims[xNode.shape.Rank()-1] = N
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	// Only blockAxis=1 (output-features axis) is currently supported.
	if blockAxis != 1 {
		return nil, errors.Errorf("FusedQuantizedDense: only Axis=1 is supported, got %d", blockAxis)
	}

	// NF4 quantization uses a fixed lookup table and does not support zero points.
	if scheme == backends.QuantNF4 && zeroPoints != nil {
		return nil, errors.Errorf("FusedQuantizedDense: ZeroPoint must be nil for NF4 quantization scheme")
	}

	data := &nodeFusedQuantizedDense{
		scheme:       scheme,
		blockAxis:    blockAxis,
		blockSize:    blockSize,
		activation:   activation,
		hasZeroPoint: zeroPoints != nil,
		hasBias:      bias != nil,
	}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedQuantizedDense, outShape, inputs, data)
	return node, nil
}

// validateGGMLTypeSupported checks that the given GGML type has a fused executor implementation.
func validateGGMLTypeSupported(opName string, ggmlType backends.GGMLQuantType) error {
	switch ggmlType {
	case backends.GGMLQ4_0, backends.GGMLQ8_0, backends.GGMLIQ4NL, backends.GGMLQ4_K, backends.GGMLQ6_K:
		return nil
	default:
		return errors.Wrapf(backends.ErrNotImplemented, "%s: GGML type %s not supported in fused path", opName, ggmlType)
	}
}

// deriveGGMLK computes the logical input-features dimension K from bytesPerRow
// and the GGML block format. GGML weights are stored as [N, bytesPerRow] Uint8,
// where each row consists of consecutive quantization blocks. Each block packs
// valuesPerBlock logical float32 values into bytesPerBlock bytes (see
// exec_fused_quantized_ggml.go for block layouts). K is therefore:
//
//	K = (bytesPerRow / bytesPerBlock) * valuesPerBlock
//
// This function validates that bytesPerRow is an exact multiple of bytesPerBlock.
func deriveGGMLK(opName string, bytesPerRow int, ggmlType backends.GGMLQuantType) (int, error) {
	vpb := ggmlType.ValuesPerBlock()
	bpb := ggmlType.BytesPerBlock()
	if vpb == 0 || bpb == 0 {
		return 0, errors.Errorf("%s: unsupported GGML type %s", opName, ggmlType)
	}
	if bytesPerRow%bpb != 0 {
		return 0, errors.Errorf("%s: bytesPerRow %d not divisible by bytesPerBlock %d for %s",
			opName, bytesPerRow, bpb, ggmlType)
	}
	return (bytesPerRow / bpb) * vpb, nil
}

// fusedQuantizedDenseGGML handles the GGML path for FusedQuantizedDense.
// GGML weights are [N, bytesPerRow] Uint8 with native block layout.
// Scales and zero points are embedded in the blocks.
func (f *Function) fusedQuantizedDenseGGML(x, weights, bias backends.Value,
	wq *backends.Quantization,
	activation backends.ActivationType) (backends.Value, error) {

	values := []backends.Value{x, weights}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("FusedQuantizedDense(GGML)", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	// Validate x dtype: only float32 is supported.
	if xNode.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x must be float32, got %s", xNode.shape.DType)
	}
	if xNode.shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x must have rank >= 1, got %d", xNode.shape.Rank())
	}

	// GGML weights: [N, bytesPerRow] Uint8.
	if wNode.shape.Rank() != 2 || wNode.shape.DType != dtypes.Uint8 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): weights must be [N, bytesPerRow] Uint8, got %v %s",
			wNode.shape.Dimensions, wNode.shape.DType)
	}
	N := wNode.shape.Dimensions[0]
	bytesPerRow := wNode.shape.Dimensions[1]

	ggmlType := wq.GGMLType
	if err := validateGGMLTypeSupported("FusedQuantizedDense(GGML)", ggmlType); err != nil {
		return nil, err
	}
	K, err := deriveGGMLK("FusedQuantizedDense(GGML)", bytesPerRow, ggmlType)
	if err != nil {
		return nil, err
	}

	// Validate that x's last dimension matches K.
	xK := xNode.shape.Dimensions[xNode.shape.Rank()-1]
	if xK != K {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x's last dim (%d) must match K=%d derived from weights", xK, K)
	}

	// Zero points are not supported for GGML (embedded in blocks).
	if wq.ZeroPoint != nil {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): ZeroPoint must be nil for GGML scheme")
	}

	// Output shape: [batch..., N]
	outDims := make([]int, xNode.shape.Rank())
	copy(outDims, xNode.shape.Dimensions[:xNode.shape.Rank()-1])
	outDims[xNode.shape.Rank()-1] = N
	outShape := shapes.Make(xNode.shape.DType, outDims...)

	data := &nodeFusedQuantizedDense{
		scheme:     backends.QuantGGML,
		activation: activation,
		hasBias:    bias != nil,
		ggmlType:   ggmlType,
		ggmlN:      N,
		ggmlK:      K,
	}
	node, _ := f.getOrCreateNode(backends.OpTypeFusedQuantizedDense, outShape, inputs, data)
	return node, nil
}

// nodeQuantizedEmbeddingLookup stores parameters for the quantized embedding lookup op.
type nodeQuantizedEmbeddingLookup struct {
	ggmlType backends.GGMLQuantType
	ggmlK    int // Logical embedding dimension (valuesPerBlock * numBlocks).
}

func (d *nodeQuantizedEmbeddingLookup) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeQuantizedEmbeddingLookup)
	return d.ggmlType == o.ggmlType && d.ggmlK == o.ggmlK
}

// QuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
// with on-the-fly dequantization.
// data: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
// indices: integer tensor with last dim = 1 (same as Gather convention).
// Output: [batch..., K] Float32 where K is derived from the block format.
func (f *Function) QuantizedEmbeddingLookup(data, indices backends.Value,
	wq *backends.Quantization) (backends.Value, error) {

	if wq.Scheme != backends.QuantGGML {
		return nil, errors.Wrapf(backends.ErrNotImplemented,
			"QuantizedEmbeddingLookup: only QuantGGML scheme is supported, got %s -- "+
				"please create a feature request if you need support for a different quantization scheme", wq.Scheme)
	}

	inputs, err := f.verifyAndCastValues("QuantizedEmbeddingLookup", data, indices)
	if err != nil {
		return nil, err
	}
	dNode := inputs[0]
	iNode := inputs[1]

	// Validate data: [vocabSize, bytesPerRow] Uint8.
	if dNode.shape.Rank() != 2 || dNode.shape.DType != dtypes.Uint8 {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: data must be [vocabSize, bytesPerRow] Uint8, got %v %s",
			dNode.shape.Dimensions, dNode.shape.DType)
	}
	bytesPerRow := dNode.shape.Dimensions[1]

	// Validate indices: must be integer, last dim = 1.
	if !iNode.shape.DType.IsInt() {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: indices must be integer, got %s", iNode.shape.DType)
	}
	if iNode.shape.Rank() < 1 || iNode.shape.Dimensions[iNode.shape.Rank()-1] != 1 {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: indices last dim must be 1, got shape %v", iNode.shape.Dimensions)
	}

	ggmlType := wq.GGMLType
	if err := validateGGMLTypeSupported("QuantizedEmbeddingLookup", ggmlType); err != nil {
		return nil, err
	}

	// Derive K from bytesPerRow.
	K, err := deriveGGMLK("QuantizedEmbeddingLookup", bytesPerRow, ggmlType)
	if err != nil {
		return nil, err
	}

	// Output shape: [batch..., K] Float32.
	// indices shape is [batch..., 1]. Output replaces the last dim with K.
	outDims := make([]int, iNode.shape.Rank())
	copy(outDims, iNode.shape.Dimensions)
	outDims[len(outDims)-1] = K
	outShape := shapes.Make(dtypes.Float32, outDims...)

	nodeData := &nodeQuantizedEmbeddingLookup{
		ggmlType: ggmlType,
		ggmlK:    K,
	}
	node, _ := f.getOrCreateNode(backends.OpTypeQuantizedEmbeddingLookup, outShape, inputs, nodeData)
	return node, nil
}

// buildSDPANode builds the SDPA computation node.
func (f *Function) buildSDPANode(opType backends.OpType, opName string,
	query, key, value, mask backends.Value,
	numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool, options *backends.ScaledDotProductAttentionConfig,
) (backends.Value, error) {
	values := []backends.Value{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}
	inputs, err := f.verifyAndCastValues(opName, values...)
	if err != nil {
		return nil, err
	}
	qNode := inputs[0]

	if qNode.shape.Rank() != 4 {
		return nil, errors.Errorf("%s: query must have rank 4, got %d", opName, qNode.shape.Rank())
	}
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, errors.Errorf("%s: numHeads (%d) must be positive and divisible by numKVHeads (%d)", opName, numHeads, numKVHeads)
	}

	data := &nodeFusedScaledDotProductAttention{numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout, scale: scale, causal: causal, options: options}
	node, _ := f.getOrCreateNode(opType, qNode.shape.Clone(), inputs, data)
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
	// Bias nodes are at inputs[2:] in the same order they were appended.
	fusedInputs := append([]*Node{dotNode}, inputs[2:]...)

	data := &nodeFusedAttentionQKVProjection{qDim: queryDim, kvDim: keyValueDim, hasBiasQ: biasQ != nil, hasBiasK: biasK != nil, hasBiasV: biasV != nil}
	node := f.newMultiOutputsNode(backends.OpTypeFusedAttentionQKVProjection, []shapes.Shape{qShape, kShape, vShape}, fusedInputs...)
	node.data = data
	queryOut = node.multiOutputsNodes[0]
	keyOut = node.multiOutputsNodes[1]
	valueOut = node.multiOutputsNodes[2]
	return
}
