// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// Quantization describes how a graph-level quantized tensor is dequantized.
// This is the graph-layer counterpart of backends.Quantization, holding *Node
// references for Scale and ZeroPoint instead of backends.Value.
type Quantization struct {
	// Scheme: Linear (standard) or NF4.
	Scheme backends.QuantizationScheme

	// Scale is the multiplicative factor.
	// Shape: [K, NumBlocks] (block-wise), where K is the input-features
	// (contracting) dimension of the [K, N] weight matrix and
	// NumBlocks = ceil(N / BlockSize).
	Scale *Node

	// ZeroPoint is the additive offset (only for Linear).
	// If nil, the quantization is assumed symmetric.
	ZeroPoint *Node

	// BlockAxis is the dimension of the quantized tensor that is blocked.
	// This is the output-features dimension (axis 1) of a [K, N] weight matrix.
	// Currently only BlockAxis=1 is supported.
	BlockAxis int

	// BlockSize is the number of elements in BlockAxis that share one scale.
	// If BlockSize == values.Shape()[BlockAxis], it's effectively per-axis quantization.
	BlockSize int
}

// toBackend converts the graph-level Quantization to a backends.Quantization
// by extracting the backend Values from the *Node fields.
func (q *Quantization) toBackend() *backends.Quantization {
	bq := &backends.Quantization{
		Scheme:    q.Scheme,
		Scale:     q.Scale.outputOps[0],
		BlockAxis: q.BlockAxis,
		BlockSize: q.BlockSize,
	}
	if q.ZeroPoint != nil {
		bq.ZeroPoint = q.ZeroPoint.outputOps[0]
	}
	return bq
}

// nodeInputsFusedQuantizedDense holds the inputs used for the call to backends.FusedQuantizedDense.
// Hand-written (not generated) because the backend interface uses a *backends.Quantization struct
// but the graph layer stores scale and zeroPoint as individual *Node values.
type nodeInputsFusedQuantizedDense struct {
	x          *Node
	weights    *Node
	bias       *Node
	wq         *Quantization
	activation backends.ActivationType
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) Type() NodeType {
	return NodeTypeFusedQuantizedDense
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) String() string {
	return fmt.Sprintf("%s(x=[#%d], weights=[#%d], bias=%s, scale=[#%d], zeroPoint=%s, scheme=%s, blockAxis=%v, blockSize=%v, activation=%s)",
		ni.Type(),
		ni.x.Id(),
		ni.weights.Id(),
		strNillableNode(ni.bias),
		ni.wq.Scale.Id(),
		strNillableNode(ni.wq.ZeroPoint),
		ni.wq.Scheme,
		ni.wq.BlockAxis,
		ni.wq.BlockSize,
		ni.activation,
	)
}

// BackendFusedSoftmax computes softmax along the specified axis.
// Internal: prefer graph.Softmax which handles fallback and gradients.
func BackendFusedSoftmax(x *Node, axis int) *Node { return backendFusedSoftmax(x, axis) }

// BackendFusedGelu computes GELU activation (exact or approximate).
// Internal: prefer activations.Apply which handles fallback and gradients.
func BackendFusedGelu(x *Node, exact bool) *Node { return backendFusedGelu(x, exact) }

// BackendFusedLayerNorm applies layer normalization. gamma and beta are optional (nil to skip).
// Internal: prefer nn.LayerNorm which handles fallback and gradients.
func BackendFusedLayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	return backendFusedLayerNorm(x, axes, epsilon, gamma, beta)
}

// BackendFusedDense performs fused matmul + optional bias + optional activation.
// Internal: prefer nn.Dense which handles fallback and gradients.
func BackendFusedDense(x, weight, bias *Node, activation backends.ActivationType) *Node {
	return backendFusedDense(x, weight, bias, activation)
}

// BackendFusedScaledDotProductAttention computes multi-head scaled dot-product attention.
// Internal: prefer the InternalFusedOpCaller wrapper in layers which handles fallback and gradients.
func BackendFusedScaledDotProductAttention(query, key, value, mask *Node, numHeads, numKVHeads int, axesLayout backends.AxesLayout, scale float64, causal bool, options *backends.ScaledDotProductAttentionConfig) *Node {
	return backendFusedScaledDotProductAttention(query, key, value, mask, numHeads, numKVHeads, axesLayout, scale, causal, options)
}

// BackendFusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
// Internal: prefer nn.QuantizedDense which handles fallback and gradients.
func BackendFusedQuantizedDense(x, weights, bias *Node,
	wq *Quantization, activation backends.ActivationType) *Node {

	if wq == nil || wq.Scale == nil {
		exceptions.Panicf("BackendFusedQuantizedDense: wq and wq.Scale must not be nil")
	}

	// inputNodes ordering matches the builder: [x, weights, scale, zeroPoint?, bias?].
	inputNodes := []*Node{x, weights, wq.Scale}
	if wq.ZeroPoint != nil {
		inputNodes = append(inputNodes, wq.ZeroPoint)
	}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	inputs := &nodeInputsFusedQuantizedDense{
		x:          x,
		weights:    weights,
		bias:       bias,
		wq:         wq,
		activation: activation,
	}

	var biasVal backends.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.FusedQuantizedDense(
		x.outputOps[0], weights.outputOps[0], biasVal, wq.toBackend(), activation)
	if err != nil {
		panic(err)
	}
	node := &Node{
		outputOps:    []backends.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return node
}

// BackendFusedAttentionQKVProjection performs fused Query-Key-Value projection.
// Internal: prefer attention.QKVProjection which handles fallback and gradients.
func BackendFusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV *Node, queryDim, keyValueDim int) (query, key, value *Node) {
	return backendFusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV, queryDim, keyValueDim)
}

// InternalFusedOpCaller attempts to call fused, and if it panics with
// backends.ErrNotImplemented, falls back to the decomposed version. Any other
// panic is re-thrown — this prevents real bugs in fused op implementations from
// being silently swallowed.
//
// Internal: this is used by higher-level wrappers (graph.Softmax, nn.Dense,
// nn.LayerNorm) that pair a fused backend call with a decomposed fallback.
//
// When the fused call succeeds, the decomposed version is also stored as the
// VJP alternate output, so that reverse-mode autodiff can compute gradients
// through the decomposed graph without hand-written VJPs.
func InternalFusedOpCaller(fused, decomposed func() *Node) *Node {
	// Build decomposed output first so it has a lower nodeIdx than the fused
	// node. The gradient loop iterates from output down to 0, so it will
	// visit the fused node first and transfer its VJP to the decomposed
	// output, which is then processed normally.
	decomposedOutput := decomposed()

	var output *Node
	err := exceptions.TryCatch[error](func() {
		output = fused()
	})
	if err != nil {
		if backends.IsNotImplemented(err) {
			// Expected: fused op doesn't support this config; fall back to decomposed.
			return decomposedOutput
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	// Store decomposed version for VJP computation; dead-code-eliminated if unused.
	output.vjpAlternateOutputs = []*Node{decomposedOutput}
	return output
}

// InternalFusedOpCallerMulti is the multi-output counterpart of InternalFusedOpCaller.
// It handles fused ops that return multiple outputs (e.g. FusedAttentionQKVProjection returning q, k, v).
//
// Like InternalFusedOpCaller, the decomposed version is built first so it has lower
// node indices than the fused nodes. When the fused call succeeds, the decomposed
// outputs are stored as vjpAlternateOutputs on the multi-output parent node for
// reverse-mode autodiff.
func InternalFusedOpCallerMulti(fused, decomposed func() []*Node) []*Node {
	decomposedOutputs := decomposed()

	var outputs []*Node
	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if backends.IsNotImplemented(err) {
			return decomposedOutputs
		}
		panic(err)
	}

	// Find the multi-output parent node via the first split node.
	// The fused function returns split nodes; we need the underlying multi-output node.
	if len(outputs) > 0 {
		firstSplit := outputs[0]
		splitInputs, ok := firstSplit.inputs.(*nodeInputsSplitNode)
		if !ok {
			exceptions.Panicf("InternalFusedOpCallerMulti: fused function returned non-split nodes; cannot set vjpAlternateOutputs")
		}
		parent := splitInputs.multiOutputNode
		parent.vjpAlternateOutputs = decomposedOutputs
	}

	return outputs
}
