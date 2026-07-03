// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
	"github.com/gomlx/gomlx/support/exceptions"
	"k8s.io/klog/v2"
)

// Quantization describes how a graph-level quantized tensor is dequantized.
// This is the graph-layer counterpart of compute.Quantization, holding *Node
// references for Scale and ZeroPoint instead of compute.Value.
type Quantization struct {
	// Scheme: Linear, NF4, or GGML.
	Scheme compute.QuantizationScheme

	// Scale is the multiplicative factor.
	// Shape: [K, NumBlocks] (block-wise), where K is the input-features
	// (contracting) dimension of the [K, N] weight matrix and
	// NumBlocks = ceil(N / BlockSize).
	// Nil for QuantGGML (scales are embedded in the blocks).
	Scale *Node

	// ZeroPoint is the additive offset (only for Linear).
	// If nil, the quantization is assumed symmetric.
	// Unused for QuantGGML and QuantNF4.
	ZeroPoint *Node

	// BlockAxis is the dimension of the quantized tensor that is blocked.
	// This is the output-features dimension (axis 1) of a [K, N] weight matrix.
	// Currently only BlockAxis=1 is supported.
	// Unused for QuantGGML.
	BlockAxis int

	// BlockSize is the number of elements in BlockAxis that share one scale.
	// If BlockSize == values.Shape()[BlockAxis], it's effectively per-axis quantization.
	// Unused for QuantGGML.
	BlockSize int

	// GGMLType specifies the concrete GGML block format (Q4_0, Q8_0, etc.).
	// Only used when Scheme == QuantGGML.
	GGMLType compute.GGMLQuantType
}

// toBackend converts the graph-level Quantization to a compute.Quantization
// by extracting the backend Values from the *Node fields.
func (q *Quantization) toBackend() *compute.Quantization {
	bq := &compute.Quantization{
		Scheme:    q.Scheme,
		BlockAxis: q.BlockAxis,
		BlockSize: q.BlockSize,
		GGMLType:  q.GGMLType,
	}
	if q.Scale != nil {
		bq.Scale = q.Scale.outputOps[0]
	}
	if q.ZeroPoint != nil {
		bq.ZeroPoint = q.ZeroPoint.outputOps[0]
	}
	return bq
}

// nodeInputsFusedQuantizedDense holds the inputs used for the call to compute.FusedQuantizedDense.
// Hand-written (not generated) because the backend interface uses a *compute.Quantization struct
// but the graph layer stores scale and zeroPoint as individual *Node values.
type nodeInputsFusedQuantizedDense struct {
	x          *Node
	weights    *Node
	bias       *Node
	wq         *Quantization
	activation compute.ActivationType
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) Type() NodeType {
	return NodeTypeFusedQuantizedDense
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) String() string {
	return fmt.Sprintf("%s(x=[#%d], weights=[#%d], bias=%s, scale=%s, zeroPoint=%s, scheme=%s, blockAxis=%v, blockSize=%v, ggmlType=%s, activation=%s)",
		ni.Type(),
		ni.x.Id(),
		ni.weights.Id(),
		strNillableNode(ni.bias),
		strNillableNode(ni.wq.Scale),
		strNillableNode(ni.wq.ZeroPoint),
		ni.wq.Scheme,
		ni.wq.BlockAxis,
		ni.wq.BlockSize,
		ni.wq.GGMLType,
		ni.activation,
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendFusedQuantizedDense is:
	// [x, weights, scale?, zeroPoint?, bias?]
	idx := 0
	newX := newInputs[idx]
	idx++
	newWeights := newInputs[idx]
	idx++
	var newScale *Node
	if ni.wq.Scale != nil {
		newScale = newInputs[idx]
		idx++
	}
	var newZeroPoint *Node
	if ni.wq.ZeroPoint != nil {
		newZeroPoint = newInputs[idx]
		idx++
	}
	var newBias *Node
	if ni.bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	// Note: We need a new Quantization spec because Scale and ZeroPoint might have changed.
	newWq := &Quantization{
		Scale:     newScale,
		ZeroPoint: newZeroPoint,
		Scheme:    ni.wq.Scheme,
		BlockAxis: ni.wq.BlockAxis,
		BlockSize: ni.wq.BlockSize,
		GGMLType:  ni.wq.GGMLType,
	}
	return BackendFusedQuantizedDense(newX, newWeights, newBias, newWq, ni.activation)
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
func BackendFusedDense(x, weight, bias *Node, activation compute.ActivationType) *Node {
	return backendFusedDense(x, weight, bias, activation)
}

// BackendFusedScaledDotProductAttention computes multi-head scaled dot-product attention via the
// backend's fused op. It returns the attention output plus statesForVJP: the backend-specific state
// tensors the fused backward consumes (the cuDNN flash softmax stats today; a slice so other
// backends can carry a different set, e.g. an extra workspace tensor). statesForVJP is empty when
// the backend has no fused backward. The gradient is the registered VJP
// (fusedScaledDotProductAttentionVJP), which threads statesForVJP into the fused backward so the
// [B,H,S,S] scores never materialize in either pass. Backends without the fused op panic with a
// wrapped compute.ErrNotImplemented, so callers fall back.
//
// Internal: prefer to use the package ml/layers/attention instead, which provides the standard (and
// stable) API for attention layers. This is an internal API that may change without notice.
func BackendFusedScaledDotProductAttention(query, key, value, mask *Node, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (output *Node, statesForVJP []*Node) {
	return backendFusedScaledDotProductAttention(query, key, value, mask, numHeads, numKVHeads, axesLayout, scale, causal, options)
}

// fusedScaledDotProductAttentionVJP is the registered VJP for NodeTypeFusedScaledDotProductAttention.
// It threads statesForVJP into the fused backward, so the [B,H,S,S] scores never materialize.
func fusedScaledDotProductAttentionVJP(node *Node, vjps []*Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedScaledDotProductAttention)
	splitOutputs := splitNode(node)
	output, statesForVJP := splitOutputs[0], splitOutputs[1:]
	dOutput := vjps[0] // adjoint of the attention output; the state adjoints are unused.
	dQuery, dKey, dValue := backendFusedScaledDotProductAttentionVJP(
		inputs.query, inputs.key, inputs.value, inputs.mask,
		inputs.numHeads, inputs.numKVHeads, inputs.axesLayout, inputs.scale, inputs.causal, inputs.options,
		output, statesForVJP, dOutput)
	// Grads align to node.inputNodes = [query, key, value (, mask)]. The fused backward runs in bf16;
	// cast each grad back to its input's dtype so autodiff accumulation matches.
	grads := []*Node{
		ConvertDType(dQuery, inputs.query.DType()),
		ConvertDType(dKey, inputs.key.DType()),
		ConvertDType(dValue, inputs.value.DType()),
	}
	if inputs.mask != nil {
		grads = append(grads, nil) // causal flash takes no mask operand; mask is not differentiated.
	}
	return grads
}

// BackendFusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
// Internal: prefer nn.QuantizedDense which handles fallback and gradients.
func BackendFusedQuantizedDense(x, weights, bias *Node,
	wq *Quantization, activation compute.ActivationType) *Node {

	if wq == nil {
		exceptions.Panicf("BackendFusedQuantizedDense: wq must not be nil")
	}
	if wq.Scale == nil && wq.Scheme != compute.QuantGGML {
		exceptions.Panicf("BackendFusedQuantizedDense: wq.Scale must not be nil for scheme %s", wq.Scheme)
	}

	// inputNodes ordering: [x, weights, scale?, zeroPoint?, bias?].
	inputNodes := []*Node{x, weights}
	if wq.Scale != nil {
		inputNodes = append(inputNodes, wq.Scale)
	}
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

	var biasVal compute.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.FusedQuantizedDense(
		x.outputOps[0], weights.outputOps[0], biasVal, wq.toBackend(), activation)
	if err != nil {
		panic(err)
	}
	node := &Node{
		outputOps:    []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return node
}

// nodeInputsQuantizedEmbeddingLookup holds the inputs used for the call to compute.QuantizedEmbeddingLookup.
// Hand-written (not generated) to match the pattern of BackendFusedQuantizedDense: it accepts
// a graph-level *Quantization and converts it to compute.Quantization via toBackend().
type nodeInputsQuantizedEmbeddingLookup struct {
	data    *Node
	indices *Node
	tq      *Quantization
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsQuantizedEmbeddingLookup) Type() NodeType {
	return NodeTypeQuantizedEmbeddingLookup
}

// String implements the interface NodeInputs.
func (ni *nodeInputsQuantizedEmbeddingLookup) String() string {
	return fmt.Sprintf("%s(data=[#%d], indices=[#%d], scheme=%s, ggmlType=%s)",
		ni.Type(),
		ni.data.Id(),
		ni.indices.Id(),
		ni.tq.Scheme,
		ni.tq.GGMLType,
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsQuantizedEmbeddingLookup) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendQuantizedEmbeddingLookup is: [data, indices]
	return BackendQuantizedEmbeddingLookup(newInputs[0], newInputs[1], ni.tq)
}

// BackendQuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
// with on-the-fly dequantization.
// Internal: prefer nn.QuantizedGather which handles fallback and gradients.
func BackendQuantizedEmbeddingLookup(data, indices *Node, tq *Quantization) *Node {
	if tq == nil {
		exceptions.Panicf("BackendQuantizedEmbeddingLookup: tq must not be nil")
	}

	inputNodes := []*Node{data, indices}
	g := validateBuildingGraphFromInputs(inputNodes...)

	inputs := &nodeInputsQuantizedEmbeddingLookup{
		data:    data,
		indices: indices,
		tq:      tq,
	}

	result, err := g.currentFunc.backendFunc.QuantizedEmbeddingLookup(
		data.outputOps[0], indices.outputOps[0], tq.toBackend())
	if err != nil {
		panic(err)
	}
	node := &Node{
		outputOps:    []compute.Value{result},
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

// FusionEnv is the name of the environment variable that controls whether fused ops are used.
// The default value is true.
const FusionEnv = "GOMLX_FUSION"

// InternalFusedOpCaller attempts to call fused, and if it panics with
// compute.ErrNotImplemented, falls back to the decomposed version. Any other
// panic is re-thrown — this prevents real bugs in fused op implementations from
// being silently swallowed.
//
// Internal: this is used by higher-level wrappers (graph.Softmax, nn.Dense,
// nn.LayerNorm) that pair a fused backend call with a decomposed fallback.
//
// When the fused call succeeds, the decomposed version is also stored as the
// VJP alternate output, so that reverse-mode autodiff can compute gradients
// through the decomposed graph without hand-written VJPs.
//
// Note: If "GOMLX_FUSION" (see FusionEnv constant) is disabled (set to "0", "false", etc.) the fused
// version is never used. The default is enabled though.
func InternalFusedOpCaller(fused, decomposed func() *Node) *Node {
	// Build decomposed output first so it has a lower nodeIdx than the fused
	// node. The gradient loop iterates from output down to 0, so it will
	// visit the fused node first and transfer its VJP to the decomposed
	// output, which is then processed normally.
	decomposedOutput := decomposed()

	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposedOutput
	}

	klog.V(2).Info("Attempt to call fused op")
	var output *Node
	err := exceptions.TryCatch[error](func() {
		output = fused()
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			klog.V(2).Infof("Failed to call fused op (not implemented). Falling back to decomposed op: %v", err)
			// Expected: fused op doesn't support this config; fall back to decomposed.
			return decomposedOutput
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	klog.V(2).Info("Successfully called fused op.")
	// Store decomposed version for VJP computation; dead-code-eliminated if unused.
	output.vjpAlternateOutputs = []*Node{decomposedOutput}
	return output
}

// InternalFusedOpCallerMulti is the multi-output counterpart of InternalFusedOpCaller.
// It handles fused ops that return multiple outputs (e.g. FusedAttentionQKVProjection returning q, k, v).
//
// Like InternalFusedOpCaller, the decomposed version is built first so it has lower
// node indices than the fused nodes. When the fused call succeeds, the decomposed
// outputs are stored as vjpAlternateOutputs on each individual output node (which
// are split nodes under the hood) for reverse-mode autodiff.
func InternalFusedOpCallerMulti(fused, decomposed func() []*Node) []*Node {
	decomposedOutputs := decomposed()

	var outputs []*Node
	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			return decomposedOutputs
		}
		panic(err)
	}

	if len(outputs) != len(decomposedOutputs) {
		exceptions.Panicf("InternalFusedOpCallerMulti: fused function returned %d outputs, but decomposed returned %d", len(outputs), len(decomposedOutputs))
	}
	for i, output := range outputs {
		output.vjpAlternateOutputs = []*Node{decomposedOutputs[i]}
	}

	return outputs
}

// InternalBackendOutputs retrieves the unexported backend outputs of a node.
//
// WARNING: This is an internal GoMLX API. It is exposed specifically for
// layer packages (like `attention`) that need to access backend details to
// stitch together fused operations (e.g., FusedScaledDotProductAttention).
//
// End-users building standard computation graphs should never need to call this.
func InternalBackendOutputs(n *Node) []compute.Value {
	return n.outputOps
}
