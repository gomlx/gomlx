// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
	"github.com/gomlx/gomlx/support/exceptions"
	"k8s.io/klog/v2"
)

// Quantization describes how a graph-level quantized tensor maps between quantized codes and real values.
// This is the graph-layer counterpart of compute.Quantization, holding *Node references for Scale
// and ZeroPoint instead of compute.Value so GoMLX tracks them as graph dependencies.
//
// For QuantLinear, the dequantization formula is standard zero-point affine quantization:
//
//	real_value = (float(quant_value) - float(zero_point)) * scale
//
// (When ZeroPoint is nil, zero_point = 0: real_value = float(quant_value) * scale.)
//
// Migration note for existing callers:
//   - Standard integer/float zero-points (ONNX, PyTorch, TFLite, GPTQ/AWQ qzeros) are passed directly as ZeroPoint.
//   - Legacy callers storing a post-scale float additive offset (real = float(q)*scale + offset) should convert
//     it via ZeroPoint: Neg(Div(oldOffset, scale)) (or -offset/scale on host); in weight-only float mode
//     (XQuantization == nil), float ZeroPoint values are subtracted directly in Float32 without integer rounding.
//   - Scalar [] / [1] and 1D [N] Scale/ZeroPoint nodes can be passed directly without [K, NumBlocks] broadcasts.
type Quantization struct {
	// Scheme: QuantLinear, QuantNF4, or QuantGGML.
	Scheme compute.QuantizationScheme

	// Scale is the multiplicative scale factor (Float32 / Float16 / BFloat16).
	// Accepted shapes for a 2D tensor [D0, D1] (or flattened [M, K] / [M, N] tensor):
	//   - Scalar [] or [1]: per-tensor scale.
	//   - 1D [D_other]: per-channel scale along the non-contracting axis (e.g. [N] for weights, [M] for x).
	//   - 2D [ceil(D0/BlockSize), D1] when BlockAxis == 0, or [D0, ceil(D1/BlockSize)] when BlockAxis == 1.
	// Nil for QuantGGML (scales are embedded in the blocks).
	Scale *Node

	// ZeroPoint is the optional zero-point subtracted BEFORE multiplying by Scale (only for QuantLinear).
	// Must be broadcast-compatible with Scale's shape. Nil means symmetric quantization (zero_point = 0).
	// Unused for QuantGGML and QuantNF4.
	ZeroPoint *Node

	// BlockAxis is the axis of the quantized tensor partitioned into blocks of size BlockSize
	// (0 or 1 for a 2D weight matrix; -1 or K-axis for input x).
	// Ignored when Scale is scalar or 1D per-channel, or when Scheme == QuantGGML.
	BlockAxis int

	// BlockSize is the number of elements along BlockAxis sharing one scale/zero-point.
	// If 0, inferred from ceil(dim(BlockAxis) / Scale.Shape().Dimensions[BlockAxis]).
	BlockSize int

	// GGMLType specifies the concrete GGML block format (Q4_0, Q8_0, Q4_K, Q8_K, etc.).
	// Only used when Scheme == QuantGGML.
	GGMLType compute.GGMLQuantType

	// QuantizedDType optionally specifies the target quantized output dtype (Int8 or Uint8)
	// when used as OutputQuantization and ZeroPoint is nil or float.
	QuantizedDType dtypes.DType
}

// toBackend converts the graph-level Quantization to a compute.Quantization
// by extracting the backend Values from the *Node fields.
func (q *Quantization) toBackend() *compute.Quantization {
	if q == nil {
		return nil
	}
	bq := &compute.Quantization{
		Scheme:         q.Scheme,
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
	}
	if q.Scale != nil {
		bq.Scale = q.Scale.outputOps[0]
	}
	if q.ZeroPoint != nil {
		bq.ZeroPoint = q.ZeroPoint.outputOps[0]
	}
	return bq
}

func (q *Quantization) appendNodes(dst []*Node) []*Node {
	if q == nil {
		return dst
	}
	if q.Scale != nil {
		dst = append(dst, q.Scale)
	}
	if q.ZeroPoint != nil {
		dst = append(dst, q.ZeroPoint)
	}
	return dst
}

func (q *Quantization) cloneFromInputs(newInputs []*Node, idx *int) *Quantization {
	if q == nil {
		return nil
	}
	cloned := *q
	if q.Scale != nil {
		cloned.Scale = newInputs[*idx]
		*idx++
	}
	if q.ZeroPoint != nil {
		cloned.ZeroPoint = newInputs[*idx]
		*idx++
	}
	return &cloned
}

func (q *Quantization) String() string {
	if q == nil {
		return "nil"
	}
	return fmt.Sprintf("{Scheme:%s Scale:%s ZeroPoint:%s BlockAxis:%d BlockSize:%d GGMLType:%s QuantizedDType:%s}",
		q.Scheme, strNillableNode(q.Scale), strNillableNode(q.ZeroPoint), q.BlockAxis, q.BlockSize, q.GGMLType, q.QuantizedDType)
}

// QuantizedDenseConfig holds configuration and quantization metadata for BackendFusedQuantizedDense,
// BackendFusedQuantizedDenseVJP, and nn.QuantizedDense. It is the graph-layer counterpart of
// compute.FusedQuantizedDenseConfig.
type QuantizedDenseConfig struct {
	// WeightQuantization describes how the weight tensor is quantized. Required (must not be nil).
	WeightQuantization *Quantization

	// XQuantization describes how the input x is quantized for integer-matmul schemes:
	//   - QuantLinear (Int8/Uint8 x + Scale/ZeroPoint), or
	//   - QuantGGML (GGMLQ8_0 / GGMLQ8_K packed Uint8 x).
	// Nil (the default) means x is passed in floating-point.
	XQuantization *Quantization

	// OutputQuantization optionally re-quantizes the post-activation output tensor using
	// caller-provided Scale and ZeroPoint (e.g. ONNX QLinearMatMul).
	// Nil (the default) returns a floating-point output.
	OutputQuantization *Quantization

	// QuantizedMatmuls is an optional optimization hint used when XQuantization == nil.
	// If true, the backend may dynamically quantize x on-the-fly to run integer SIMD dot products.
	QuantizedMatmuls bool

	// Activation is applied after matmul + scale + bias; defaults to ActivationNone.
	Activation compute.ActivationConfig

	// WeightLayout specifies whether weight is [K, N] (DenseLayoutInputOutputs, default)
	// or [N, K] (DenseLayoutOutputsInput; required for QuantGGML where shape is [N, bytesPerRow]).
	WeightLayout compute.DenseLayout
}

func (cfg QuantizedDenseConfig) toBackend() compute.FusedQuantizedDenseConfig {
	return compute.FusedQuantizedDenseConfig{
		WeightQuantization: cfg.WeightQuantization.toBackend(),
		XQuantization:      cfg.XQuantization.toBackend(),
		OutputQuantization: cfg.OutputQuantization.toBackend(),
		QuantizedMatmuls:   cfg.QuantizedMatmuls,
		Activation:         cfg.Activation,
		WeightLayout:       cfg.WeightLayout,
	}
}

func (cfg QuantizedDenseConfig) appendNodes(dst []*Node) []*Node {
	dst = cfg.WeightQuantization.appendNodes(dst)
	dst = cfg.XQuantization.appendNodes(dst)
	dst = cfg.OutputQuantization.appendNodes(dst)
	return dst
}

func (cfg QuantizedDenseConfig) cloneFromInputs(newInputs []*Node, idx *int) QuantizedDenseConfig {
	cloned := cfg
	cloned.WeightQuantization = cfg.WeightQuantization.cloneFromInputs(newInputs, idx)
	cloned.XQuantization = cfg.XQuantization.cloneFromInputs(newInputs, idx)
	cloned.OutputQuantization = cfg.OutputQuantization.cloneFromInputs(newInputs, idx)
	return cloned
}

// nodeInputsFusedQuantizedDense holds the inputs used for the call to compute.FusedQuantizedDense.
// Hand-written (not generated) because the backend interface uses compute.FusedQuantizedDenseConfig
// while the graph layer tracks Scale and ZeroPoint tensors as *Node dependencies.
type nodeInputsFusedQuantizedDense struct {
	x      *Node
	weight *Node
	bias   *Node
	cfg    QuantizedDenseConfig
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) Type() NodeType {
	return NodeTypeFusedQuantizedDense
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) String() string {
	return fmt.Sprintf("%s(x=[#%d], weight=[#%d], bias=%s, wq=%s, xq=%s, oq=%s, quantizedMatmuls=%v, activation=%+v, layout=%s)",
		ni.Type(),
		ni.x.Id(),
		ni.weight.Id(),
		strNillableNode(ni.bias),
		ni.cfg.WeightQuantization,
		ni.cfg.XQuantization,
		ni.cfg.OutputQuantization,
		ni.cfg.QuantizedMatmuls,
		ni.cfg.Activation,
		ni.cfg.WeightLayout,
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDense) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendFusedQuantizedDense is:
	// [x, weight, wq.Scale?, wq.ZeroPoint?, xq.Scale?, xq.ZeroPoint?, oq.Scale?, oq.ZeroPoint?, bias?]
	idx := 0
	newX := newInputs[idx]
	idx++
	newWeight := newInputs[idx]
	idx++
	newCfg := ni.cfg.cloneFromInputs(newInputs, &idx)
	var newBias *Node
	if ni.bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	return BackendFusedQuantizedDense(newX, newWeight, newBias, newCfg)
}

// BackendFusedSoftmax computes softmax along the specified axis.
// Internal: prefer graph.Softmax which handles fallback and gradients.
func BackendFusedSoftmax(x *Node, axis int) *Node { return backendFusedSoftmax(x, axis) }

// BackendFusedActivation applies the configured activation function.
// Internal: prefer activations.Apply which handles fallback and gradients.
func BackendFusedActivation(x *Node, cfg compute.ActivationConfig) *Node {
	return backendFusedActivation(x, cfg)
}

// BackendFusedActivationVJP computes the vector-jacobian product of the configured activation.
// Internal: this is used for reverse-mode autodiff of FusedActivation.
func BackendFusedActivationVJP(y, x, dOutput *Node, cfg compute.ActivationConfig) *Node {
	return backendFusedActivationVJP(y, x, dOutput, cfg)
}

// BackendFusedLayerNorm applies layer normalization. gamma and beta are optional (nil to skip).
// Internal: prefer nn.LayerNorm which handles fallback and gradients.
func BackendFusedLayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	return backendFusedLayerNorm(x, axes, epsilon, gamma, beta)
}

// nodeInputsFusedLayerNormVJP holds the inputs used for the call to compute.FusedLayerNormVJP.
type nodeInputsFusedLayerNormVJP struct {
	x       *Node
	dOutput *Node
	axes    []int
	epsilon float64
	gamma   *Node
	beta    *Node
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedLayerNormVJP) Type() NodeType {
	return NodeTypeFusedLayerNormVJP
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedLayerNormVJP) String() string {
	return fmt.Sprintf("%s(x=[#%d], dOutput=[#%d], axes=%v, epsilon=%v, gamma=%s, beta=%s)",
		ni.Type(),
		ni.x.Id(),
		ni.dOutput.Id(),
		ni.axes,
		ni.epsilon,
		strNillableNode(ni.gamma),
		strNillableNode(ni.beta),
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsFusedLayerNormVJP) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendFusedLayerNormVJP: [x, dOutput, gamma?, beta?]
	idx := 0
	newX := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newGamma *Node
	if ni.gamma != nil {
		newGamma = newInputs[idx]
		idx++
	}
	var newBeta *Node
	if ni.beta != nil {
		newBeta = newInputs[idx]
		idx++
	}
	r0, _, _ := BackendFusedLayerNormVJP(newX, newDOutput, ni.axes, ni.epsilon, newGamma, newBeta)
	if r0.Type() == NodeTypeSplitNode {
		return r0.inputNodes[0]
	}
	return r0
}

// BackendFusedLayerNormVJP computes the vector-jacobian product (gradients) of FusedLayerNorm
// with respect to its inputs (x, gamma, and beta).
// Internal: this is used for reverse-mode autodiff of FusedLayerNorm.
func BackendFusedLayerNormVJP(x, dOutput *Node, axes []int, epsilon float64, gamma, beta *Node) (dx, dGamma, dBeta *Node) {
	inputNodes := []*Node{x, dOutput}
	if gamma != nil {
		inputNodes = append(inputNodes, gamma)
	}
	if beta != nil {
		inputNodes = append(inputNodes, beta)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsFusedLayerNormVJP{
		x:       x,
		dOutput: dOutput,
		axes:    slices.Clone(axes),
		epsilon: epsilon,
		gamma:   gamma,
		beta:    beta,
	}
	var gammaVal, betaVal compute.Value
	if gamma != nil {
		gammaVal = gamma.outputOps[0]
	}
	if beta != nil {
		betaVal = beta.outputOps[0]
	}
	v0, v1, v2, err := g.currentFunc.backendFunc.FusedLayerNormVJP(
		x.outputOps[0], dOutput.outputOps[0], inputs.axes, inputs.epsilon, gammaVal, betaVal)
	if err != nil {
		panic(err)
	}

	outputOps := []compute.Value{v0}
	outputShapes := []shapes.Shape{mustNoError(g.builder.OpShape(v0))}
	if gamma != nil {
		outputOps = append(outputOps, v1)
		outputShapes = append(outputShapes, mustNoError(g.builder.OpShape(v1)))
	}
	if beta != nil {
		outputOps = append(outputOps, v2)
		outputShapes = append(outputShapes, mustNoError(g.builder.OpShape(v2)))
	}

	node := &Node{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := splitNode(node)
	outIdx := 0
	dx = splitNodes[outIdx]
	outIdx++
	if gamma != nil {
		dGamma = splitNodes[outIdx]
		outIdx++
	}
	if beta != nil {
		dBeta = splitNodes[outIdx]
	}
	return
}

// fusedLayerNormVJP is the registered VJP for NodeTypeFusedLayerNorm.
func fusedLayerNormVJP(node *Node, vjp *Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedLayerNorm)
	dx, dGamma, dBeta := BackendFusedLayerNormVJP(inputs.x, vjp, inputs.axes, inputs.epsilon, inputs.gamma, inputs.beta)
	res := []*Node{ConvertDType(dx, inputs.x.DType())}
	if inputs.gamma != nil {
		res = append(res, ConvertDType(dGamma, inputs.gamma.DType()))
	}
	if inputs.beta != nil {
		res = append(res, ConvertDType(dBeta, inputs.beta.DType()))
	}
	return res
}

// BackendFusedDense performs fused matmul + optional bias + optional activation.
// Internal: prefer nn.Dense which handles fallback and gradients.
func BackendFusedDense(x, weight, bias *Node, options compute.DenseConfig) *Node {
	return backendFusedDense(x, weight, bias, options)
}

// nodeInputsFusedDenseVJP holds the inputs used for the call to compute.FusedDenseVJP.
type nodeInputsFusedDenseVJP struct {
	x       *Node
	weight  *Node
	bias    *Node
	y       *Node
	dOutput *Node
	options compute.DenseConfig
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedDenseVJP) Type() NodeType {
	return NodeTypeFusedDenseVJP
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedDenseVJP) String() string {
	return fmt.Sprintf("%s(x=[#%d], weight=[#%d], bias=%s, y=[#%d], dOutput=[#%d], options=%+v)",
		ni.Type(),
		ni.x.Id(),
		ni.weight.Id(),
		strNillableNode(ni.bias),
		ni.y.Id(),
		ni.dOutput.Id(),
		ni.options,
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsFusedDenseVJP) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendFusedDenseVJP: [x, weight, y, dOutput, bias?]
	idx := 0
	newX := newInputs[idx]
	idx++
	newWeight := newInputs[idx]
	idx++
	newY := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newBias *Node
	if ni.bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	r0, _, _ := BackendFusedDenseVJP(newX, newWeight, newBias, newY, newDOutput, ni.options)
	return r0.inputNodes[0]
}

// BackendFusedDenseVJP computes the vector-jacobian product (gradients) of FusedDense with respect to
// its inputs (x, weight, and bias).
// Internal: this is used for reverse-mode autodiff of FusedDense.
func BackendFusedDenseVJP(x, weight, bias, y, dOutput *Node, options compute.DenseConfig) (dx, dWeight, dBias *Node) {
	inputNodes := []*Node{x, weight, y, dOutput}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsFusedDenseVJP{
		x:       x,
		weight:  weight,
		bias:    bias,
		y:       y,
		dOutput: dOutput,
		options: options,
	}
	var biasVal compute.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}
	v0, v1, v2, err := g.currentFunc.backendFunc.FusedDenseVJP(
		x.outputOps[0], weight.outputOps[0], biasVal, y.outputOps[0], dOutput.outputOps[0], options)
	if err != nil {
		panic(err)
	}

	var outputOps []compute.Value
	var outputShapes []shapes.Shape
	if bias != nil {
		outputOps = []compute.Value{v0, v1, v2}
		outputShapes = []shapes.Shape{
			mustNoError(g.builder.OpShape(v0)),
			mustNoError(g.builder.OpShape(v1)),
			mustNoError(g.builder.OpShape(v2)),
		}
	} else {
		outputOps = []compute.Value{v0, v1}
		outputShapes = []shapes.Shape{
			mustNoError(g.builder.OpShape(v0)),
			mustNoError(g.builder.OpShape(v1)),
		}
	}

	node := &Node{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := splitNode(node)
	dx, dWeight = splitNodes[0], splitNodes[1]
	if bias != nil {
		dBias = splitNodes[2]
	}
	return
}

// fusedDenseVJP is the registered VJP for NodeTypeFusedDense.
func fusedDenseVJP(node *Node, vjp *Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedDense)
	dx, dWeight, dBias := BackendFusedDenseVJP(inputs.x, inputs.weight, inputs.bias, node, vjp, inputs.options)
	if inputs.bias != nil {
		return []*Node{
			ConvertDType(dx, inputs.x.DType()),
			ConvertDType(dWeight, inputs.weight.DType()),
			ConvertDType(dBias, inputs.bias.DType()),
		}
	}
	return []*Node{
		ConvertDType(dx, inputs.x.DType()),
		ConvertDType(dWeight, inputs.weight.DType()),
	}
}

// fusedActivationVJP is the registered VJP for NodeTypeFusedActivation.
func fusedActivationVJP(node *Node, vjp *Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedActivation)
	dx := BackendFusedActivationVJP(node, inputs.x, vjp, inputs.cfg)
	return []*Node{ConvertDType(dx, inputs.x.DType())}
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
func BackendFusedScaledDotProductAttention(query, key, value *Node, axesLayout compute.AttentionAxesLayout, options *compute.ScaledDotProductAttentionConfig) (output *Node, statesForVJP []*Node) {
	return backendFusedScaledDotProductAttention(query, key, value, axesLayout, options)
}

// fusedScaledDotProductAttentionVJP is the registered VJP for NodeTypeFusedScaledDotProductAttention.
// It threads statesForVJP into the fused backward, so the [B,H,S,S] scores never materialize.
func fusedScaledDotProductAttentionVJP(node *Node, vjps []*Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedScaledDotProductAttention)
	splitOutputs := splitNode(node)
	output, statesForVJP := splitOutputs[0], splitOutputs[1:]
	dOutput := vjps[0] // adjoint of the attention output; the state adjoints are unused.
	dQuery, dKey, dValue := backendFusedScaledDotProductAttentionVJP(
		inputs.query, inputs.key, inputs.value,
		inputs.axesLayout, inputs.options,
		output, statesForVJP, dOutput)
	// Grads align to node.inputNodes = [query, key, value]. The fused backward runs in bf16;
	// cast each grad back to its input's dtype so autodiff accumulation matches.
	grads := []*Node{
		ConvertDType(dQuery, inputs.query.DType()),
		ConvertDType(dKey, inputs.key.DType()),
		ConvertDType(dValue, inputs.value.DType()),
	}
	return grads
}

// BackendFusedQuantizedDense performs fused quantized matmul + optional bias + optional activation
// (+ optional output re-quantization when cfg.OutputQuantization != nil).
//
// Bias handling (when bias != nil, shape [N]):
//   - Floating-point bias: added in floating-point AFTER dequantization / (xScale * wScale) rescaling
//     and BEFORE cfg.Activation (and cfg.OutputQuantization).
//   - Int32 bias (valid only when cfg.XQuantization != nil with Scheme == compute.QuantLinear and
//     neither XQuantization.Scale nor WeightQuantization.Scale partitions K into multiple blocks):
//     added directly to the Int32 matmul accumulator BEFORE converting to Float32 and multiplying by
//     xScale[m] * wScale[n], i.e. z[m, n] = float32(acc_int32[m, n] + bias[n]) * (xScale[m] * wScale[n]).
//
// Internal: prefer nn.QuantizedDense which handles fallback and gradients.
func BackendFusedQuantizedDense(x, weight, bias *Node, cfg QuantizedDenseConfig) *Node {
	if cfg.WeightQuantization == nil {
		exceptions.Panicf("BackendFusedQuantizedDense: cfg.WeightQuantization must not be nil")
	}
	wq := cfg.WeightQuantization
	if wq.Scale == nil && wq.Scheme != compute.QuantGGML {
		exceptions.Panicf("BackendFusedQuantizedDense: WeightQuantization.Scale must not be nil for scheme %s", wq.Scheme)
	}

	// inputNodes ordering:
	// [x, weight, wq.Scale?, wq.ZeroPoint?, xq.Scale?, xq.ZeroPoint?, oq.Scale?, oq.ZeroPoint?, bias?].
	inputNodes := []*Node{x, weight}
	inputNodes = cfg.appendNodes(inputNodes)
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	// Clone Quantization pointers so caller mutations after graph construction do not alias node inputs.
	idx := 2
	clonedCfg := cfg.cloneFromInputs(inputNodes, &idx)
	inputs := &nodeInputsFusedQuantizedDense{
		x:      x,
		weight: weight,
		bias:   bias,
		cfg:    clonedCfg,
	}

	var biasVal compute.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.FusedQuantizedDense(
		x.outputOps[0], weight.outputOps[0], biasVal, clonedCfg.toBackend())
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

// nodeInputsFusedQuantizedDenseVJP holds the inputs used for the call to compute.FusedQuantizedDenseVJP.
type nodeInputsFusedQuantizedDenseVJP struct {
	x       *Node
	weight  *Node
	bias    *Node
	y       *Node
	dOutput *Node
	cfg     QuantizedDenseConfig
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDenseVJP) Type() NodeType {
	return NodeTypeFusedQuantizedDenseVJP
}

// String implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDenseVJP) String() string {
	return fmt.Sprintf("%s(x=%s, weight=[#%d], bias=%s, y=%s, dOutput=[#%d], wq=%s, quantizedMatmuls=%v, activation=%+v, layout=%s)",
		ni.Type(),
		strNillableNode(ni.x),
		ni.weight.Id(),
		strNillableNode(ni.bias),
		strNillableNode(ni.y),
		ni.dOutput.Id(),
		ni.cfg.WeightQuantization,
		ni.cfg.QuantizedMatmuls,
		ni.cfg.Activation,
		ni.cfg.WeightLayout,
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *nodeInputsFusedQuantizedDenseVJP) CloneWithInputs(originalNode *Node, newInputs ...*Node) *Node {
	// inputNodes ordering from BackendFusedQuantizedDenseVJP:
	// [weight, dOutput, x?, y?, wq.Scale?, wq.ZeroPoint?, xq.Scale?, xq.ZeroPoint?, oq.Scale?, oq.ZeroPoint?, bias?]
	idx := 0
	newWeight := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newX *Node
	if ni.x != nil {
		newX = newInputs[idx]
		idx++
	}
	var newY *Node
	if ni.y != nil {
		newY = newInputs[idx]
		idx++
	}
	newCfg := ni.cfg.cloneFromInputs(newInputs, &idx)
	var newBias *Node
	if ni.bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	r0, _ := BackendFusedQuantizedDenseVJP(newX, newWeight, newBias, newY, newDOutput, newCfg)
	if r0.Type() == NodeTypeSplitNode {
		return r0.inputNodes[0]
	}
	return r0
}

// BackendFusedQuantizedDenseVJP computes the vector-jacobian product (gradients) of FusedQuantizedDense
// with respect to its floating-point inputs (x and optional bias), dequantizing weight on-the-fly
// without materializing the full float weight matrix.
// Internal: used for reverse-mode autodiff of NodeTypeFusedQuantizedDense.
func BackendFusedQuantizedDenseVJP(x, weight, bias, y, dOutput *Node, cfg QuantizedDenseConfig) (dx, dBias *Node) {
	if cfg.WeightQuantization == nil {
		exceptions.Panicf("BackendFusedQuantizedDenseVJP: cfg.WeightQuantization must not be nil")
	}
	inputNodes := []*Node{weight, dOutput}
	if x != nil {
		inputNodes = append(inputNodes, x)
	}
	if y != nil {
		inputNodes = append(inputNodes, y)
	}
	cfgStartIdx := len(inputNodes)
	inputNodes = cfg.appendNodes(inputNodes)
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	idx := cfgStartIdx
	clonedCfg := cfg.cloneFromInputs(inputNodes, &idx)
	inputs := &nodeInputsFusedQuantizedDenseVJP{
		x:       x,
		weight:  weight,
		bias:    bias,
		y:       y,
		dOutput: dOutput,
		cfg:     clonedCfg,
	}

	var xVal, biasVal, yVal compute.Value
	if x != nil {
		xVal = x.outputOps[0]
	}
	if bias != nil {
		biasVal = bias.outputOps[0]
	}
	if y != nil {
		yVal = y.outputOps[0]
	}

	v0, v1, err := g.currentFunc.backendFunc.FusedQuantizedDenseVJP(
		xVal, weight.outputOps[0], biasVal, yVal, dOutput.outputOps[0], clonedCfg.toBackend())
	if err != nil {
		panic(err)
	}

	outputOps := []compute.Value{v0}
	outputShapes := []shapes.Shape{mustNoError(g.builder.OpShape(v0))}
	if bias != nil {
		outputOps = append(outputOps, v1)
		outputShapes = append(outputShapes, mustNoError(g.builder.OpShape(v1)))
	}

	node := &Node{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := splitNode(node)
	dx = splitNodes[0]
	if bias != nil {
		dBias = splitNodes[1]
	}
	return
}

// fusedQuantizedDenseVJP is the registered VJP for NodeTypeFusedQuantizedDense.
func fusedQuantizedDenseVJP(node *Node, vjp *Node, _ shapes.Shape) []*Node {
	inputs := node.inputs.(*nodeInputsFusedQuantizedDense)
	var xForVJP, yForVJP *Node
	if inputs.x.Shape().IsDynamic() || inputs.x.DType() != vjp.DType() || len(inputs.x.Shape().AxisNames) > 0 {
		xForVJP = inputs.x
	}
	if inputs.cfg.Activation.Type != compute.ActivationNone {
		yForVJP = node
	}
	dx, dBias := BackendFusedQuantizedDenseVJP(xForVJP, inputs.weight, inputs.bias, yForVJP, vjp, inputs.cfg)
	// Align 1-to-1 with node.inputNodes:
	//   - dx at index 0 (for x)
	//   - nil for weight and quantization Scale/ZeroPoint nodes (discrete/frozen)
	//   - dBias at len(vjps)-1 (for bias, when bias != nil)
	vjps := make([]*Node, len(node.inputNodes))
	vjps[0] = ConvertDType(dx, inputs.x.DType())
	if inputs.bias != nil {
		vjps[len(vjps)-1] = ConvertDType(dBias, inputs.bias.DType())
	}
	return vjps
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
// If useDecomposedAlternateVJP is set, the decomposed version is stored as the
// vjpAlternateOutputs on the returned node, so that reverse-mode autodiff can
// compute gradients through the decomposed graph without hand-written VJPs.
// This should be set to true if there is no corresponding fused VJP.
//
// The returned opFused indicates whether the fused operation was taken (true) or
// fell back to decomposed (false).
//
// Internal: this is used by higher-level wrappers (graph.Softmax, nn.Dense,
// nn.LayerNorm) that pair a fused backend call with a decomposed fallback.
//
// Note: If "GOMLX_FUSION" (see FusionEnv constant) is disabled (set to "0", "false", etc.) the fused
// version is never used. The default is enabled though.
func InternalFusedOpCaller(fused, decomposed func() *Node, useDecomposedAlternateVJP bool) (output *Node, opFused bool) {
	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed(), false
	}

	// When useDecomposedAlternateVJP is true, build the decomposed output first so it has a lower
	// nodeIdx than the fused node. The gradient loop iterates from output down to 0, so it will
	// visit the fused node first and transfer its VJP to the decomposed output.
	var decomposedOutput *Node
	if useDecomposedAlternateVJP {
		decomposedOutput = decomposed()
	}

	klog.V(2).Info("Attempt to call fused op")
	err := exceptions.TryCatch[error](func() {
		output = fused()
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			if klog.V(2).Enabled() {
				klog.Infof("InternalFusedOpCaller(): failed to call fused op (not implemented). Falling back to decomposed op: %v", err)
			}
			if decomposedOutput == nil {
				decomposedOutput = decomposed()
			}
			// Expected: fused op doesn't support this config; fall back to decomposed.
			return decomposedOutput, false
		}
		// Unexpected error — re-panic so it surfaces instead of being silently masked.
		panic(err)
	}
	klog.V(2).Info("InternalFusedOpCaller(): Successfully called fused op.")
	// Store decomposed version for VJP computation if requested; dead-code-eliminated if unused.
	if useDecomposedAlternateVJP {
		output.vjpAlternateOutputs = []*Node{decomposedOutput}
	}
	return output, true
}

// InternalFusedOpCallerMulti is the multi-output counterpart of InternalFusedOpCaller.
// It handles fused ops that return multiple outputs (e.g. FusedAttentionQKVProjection returning q, k, v).
//
// Like InternalFusedOpCaller, the decomposed version is built first so it has lower
// node indices than the fused nodes. When the fused call succeeds and useDecomposedAlternateVJP
// is set, the decomposed outputs are stored as vjpAlternateOutputs on each individual
// output node (which are split nodes under the hood) for reverse-mode autodiff.
// This should be set to true if there is no corresponding fused VJP.
//
// The returned opFused indicates whether the fused operation was taken (true) or
// fell back to decomposed (false).
func InternalFusedOpCallerMulti(fused, decomposed func() []*Node, useDecomposedAlternateVJP bool) (outputs []*Node, opFused bool) {
	decomposedOutputs := decomposed()

	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			if klog.V(2).Enabled() {
				klog.Infof("InternalFusedOpCallerMulti(): failed to call fused op (not implemented). Falling back to decomposed op: %v", err)
			}
			return decomposedOutputs, false
		}
		panic(err)
	}
	if len(outputs) != len(decomposedOutputs) {
		exceptions.Panicf("InternalFusedOpCallerMulti: fused function returned %d outputs, but decomposed returned %d", len(outputs), len(decomposedOutputs))
	}
	klog.V(2).Info("InternalFusedOpCallerMulti(): Successfully called fused op.")

	if useDecomposedAlternateVJP {
		for i, output := range outputs {
			output.vjpAlternateOutputs = []*Node{decomposedOutputs[i]}
		}
	}

	return outputs, true
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
