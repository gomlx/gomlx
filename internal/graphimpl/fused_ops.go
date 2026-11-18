// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

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

// FusionEnv is the environment variable that controls whether fused ops are used (default true).
const FusionEnv = "GOMLX_FUSION"

// Quantization describes quantization parameters at the graphimpl layer using *NodeImpl references.
type Quantization struct {
	Scheme         compute.QuantizationScheme
	Scale          *NodeImpl
	ZeroPoint      *NodeImpl
	BlockAxis      int
	BlockSize      int
	GGMLType       compute.GGMLQuantType
	QuantizedDType dtypes.DType
	PackedDType    dtypes.DType
}

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
		PackedDType:    q.PackedDType,
	}
	if q.Scale != nil {
		bq.Scale = q.Scale.outputOps[0]
	}
	if q.ZeroPoint != nil {
		bq.ZeroPoint = q.ZeroPoint.outputOps[0]
	}
	return bq
}

func (q *Quantization) appendNodes(dst []*NodeImpl) []*NodeImpl {
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

func (q *Quantization) cloneFromInputs(newInputs []*NodeImpl, idx *int) *Quantization {
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
	return fmt.Sprintf("{Scheme:%s Scale:%s ZeroPoint:%s BlockAxis:%d BlockSize:%d GGMLType:%s QuantizedDType:%s PackedDType:%s}",
		q.Scheme, strNillableNode(q.Scale), strNillableNode(q.ZeroPoint), q.BlockAxis, q.BlockSize, q.GGMLType, q.QuantizedDType, q.PackedDType)
}

// QuantizedDenseConfig holds configuration for BackendFusedQuantizedDense and BackendFusedQuantizedDenseVJP.
type QuantizedDenseConfig struct {
	WeightQuantization *Quantization
	XQuantization      *Quantization
	OutputQuantization *Quantization
	QuantizedMatmul    bool
	Activation         compute.ActivationConfig
	WeightLayout       compute.DenseLayout
}

func (cfg QuantizedDenseConfig) toBackend() compute.FusedQuantizedDenseConfig {
	return compute.FusedQuantizedDenseConfig{
		WeightQuantization: cfg.WeightQuantization.toBackend(),
		XQuantization:      cfg.XQuantization.toBackend(),
		OutputQuantization: cfg.OutputQuantization.toBackend(),
		QuantizedMatmul:    cfg.QuantizedMatmul,
		Activation:         cfg.Activation,
		WeightLayout:       cfg.WeightLayout,
	}
}

func (cfg QuantizedDenseConfig) appendNodes(dst []*NodeImpl) []*NodeImpl {
	dst = cfg.WeightQuantization.appendNodes(dst)
	dst = cfg.XQuantization.appendNodes(dst)
	dst = cfg.OutputQuantization.appendNodes(dst)
	return dst
}

func (cfg QuantizedDenseConfig) cloneFromInputs(newInputs []*NodeImpl, idx *int) QuantizedDenseConfig {
	cloned := cfg
	cloned.WeightQuantization = cfg.WeightQuantization.cloneFromInputs(newInputs, idx)
	cloned.XQuantization = cfg.XQuantization.cloneFromInputs(newInputs, idx)
	cloned.OutputQuantization = cfg.OutputQuantization.cloneFromInputs(newInputs, idx)
	return cloned
}

// NodeInputsFusedQuantizedDense holds the inputs for compute.FusedQuantizedDense.
type NodeInputsFusedQuantizedDense struct {
	X      *NodeImpl
	Weight *NodeImpl
	Bias   *NodeImpl
	Cfg    QuantizedDenseConfig
}

func (ni *NodeInputsFusedQuantizedDense) Type() NodeType { return NodeTypeFusedQuantizedDense }

func (ni *NodeInputsFusedQuantizedDense) String() string {
	return fmt.Sprintf("%s(x=[#%d], weight=[#%d], bias=%s, wq=%s, xq=%s, oq=%s, quantizedMatmul=%v, activation=%+v, layout=%s)",
		ni.Type(),
		ni.X.Id(),
		ni.Weight.Id(),
		strNillableNode(ni.Bias),
		ni.Cfg.WeightQuantization,
		ni.Cfg.XQuantization,
		ni.Cfg.OutputQuantization,
		ni.Cfg.QuantizedMatmul,
		ni.Cfg.Activation,
		ni.Cfg.WeightLayout,
	)
}

func (ni *NodeInputsFusedQuantizedDense) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	idx := 0
	newX := newInputs[idx]
	idx++
	newWeight := newInputs[idx]
	idx++
	newCfg := ni.Cfg.cloneFromInputs(newInputs, &idx)
	var newBias *NodeImpl
	if ni.Bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	return BackendFusedQuantizedDense(newX, newWeight, newBias, newCfg)
}

// NodeInputsFusedLayerNormVJP holds the inputs for compute.FusedLayerNormVJP.
type NodeInputsFusedLayerNormVJP struct {
	X       *NodeImpl
	DOutput *NodeImpl
	Axes    []int
	Epsilon float64
	Gamma   *NodeImpl
	Beta    *NodeImpl
}

func (ni *NodeInputsFusedLayerNormVJP) Type() NodeType { return NodeTypeFusedLayerNormVJP }

func (ni *NodeInputsFusedLayerNormVJP) String() string {
	return fmt.Sprintf("%s(x=[#%d], dOutput=[#%d], axes=%v, epsilon=%v, gamma=%s, beta=%s)",
		ni.Type(),
		ni.X.Id(),
		ni.DOutput.Id(),
		ni.Axes,
		ni.Epsilon,
		strNillableNode(ni.Gamma),
		strNillableNode(ni.Beta),
	)
}

func (ni *NodeInputsFusedLayerNormVJP) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	idx := 0
	newX := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newGamma *NodeImpl
	if ni.Gamma != nil {
		newGamma = newInputs[idx]
		idx++
	}
	var newBeta *NodeImpl
	if ni.Beta != nil {
		newBeta = newInputs[idx]
		idx++
	}
	r0, _, _ := BackendFusedLayerNormVJP(newX, newDOutput, ni.Axes, ni.Epsilon, newGamma, newBeta)
	if r0.Type() == NodeTypeSplitNode {
		return r0.inputNodes[0]
	}
	return r0
}

// BackendFusedLayerNormVJP computes the VJP of FusedLayerNorm.
func BackendFusedLayerNormVJP(x, dOutput *NodeImpl, axes []int, epsilon float64, gamma, beta *NodeImpl) (dx, dGamma, dBeta *NodeImpl) {
	inputNodes := []*NodeImpl{x, dOutput}
	if gamma != nil {
		inputNodes = append(inputNodes, gamma)
	}
	if beta != nil {
		inputNodes = append(inputNodes, beta)
	}
	g := ValidateBuildingGraphFromInputs(inputNodes...)
	inputs := &NodeInputsFusedLayerNormVJP{
		X:       x,
		DOutput: dOutput,
		Axes:    slices.Clone(axes),
		Epsilon: epsilon,
		Gamma:   gamma,
		Beta:    beta,
	}
	var gammaVal, betaVal compute.Value
	if gamma != nil {
		gammaVal = gamma.outputOps[0]
	}
	if beta != nil {
		betaVal = beta.outputOps[0]
	}
	v0, v1, v2, err := g.currentFunc.backendFunc.FusedLayerNormVJP(
		x.outputOps[0], dOutput.outputOps[0], inputs.Axes, inputs.Epsilon, gammaVal, betaVal)
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

	node := &NodeImpl{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := SplitNode(node)
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

// NodeInputsFusedDenseVJP holds the inputs for compute.FusedDenseVJP.
type NodeInputsFusedDenseVJP struct {
	X       *NodeImpl
	Weight  *NodeImpl
	Bias    *NodeImpl
	Y       *NodeImpl
	DOutput *NodeImpl
	Options compute.DenseConfig
}

func (ni *NodeInputsFusedDenseVJP) Type() NodeType { return NodeTypeFusedDenseVJP }

func (ni *NodeInputsFusedDenseVJP) String() string {
	return fmt.Sprintf("%s(x=[#%d], weight=[#%d], bias=%s, y=[#%d], dOutput=[#%d], options=%+v)",
		ni.Type(),
		ni.X.Id(),
		ni.Weight.Id(),
		strNillableNode(ni.Bias),
		ni.Y.Id(),
		ni.DOutput.Id(),
		ni.Options,
	)
}

func (ni *NodeInputsFusedDenseVJP) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	idx := 0
	newX := newInputs[idx]
	idx++
	newWeight := newInputs[idx]
	idx++
	newY := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newBias *NodeImpl
	if ni.Bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	r0, _, _ := BackendFusedDenseVJP(newX, newWeight, newBias, newY, newDOutput, ni.Options)
	return r0.inputNodes[0]
}

// BackendFusedDenseVJP computes the VJP of FusedDense.
func BackendFusedDenseVJP(x, weight, bias, y, dOutput *NodeImpl, options compute.DenseConfig) (dx, dWeight, dBias *NodeImpl) {
	inputNodes := []*NodeImpl{x, weight, y, dOutput}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := ValidateBuildingGraphFromInputs(inputNodes...)
	inputs := &NodeInputsFusedDenseVJP{
		X:       x,
		Weight:  weight,
		Bias:    bias,
		Y:       y,
		DOutput: dOutput,
		Options: options,
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

	node := &NodeImpl{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := SplitNode(node)
	dx, dWeight = splitNodes[0], splitNodes[1]
	if bias != nil {
		dBias = splitNodes[2]
	}
	return
}

// BackendFusedQuantizedDense performs fused quantized matmul + optional bias + optional activation.
func BackendFusedQuantizedDense(x, weight, bias *NodeImpl, cfg QuantizedDenseConfig) *NodeImpl {
	if cfg.WeightQuantization == nil {
		exceptions.Panicf("BackendFusedQuantizedDense: cfg.WeightQuantization must not be nil")
	}
	wq := cfg.WeightQuantization
	if wq.Scale == nil && wq.Scheme != compute.QuantGGML {
		exceptions.Panicf("BackendFusedQuantizedDense: WeightQuantization.Scale must not be nil for scheme %s", wq.Scheme)
	}

	inputNodes := []*NodeImpl{x, weight}
	inputNodes = cfg.appendNodes(inputNodes)
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := ValidateBuildingGraphFromInputs(inputNodes...)

	idx := 2
	clonedCfg := cfg.cloneFromInputs(inputNodes, &idx)
	inputs := &NodeInputsFusedQuantizedDense{
		X:      x,
		Weight: weight,
		Bias:   bias,
		Cfg:    clonedCfg,
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
	node := &NodeImpl{
		outputOps:    []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return node
}

// NodeInputsFusedQuantizedDenseVJP holds the inputs for compute.FusedQuantizedDenseVJP.
type NodeInputsFusedQuantizedDenseVJP struct {
	X       *NodeImpl
	Weight  *NodeImpl
	Bias    *NodeImpl
	Y       *NodeImpl
	DOutput *NodeImpl
	Cfg     QuantizedDenseConfig
}

func (ni *NodeInputsFusedQuantizedDenseVJP) Type() NodeType { return NodeTypeFusedQuantizedDenseVJP }

func (ni *NodeInputsFusedQuantizedDenseVJP) String() string {
	return fmt.Sprintf("%s(x=%s, weight=[#%d], bias=%s, y=%s, dOutput=[#%d], wq=%s, quantizedMatmul=%v, activation=%+v, layout=%s)",
		ni.Type(),
		strNillableNode(ni.X),
		ni.Weight.Id(),
		strNillableNode(ni.Bias),
		strNillableNode(ni.Y),
		ni.DOutput.Id(),
		ni.Cfg.WeightQuantization,
		ni.Cfg.QuantizedMatmul,
		ni.Cfg.Activation,
		ni.Cfg.WeightLayout,
	)
}

func (ni *NodeInputsFusedQuantizedDenseVJP) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	idx := 0
	newWeight := newInputs[idx]
	idx++
	newDOutput := newInputs[idx]
	idx++
	var newX *NodeImpl
	if ni.X != nil {
		newX = newInputs[idx]
		idx++
	}
	var newY *NodeImpl
	if ni.Y != nil {
		newY = newInputs[idx]
		idx++
	}
	newCfg := ni.Cfg.cloneFromInputs(newInputs, &idx)
	var newBias *NodeImpl
	if ni.Bias != nil {
		newBias = newInputs[idx]
		idx++
	}
	r0, _ := BackendFusedQuantizedDenseVJP(newX, newWeight, newBias, newY, newDOutput, newCfg)
	if r0.Type() == NodeTypeSplitNode {
		return r0.inputNodes[0]
	}
	return r0
}

// BackendFusedQuantizedDenseVJP computes the VJP of FusedQuantizedDense.
func BackendFusedQuantizedDenseVJP(x, weight, bias, y, dOutput *NodeImpl, cfg QuantizedDenseConfig) (dx, dBias *NodeImpl) {
	if cfg.WeightQuantization == nil {
		exceptions.Panicf("BackendFusedQuantizedDenseVJP: cfg.WeightQuantization must not be nil")
	}
	inputNodes := []*NodeImpl{weight, dOutput}
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
	g := ValidateBuildingGraphFromInputs(inputNodes...)

	idx := cfgStartIdx
	clonedCfg := cfg.cloneFromInputs(inputNodes, &idx)
	inputs := &NodeInputsFusedQuantizedDenseVJP{
		X:       x,
		Weight:  weight,
		Bias:    bias,
		Y:       y,
		DOutput: dOutput,
		Cfg:     clonedCfg,
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

	node := &NodeImpl{
		outputOps:    outputOps,
		outputShapes: outputShapes,
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	splitNodes := SplitNode(node)
	dx = splitNodes[0]
	if bias != nil {
		dBias = splitNodes[1]
	}
	return
}

// NodeInputsQuantizedGather holds the inputs for compute.QuantizedGather.
type NodeInputsQuantizedGather struct {
	X       *NodeImpl
	Indices *NodeImpl
	TQ      *Quantization
}

func (ni *NodeInputsQuantizedGather) Type() NodeType { return NodeTypeQuantizedGather }

func (ni *NodeInputsQuantizedGather) String() string {
	return fmt.Sprintf("%s(x=[#%d], indices=[#%d], tq=%s)",
		ni.Type(), ni.X.Id(), ni.Indices.Id(), ni.TQ)
}

func (ni *NodeInputsQuantizedGather) CloneWithInputs(_ *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	idx := 2
	newTQ := ni.TQ.cloneFromInputs(newInputs, &idx)
	return BackendQuantizedGather(newInputs[0], newInputs[1], newTQ)
}

// BackendQuantizedGather performs a quantized row gather.
func BackendQuantizedGather(x, indices *NodeImpl, tq *Quantization) *NodeImpl {
	if tq != nil && tq.Scale == nil && tq.Scheme != compute.QuantGGML && tq.Scheme != compute.QuantNone {
		exceptions.Panicf("BackendQuantizedGather: Scale must not be nil for scheme %s", tq.Scheme)
	}

	inputNodes := []*NodeImpl{x, indices}
	inputNodes = tq.appendNodes(inputNodes)
	g := ValidateBuildingGraphFromInputs(inputNodes...)

	idx := 2
	clonedTQ := tq.cloneFromInputs(inputNodes, &idx)
	inputs := &NodeInputsQuantizedGather{
		X:       x,
		Indices: indices,
		TQ:      clonedTQ,
	}

	result, err := g.currentFunc.backendFunc.QuantizedGather(
		x.outputOps[0], indices.outputOps[0], clonedTQ.toBackend())
	if err != nil {
		panic(err)
	}
	node := &NodeImpl{
		outputOps:    []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return node
}

// FusedOpCaller attempts to call fused, falling back to decomposed on compute.ErrNotImplemented.
func FusedOpCaller(fused, decomposed func() *NodeImpl, useDecomposedAlternateVJP bool) (output *NodeImpl, opFused bool) {
	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed(), false
	}

	var decomposedOutput *NodeImpl
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
				klog.Infof("FusedOpCaller(): failed to call fused op (not implemented). Falling back to decomposed op: %v", err)
			}
			if decomposedOutput == nil {
				decomposedOutput = decomposed()
			}
			return decomposedOutput, false
		}
		panic(err)
	}
	klog.V(2).Info("FusedOpCaller(): Successfully called fused op.")
	if useDecomposedAlternateVJP {
		output.vjpAlternateOutputs = []*NodeImpl{decomposedOutput}
	}
	return output, true
}

// FusedOpCallerMulti is the multi-output counterpart of FusedOpCaller.
func FusedOpCallerMulti(fused, decomposed func() []*NodeImpl, useDecomposedAlternateVJP bool) (outputs []*NodeImpl, opFused bool) {
	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed(), false
	}

	var decomposedOutputs []*NodeImpl
	if useDecomposedAlternateVJP {
		decomposedOutputs = decomposed()
	}

	klog.V(2).Info("Attempt to call multi-output fused op")
	err := exceptions.TryCatch[error](func() {
		outputs = fused()
	})
	if err != nil {
		if compute.IsNotImplemented(err) {
			if klog.V(2).Enabled() {
				klog.Infof("FusedOpCallerMulti(): failed to call fused op (not implemented). Falling back to decomposed op: %v", err)
			}
			if decomposedOutputs == nil {
				decomposedOutputs = decomposed()
			}
			return decomposedOutputs, false
		}
		panic(err)
	}
	klog.V(2).Info("FusedOpCallerMulti(): Successfully called fused op.")

	if useDecomposedAlternateVJP {
		if len(outputs) != len(decomposedOutputs) {
			exceptions.Panicf("FusedOpCallerMulti: fused function returned %d outputs, but decomposed returned %d", len(outputs), len(decomposedOutputs))
		}
		for i, output := range outputs {
			output.vjpAlternateOutputs = []*NodeImpl{decomposedOutputs[i]}
		}
	}
	return outputs, true
}
