// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package graphbridge provides internal conversion helpers between the public
// core/graph wrapper types and the internal internal/graphimpl engine types,
// as well as internal graph operations used by higher-level GoMLX packages
// (such as ml/model, ml/nn, ml/layers, and core/autodiff).
package graphbridge

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// Node converts an internal *graphimpl.NodeImpl to its public *graph.Node wrapper.
func Node(impl *graphimpl.NodeImpl) *graph.Node {
	if impl == nil {
		return nil
	}
	res, _ := graphimpl.WrapNodeHook(impl).(*graph.Node)
	return res
}

// NodeImpl returns the underlying *graphimpl.NodeImpl for a public *graph.Node.
func NodeImpl(n *graph.Node) *graphimpl.NodeImpl {
	if n == nil {
		return nil
	}
	return graphimpl.UnwrapNodeHook(n)
}

// Nodes converts a slice of *graphimpl.NodeImpl to []*graph.Node.
func Nodes(impls []*graphimpl.NodeImpl) []*graph.Node {
	if len(impls) == 0 {
		return nil
	}
	res := make([]*graph.Node, len(impls))
	for i, impl := range impls {
		res[i] = Node(impl)
	}
	return res
}

// NodeImpls converts a slice of *graph.Node to []*graphimpl.NodeImpl.
func NodeImpls(nodes []*graph.Node) []*graphimpl.NodeImpl {
	if len(nodes) == 0 {
		return nil
	}
	res := make([]*graphimpl.NodeImpl, len(nodes))
	for i, n := range nodes {
		res[i] = NodeImpl(n)
	}
	return res
}

// Graph converts an internal *graphimpl.GraphImpl to its public *graph.Graph wrapper.
func Graph(impl *graphimpl.GraphImpl) *graph.Graph {
	if impl == nil {
		return nil
	}
	res, _ := graphimpl.WrapGraphHook(impl).(*graph.Graph)
	return res
}

// GraphImpl returns the underlying *graphimpl.GraphImpl for a public *graph.Graph.
func GraphImpl(g *graph.Graph) *graphimpl.GraphImpl {
	if g == nil {
		return nil
	}
	return graphimpl.UnwrapGraphHook(g)
}

// Function converts an internal *graphimpl.FunctionImpl to its public *graph.Function wrapper.
func Function(impl *graphimpl.FunctionImpl) *graph.Function {
	if impl == nil {
		return nil
	}
	res, _ := graphimpl.WrapFunctionHook(impl).(*graph.Function)
	return res
}

// FunctionImpl returns the underlying *graphimpl.FunctionImpl for a public *graph.Function.
func FunctionImpl(f *graph.Function) *graphimpl.FunctionImpl {
	if f == nil {
		return nil
	}
	return graphimpl.UnwrapFunctionHook(f)
}

// Exec converts an internal *graphimpl.ExecImpl to its public *graph.Exec wrapper.
func Exec(impl *graphimpl.ExecImpl) *graph.Exec {
	if impl == nil {
		return nil
	}
	res, _ := graphimpl.WrapExecHook(impl).(*graph.Exec)
	return res
}

// ExecImpl returns the underlying *graphimpl.ExecImpl for a public *graph.Exec.
func ExecImpl(e *graph.Exec) *graphimpl.ExecImpl {
	if e == nil {
		return nil
	}
	return graphimpl.UnwrapExecHook(e)
}

// QuantizationToImpl converts a public *graph.Quantization to *graphimpl.Quantization.
func QuantizationToImpl(q *graph.Quantization) *graphimpl.Quantization {
	if q == nil {
		return nil
	}
	return &graphimpl.Quantization{
		Scheme:         q.Scheme,
		Scale:          NodeImpl(q.Scale),
		ZeroPoint:      NodeImpl(q.ZeroPoint),
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
		PackedDType:    q.PackedDType,
	}
}

// QuantizationFromImpl converts a *graphimpl.Quantization back to *graph.Quantization.
func QuantizationFromImpl(q *graphimpl.Quantization) *graph.Quantization {
	if q == nil {
		return nil
	}
	return &graph.Quantization{
		Scheme:         q.Scheme,
		Scale:          Node(q.Scale),
		ZeroPoint:      Node(q.ZeroPoint),
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
		PackedDType:    q.PackedDType,
	}
}

// QuantizedDenseConfigToImpl converts a public graph.QuantizedDenseConfig to graphimpl.QuantizedDenseConfig.
func QuantizedDenseConfigToImpl(cfg graph.QuantizedDenseConfig) graphimpl.QuantizedDenseConfig {
	return graphimpl.QuantizedDenseConfig{
		WeightQuantization: QuantizationToImpl(cfg.WeightQuantization),
		XQuantization:      QuantizationToImpl(cfg.XQuantization),
		OutputQuantization: QuantizationToImpl(cfg.OutputQuantization),
		QuantizedMatmul:    cfg.QuantizedMatmul,
		Activation:         cfg.Activation,
		WeightLayout:       cfg.WeightLayout,
	}
}

// QuantizedDenseConfigFromImpl converts a graphimpl.QuantizedDenseConfig back to graph.QuantizedDenseConfig.
func QuantizedDenseConfigFromImpl(cfg graphimpl.QuantizedDenseConfig) graph.QuantizedDenseConfig {
	return graph.QuantizedDenseConfig{
		WeightQuantization: QuantizationFromImpl(cfg.WeightQuantization),
		XQuantization:      QuantizationFromImpl(cfg.XQuantization),
		OutputQuantization: QuantizationFromImpl(cfg.OutputQuantization),
		QuantizedMatmul:    cfg.QuantizedMatmul,
		Activation:         cfg.Activation,
		WeightLayout:       cfg.WeightLayout,
	}
}

// FusedOpCaller invokes fused and falls back to decomposed if the backend returns ErrNotImplemented
// or if fusion is disabled via graph.FusionEnv.
func FusedOpCaller(fused, decomposed func() *graph.Node, useDecomposedAlternateVJP bool) (output *graph.Node, opFused bool) {
	out, ok := graphimpl.FusedOpCaller(
		func() *graphimpl.NodeImpl { return NodeImpl(fused()) },
		func() *graphimpl.NodeImpl { return NodeImpl(decomposed()) },
		useDecomposedAlternateVJP,
	)
	return Node(out), ok
}

// FusedOpCallerMulti is the multi-output variant of FusedOpCaller.
func FusedOpCallerMulti(fused, decomposed func() []*graph.Node, useDecomposedAlternateVJP bool) (outputs []*graph.Node, opFused bool) {
	outs, ok := graphimpl.FusedOpCallerMulti(
		func() []*graphimpl.NodeImpl { return NodeImpls(fused()) },
		func() []*graphimpl.NodeImpl { return NodeImpls(decomposed()) },
		useDecomposedAlternateVJP,
	)
	return Nodes(outs), ok
}

// BackendOutputs returns the underlying backend compute.Value slice for a Node.
func BackendOutputs(n *graph.Node) []compute.Value {
	impl := NodeImpl(n)
	if impl == nil {
		return nil
	}
	return impl.OutputOps()
}

// ConstBuffer creates a Constant node in g backed by buf.
func ConstBuffer(g *graph.Graph, buf compute.Buffer, donate bool) *graph.Node {
	return Node(graphimpl.ConstBuffer(GraphImpl(g), buf, donate))
}

// SharedConstant creates a Constant node in g reusing a backend constant compute.Value.
func SharedConstant(g *graph.Graph, val compute.Value) *graph.Node {
	return Node(graphimpl.SharedConstant(GraphImpl(g), val))
}

// FinalizeConstantValue releases backend resources held by a shared constant compute.Value.
func FinalizeConstantValue(val compute.Value) {
	graphimpl.FinalizeConstantValue(val)
}

// BatchNormForTraining wraps the backend BatchNormForTraining operation.
func BatchNormForTraining(operand, scale, offset *graph.Node, epsilon float32, axis int) (normalized, batchMean, batchVariance *graph.Node) {
	norm, mean, variance := graphimpl.BatchNormForTraining(NodeImpl(operand), NodeImpl(scale), NodeImpl(offset), epsilon, axis)
	return Node(norm), Node(mean), Node(variance)
}

// BatchNormForInference wraps the backend BatchNormForInference operation.
func BatchNormForInference(operand, scale, offset, mean, variance *graph.Node, epsilon float32, axis int) *graph.Node {
	return Node(graphimpl.BatchNormForInference(NodeImpl(operand), NodeImpl(scale), NodeImpl(offset), NodeImpl(mean), NodeImpl(variance), epsilon, axis))
}

// BatchNormGradient wraps the backend BatchNormGradient operation.
func BatchNormGradient(operand, scale, mean, variance, gradOutput *graph.Node, epsilon float32, axis int) (gradOperand, gradScale, gradOffset *graph.Node) {
	gop, gsc, goff := graphimpl.BatchNormGradient(NodeImpl(operand), NodeImpl(scale), NodeImpl(mean), NodeImpl(variance), NodeImpl(gradOutput), epsilon, axis)
	return Node(gop), Node(gsc), Node(goff)
}

// FusedActivation invokes BackendFusedActivation on the underlying GraphImpl.
func FusedActivation(x *graph.Node, cfg compute.ActivationConfig) *graph.Node {
	return Node(graphimpl.BackendFusedActivation(NodeImpl(x), cfg))
}

// FusedLayerNorm invokes BackendFusedLayerNorm on the underlying GraphImpl.
func FusedLayerNorm(x *graph.Node, axes []int, epsilon float64, gamma, beta *graph.Node) *graph.Node {
	return Node(graphimpl.BackendFusedLayerNorm(NodeImpl(x), axes, epsilon, NodeImpl(gamma), NodeImpl(beta)))
}

// FusedDense invokes BackendFusedDense on the underlying GraphImpl.
func FusedDense(x, weight, bias *graph.Node, options compute.DenseConfig) *graph.Node {
	return Node(graphimpl.BackendFusedDense(NodeImpl(x), NodeImpl(weight), NodeImpl(bias), options))
}

// FusedQuantizedDense invokes BackendFusedQuantizedDense on the underlying GraphImpl.
func FusedQuantizedDense(x, weight, bias *graph.Node, cfg graph.QuantizedDenseConfig) *graph.Node {
	return Node(graphimpl.BackendFusedQuantizedDense(NodeImpl(x), NodeImpl(weight), NodeImpl(bias), QuantizedDenseConfigToImpl(cfg)))
}

// QuantizedGather invokes BackendQuantizedGather on the underlying GraphImpl.
func QuantizedGather(x, indices *graph.Node, tq *graph.Quantization) *graph.Node {
	return Node(graphimpl.BackendQuantizedGather(NodeImpl(x), NodeImpl(indices), QuantizationToImpl(tq)))
}

// FusedAttentionQKVProjection invokes BackendFusedAttentionQKVProjection on the underlying GraphImpl.
func FusedAttentionQKVProjection(x, wQKV, biasQ, biasK, biasV *graph.Node, queryDim, keyValueDim int) (query, key, value *graph.Node) {
	q, k, v := graphimpl.BackendFusedAttentionQKVProjection(NodeImpl(x), NodeImpl(wQKV), NodeImpl(biasQ), NodeImpl(biasK), NodeImpl(biasV), queryDim, keyValueDim)
	return Node(q), Node(k), Node(v)
}

// FusedScaledDotProductAttention invokes BackendFusedScaledDotProductAttention on the underlying GraphImpl.
func FusedScaledDotProductAttention(query, key, value *graph.Node, axesLayout compute.AttentionAxesLayout, options *compute.ScaledDotProductAttentionConfig) (output *graph.Node, statesForVJP []*graph.Node) {
	out, states := graphimpl.BackendFusedScaledDotProductAttention(NodeImpl(query), NodeImpl(key), NodeImpl(value), axesLayout, options)
	return Node(out), Nodes(states)
}

// Executable returns the compiled compute.Executable of a compiled Graph.
func Executable(g *graph.Graph) compute.Executable {
	return GraphImpl(g).Executable()
}

// AttachState attaches auxiliary state keyed by key to the Graph.
func AttachState(g *graph.Graph, key, value any) {
	GraphImpl(g).AttachState(key, value)
}

// State retrieves auxiliary state attached to the Graph for key.
func State(g *graph.Graph, key any) any {
	return GraphImpl(g).State(key)
}
