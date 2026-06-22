// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// NodeTypeCustomCall is the NodeType for CustomCall operations.
// (1004 = OptimizationBarrier; these manual NodeTypes live outside the generated range.)
const NodeTypeCustomCall NodeType = 1005

// nodeInputsCustomCall holds the inputs for a CustomCall op.
type nodeInputsCustomCall struct {
	operands []*Node
	spec     compute.CustomCallSpec
}

// Type implements NodeInputs.
func (ni *nodeInputsCustomCall) Type() NodeType { return NodeTypeCustomCall }

// String implements NodeInputs.
func (ni *nodeInputsCustomCall) String() string { return "CustomCall(" + ni.spec.Target + ")" }

// CustomCall emits a backend-specific StableHLO custom_call (see compute.Function.CustomCall):
// an escape hatch to a target named at runtime (e.g. cuDNN flash attention, "__cudnn$fmhaSoftmax").
// It is multi-output, returning one node per spec.OutputShapes entry.
//
// A custom_call is opaque to autodiff, so it has no default gradient. Pass a non-nil vjpFn that
// returns the gradient for each operand (typically by emitting the target's backward custom_call);
// pass nil for a non-differentiable call, which then panics if differentiated.
//
// Backends without custom-call support (e.g. SimpleGo) panic with a wrapped compute.ErrNotImplemented,
// so callers can recover and fall back to a decomposed implementation.
func CustomCall(spec compute.CustomCallSpec, vjpFn VJP, operands ...*Node) []*Node {
	if len(operands) == 0 {
		exceptions.Panicf("CustomCall requires at least one operand")
	}
	g := operands[0].graph
	g.AssertBuilding()
	validateBuildingGraphFromInputs(operands...)

	inputValues := make([]compute.Value, len(operands))
	for i, o := range operands {
		inputValues[i] = o.outputOps[0]
	}
	results, err := g.currentFunc.backendFunc.CustomCall(spec, inputValues...)
	if err != nil {
		panic(errors.WithMessagef(err, "CustomCall(%q) operation failed", spec.Target))
	}

	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}

	node := &Node{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       &nodeInputsCustomCall{operands: operands, spec: spec},
		inputNodes:   operands,
		scope:        g.currentFunc,
	}
	node.customVJP = vjpFn
	g.registerNode(node)
	return splitNode(node)
}

// customCallVJP is only reached when a differentiable path runs through a CustomCall built
// without a vjpFn (a node with a vjpFn uses that customVJP directly, ahead of this registration).
func customCallVJP(node *Node, _ []*Node, _ shapes.Shape) []*Node {
	exceptions.Panicf("CustomCall(%q) has no gradient: pass a non-nil vjpFn to graph.CustomCall to make it differentiable",
		node.inputs.(*nodeInputsCustomCall).spec.Target)
	return nil
}

func init() {
	VJPRegistration[NodeTypeCustomCall] = customCallVJP
}
