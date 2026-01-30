// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Hand-written wrappers for fused ops with nil-able Value parameters
// (LayerNorm, Linear, LinearActivation).
// Softmax and Gelu have simple signatures and are auto-generated as
// exported wrappers in gen_backend_ops.go.

// nodeInputsLayerNorm holds the inputs for a LayerNorm node.
type nodeInputsLayerNorm struct {
	x       *Node
	axes    []int
	epsilon float64
	gamma   *Node
	beta    *Node
}

func (ni *nodeInputsLayerNorm) Type() NodeType {
	return NodeTypeLayerNorm
}

func (ni *nodeInputsLayerNorm) String() string {
	gammaStr, betaStr := "nil", "nil"
	if ni.gamma != nil {
		gammaStr = fmt.Sprintf("[#%d]", ni.gamma.Id())
	}
	if ni.beta != nil {
		betaStr = fmt.Sprintf("[#%d]", ni.beta.Id())
	}
	return fmt.Sprintf("LayerNorm(x=[#%d], axes=%v, epsilon=%v, gamma=%s, beta=%s)",
		ni.x.Id(), ni.axes, ni.epsilon, gammaStr, betaStr)
}

// nodeInputsLinear holds the inputs for a Linear node.
type nodeInputsLinear struct {
	x      *Node
	weight *Node
	bias   *Node
}

func (ni *nodeInputsLinear) Type() NodeType {
	return NodeTypeLinear
}

func (ni *nodeInputsLinear) String() string {
	biasStr := "nil"
	if ni.bias != nil {
		biasStr = fmt.Sprintf("[#%d]", ni.bias.Id())
	}
	return fmt.Sprintf("Linear(x=[#%d], weight=[#%d], bias=%s)",
		ni.x.Id(), ni.weight.Id(), biasStr)
}

// nodeInputsLinearActivation holds the inputs for a LinearActivation node.
type nodeInputsLinearActivation struct {
	x          *Node
	weight     *Node
	bias       *Node
	activation backends.ActivationType
}

func (ni *nodeInputsLinearActivation) Type() NodeType {
	return NodeTypeLinearActivation
}

func (ni *nodeInputsLinearActivation) String() string {
	biasStr := "nil"
	if ni.bias != nil {
		biasStr = fmt.Sprintf("[#%d]", ni.bias.Id())
	}
	return fmt.Sprintf("LinearActivation(x=[#%d], weight=[#%d], bias=%s, activation=%s)",
		ni.x.Id(), ni.weight.Id(), biasStr, ni.activation)
}

// LayerNorm wraps the backend LayerNorm call, handling nil gamma/beta.
// It calls the backend's native LayerNorm op directly — the caller must check
// Capabilities().Operations[backends.OpTypeLayerNorm] before calling.
func LayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
	inputNodes := []*Node{x}
	if gamma != nil {
		inputNodes = append(inputNodes, gamma)
	}
	if beta != nil {
		inputNodes = append(inputNodes, beta)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	var gammaVal, betaVal backends.Value
	if gamma != nil {
		gammaVal = gamma.outputOps[0]
	}
	if beta != nil {
		betaVal = beta.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.LayerNorm(x.outputOps[0], slices.Clone(axes), epsilon, gammaVal, betaVal)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsLayerNorm{x: x, axes: slices.Clone(axes), epsilon: epsilon, gamma: gamma, beta: beta}
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

// Linear wraps the backend Linear call, handling nil bias.
// It calls the backend's native Linear op directly — the caller must check
// Capabilities().Operations[backends.OpTypeLinear] before calling.
func Linear(x, weight, bias *Node) *Node {
	inputNodes := []*Node{x, weight}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	var biasVal backends.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.Linear(x.outputOps[0], weight.outputOps[0], biasVal)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsLinear{x: x, weight: weight, bias: bias}
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

// LinearActivation wraps the backend LinearActivation call, handling nil bias.
// It calls the backend's native LinearActivation op directly — the caller must check
// Capabilities().Operations[backends.OpTypeLinearActivation] before calling.
func LinearActivation(x, weight, bias *Node, activation backends.ActivationType) *Node {
	inputNodes := []*Node{x, weight}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	var biasVal backends.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.LinearActivation(x.outputOps[0], weight.outputOps[0], biasVal, activation)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsLinearActivation{x: x, weight: weight, bias: bias, activation: activation}
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
