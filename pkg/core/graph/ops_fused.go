// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Hand-written fused-op wrappers for ops with nil-able Value parameters
// (FusedLayerNorm, FusedDense, FusedDenseActivation).
// FusedSoftmax and FusedGelu have simple signatures and are auto-generated
// in gen_backend_ops.go.

// nodeInputsFusedLayerNorm holds the inputs for a LayerNorm node.
type nodeInputsFusedLayerNorm struct {
	x       *Node
	axes    []int
	epsilon float64
	gamma   *Node
	beta    *Node
}

func (ni *nodeInputsFusedLayerNorm) Type() NodeType {
	return NodeTypeFusedLayerNorm
}

func (ni *nodeInputsFusedLayerNorm) String() string {
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

// nodeInputsFusedDense holds the inputs for a Dense node.
type nodeInputsFusedDense struct {
	x      *Node
	weight *Node
	bias   *Node
}

func (ni *nodeInputsFusedDense) Type() NodeType {
	return NodeTypeFusedDense
}

func (ni *nodeInputsFusedDense) String() string {
	biasStr := "nil"
	if ni.bias != nil {
		biasStr = fmt.Sprintf("[#%d]", ni.bias.Id())
	}
	return fmt.Sprintf("Dense(x=[#%d], weight=[#%d], bias=%s)",
		ni.x.Id(), ni.weight.Id(), biasStr)
}

// nodeInputsFusedDenseActivation holds the inputs for a DenseActivation node.
type nodeInputsFusedDenseActivation struct {
	x          *Node
	weight     *Node
	bias       *Node
	activation backends.ActivationType
}

func (ni *nodeInputsFusedDenseActivation) Type() NodeType {
	return NodeTypeFusedDenseActivation
}

func (ni *nodeInputsFusedDenseActivation) String() string {
	biasStr := "nil"
	if ni.bias != nil {
		biasStr = fmt.Sprintf("[#%d]", ni.bias.Id())
	}
	return fmt.Sprintf("DenseActivation(x=[#%d], weight=[#%d], bias=%s, activation=%s)",
		ni.x.Id(), ni.weight.Id(), biasStr, ni.activation)
}

// FusedLayerNorm wraps the backend LayerNorm call, handling nil gamma/beta.
func FusedLayerNorm(x *Node, axes []int, epsilon float64, gamma, beta *Node) *Node {
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

	result, err := g.currentFunc.backendFunc.FusedLayerNorm(x.outputOps[0], slices.Clone(axes), epsilon, gammaVal, betaVal)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsFusedLayerNorm{x: x, axes: slices.Clone(axes), epsilon: epsilon, gamma: gamma, beta: beta}
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

// FusedDense wraps the backend Dense call, handling nil bias.
func FusedDense(x, weight, bias *Node) *Node {
	inputNodes := []*Node{x, weight}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	var biasVal backends.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.FusedDense(x.outputOps[0], weight.outputOps[0], biasVal)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsFusedDense{x: x, weight: weight, bias: bias}
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

// FusedDenseActivation wraps the backend DenseActivation call, handling nil bias.
func FusedDenseActivation(x, weight, bias *Node, activation backends.ActivationType) *Node {
	inputNodes := []*Node{x, weight}
	if bias != nil {
		inputNodes = append(inputNodes, bias)
	}
	g := validateBuildingGraphFromInputs(inputNodes...)

	var biasVal backends.Value
	if bias != nil {
		biasVal = bias.outputOps[0]
	}

	result, err := g.currentFunc.backendFunc.FusedDenseActivation(x.outputOps[0], weight.outputOps[0], biasVal, activation)
	if err != nil {
		panic(err)
	}

	inputs := &nodeInputsFusedDenseActivation{x: x, weight: weight, bias: bias, activation: activation}
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
