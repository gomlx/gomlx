// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// NodeType constants for fused operations.
const (
	NodeTypeFusedSoftmax          NodeType = 1000
	NodeTypeFusedLayerNorm        NodeType = 1001
	NodeTypeFusedGelu             NodeType = 1002
	NodeTypeFusedLinear           NodeType = 1003
	NodeTypeFusedLinearActivation NodeType = 1004
)

// BackendFunction returns the backend Function for the current scope of the graph.
// This can be used by external packages to access the FusedOps interface via type assertion.
func (g *Graph) BackendFunction() backends.Function {
	return g.currentFunc.backendFunc
}

// BackendValue returns the primary backend value for this node.
// This is used by fused op implementations that need to pass values to backend methods.
func (n *Node) BackendValue() backends.Value {
	return n.outputOps[0]
}

// InsertNode creates and registers a new graph node from a backend result value.
// This is used by fused op implementations to insert a fused node into the graph.
func InsertNode(g *Graph, result backends.Value, inputs NodeInputs, inputNodes ...*Node) *Node {
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

// nodeInputsFusedSoftmax holds the inputs for a fused softmax node.
type nodeInputsFusedSoftmax struct {
	x    *Node
	axis int
}

func (ni *nodeInputsFusedSoftmax) Type() NodeType {
	return NodeTypeFusedSoftmax
}

func (ni *nodeInputsFusedSoftmax) String() string {
	return fmt.Sprintf("FusedSoftmax(x=[#%d], axis=%d)", ni.x.Id(), ni.axis)
}

// nodeInputsFusedGelu holds the inputs for a fused GELU node.
type nodeInputsFusedGelu struct {
	x    *Node
	mode string
}

func (ni *nodeInputsFusedGelu) Type() NodeType {
	return NodeTypeFusedGelu
}

func (ni *nodeInputsFusedGelu) String() string {
	return fmt.Sprintf("FusedGelu(x=[#%d], mode=%s)", ni.x.Id(), ni.mode)
}

// TryFusedGelu attempts to use a fused GELU op if the backend supports it.
// Returns the result node and true if successful, or nil and false if the backend
// does not support fused GELU (caller should fall back to decomposition).
func TryFusedGelu(x *Node, mode string) (*Node, bool) {
	g := x.Graph()
	if g.Backend().Capabilities().FusedOperations[backends.FusedOpGelu] {
		if fusedOps, ok := g.BackendFunction().(backends.FusedOps); ok {
			result, err := fusedOps.Gelu(x.BackendValue(), mode)
			if err == nil {
				node := InsertNode(g, result, &nodeInputsFusedGelu{x: x, mode: mode}, x)
				return node, true
			}
		}
	}
	return nil, false
}
