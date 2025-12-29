package graph

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// backend_dynamic_bounds.go contains backend wrappers for dynamic operations with bounds.
// These are separate from gen_backend_ops.go to avoid being overwritten by code generation.

// nodeInputsDynamicReshapeWithBounds holds the inputs used for the call to backends.DynamicReshapeWithBounds.
type nodeInputsDynamicReshapeWithBounds struct {
	operand     *Node
	outputShape *Node
	bounds      []int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsDynamicReshapeWithBounds) Type() NodeType {
	return NodeTypeDynamicReshape // Reuse the same node type since it's a variant
}

// InputNodes implements the interface NodeInputs.
func (ni *nodeInputsDynamicReshapeWithBounds) InputNodes() []*Node {
	return []*Node{ni.operand, ni.outputShape}
}

// String implements the interface NodeInputs.
func (ni *nodeInputsDynamicReshapeWithBounds) String() string {
	return fmt.Sprintf("%s(operand=[#%d], outputShape=[#%d], bounds=%v)",
		ni.Type(),
		ni.operand.Id(),
		ni.outputShape.Id(),
		ni.bounds,
	)
}

// backendDynamicReshapeWithBounds is a Graph wrapper for the backend.Builder.DynamicReshapeWithBounds method.
func backendDynamicReshapeWithBounds(operand *Node, outputShape *Node, bounds []int) (node *Node) {
	inputNodes := []*Node{operand, outputShape}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsDynamicReshapeWithBounds{
		operand:     operand,
		outputShape: outputShape,
		bounds:      bounds,
	}
	result, err := g.builder.DynamicReshapeWithBounds(operand.outputOps[0], outputShape.outputOps[0], bounds)
	if err != nil {
		panic(err)
	}
	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return
}

// nodeInputsDynamicBroadcastInDimWithBounds holds the inputs for DynamicBroadcastInDimWithBounds.
type nodeInputsDynamicBroadcastInDimWithBounds struct {
	operand             *Node
	outputDimensions    *Node
	broadcastDimensions []int
	bounds              []int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) Type() NodeType {
	return NodeTypeDynamicBroadcastInDim // Reuse the same node type since it's a variant
}

// InputNodes implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) InputNodes() []*Node {
	return []*Node{ni.operand, ni.outputDimensions}
}

// String implements the interface NodeInputs.
func (ni *nodeInputsDynamicBroadcastInDimWithBounds) String() string {
	return fmt.Sprintf("%s(operand=[#%d], outputDimensions=[#%d], broadcastDimensions=%v, bounds=%v)",
		ni.Type(),
		ni.operand.Id(),
		ni.outputDimensions.Id(),
		ni.broadcastDimensions,
		ni.bounds,
	)
}

// backendDynamicBroadcastInDimWithBounds is a Graph wrapper for the backend.Builder.DynamicBroadcastInDimWithBounds method.
func backendDynamicBroadcastInDimWithBounds(operand *Node, outputDimensions *Node, broadcastDimensions []int, bounds []int) (node *Node) {
	inputNodes := []*Node{operand, outputDimensions}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsDynamicBroadcastInDimWithBounds{
		operand:             operand,
		outputDimensions:    outputDimensions,
		broadcastDimensions: broadcastDimensions,
		bounds:              bounds,
	}
	result, err := g.builder.DynamicBroadcastInDimWithBounds(operand.outputOps[0], outputDimensions.outputOps[0], broadcastDimensions, bounds)
	if err != nil {
		panic(err)
	}
	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return
}
