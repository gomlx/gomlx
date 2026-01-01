package graph

import (
	"fmt"
	"slices"

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

// backendDynamicReshapeWithBoundsAndShape is like backendDynamicReshapeWithBounds but allows
// explicitly setting the output shape for GoMLX shape propagation. This is needed when we want
// to propagate extracted concrete dimensions to downstream operations while still using dynamic
// reshape for XLA compilation.
func backendDynamicReshapeWithBoundsAndShape(operand *Node, outputShapeTensor *Node, bounds []int, outputDims []int) (node *Node) {
	// CRITICAL FIX: We MUST use dynamic reshape if the operand has symbolic dimensions,
	// even if the output dimensions are all concrete. This is because XLA may track the
	// operand as dynamic (due to bounded dynamic dimensions from earlier operations),
	// and static reshape doesn't support dynamic inputs.
	operandHasSymbolic := operand.Shape().HasSymbolicDim()

	// Check if all output dimensions are concrete (non-negative).
	allOutputConcrete := true
	for _, d := range outputDims {
		if d < 0 {
			allOutputConcrete = false
			break
		}
	}

	// CRITICAL FIX: Also check if bounds differ from outputDims. If they do, it means
	// this tensor uses bounded dynamic dimensions (physical != logical), so XLA tracks it
	// as dynamic internally. We MUST use DynamicReshape in that case.
	hasBoundedDynamic := false
	if len(bounds) == len(outputDims) {
		for i := range bounds {
			if outputDims[i] > 0 && bounds[i] > 0 && outputDims[i] != bounds[i] {
				hasBoundedDynamic = true
				break
			}
		}
	}

	// Only use static reshape if operand is fully concrete, output is fully concrete, AND no bounded dynamic
	if !operandHasSymbolic && allOutputConcrete && !hasBoundedDynamic {
		// All dimensions are concrete - verify the sizes match before using static reshape
		outputSize := 1
		for _, d := range outputDims {
			outputSize *= d
		}

		// Get operand size, using absolute value if dynamic
		operandSize := operand.Shape().Size()
		if operandSize < 0 {
			operandSize = -operandSize
		}

		// Only use static reshape if sizes match OR operand has unknown size
		if operandSize > 0 && operandSize != outputSize {
			// Fall through to dynamic path below
			allOutputConcrete = false
		}
	}

	if !operandHasSymbolic && allOutputConcrete && !hasBoundedDynamic {
		// All dimensions are concrete AND sizes match AND no bounded dynamic - use static Reshape at XLA level
		// Only validate operand, not outputShapeTensor which we won't use
		g := validateBuildingGraphFromInputs(operand)

		// Create proper node inputs for static reshape
		ni := &nodeInputsReshape{
			x:          operand,
			dimensions: slices.Clone(outputDims),
		}

		result, err := g.builder.Reshape(operand.outputOps[0], outputDims...)
		if err != nil {
			panic(err)
		}
		outputShape := shapes.Make(operand.DType(), outputDims...)
		// For static reshape, use nodeInputsReshape like regular Reshape does
		node = &Node{
			outputOps:    []backends.Op{result},
			outputShapes: []shapes.Shape{outputShape},
			graph:        g,
			inputs:       ni,
			inputNodes:   []*Node{operand}, // Only the operand
		}
		g.registerNode(node)
		return
	}

	// Some dimensions are dynamic - use DynamicReshapeWithBounds
	inputNodes := []*Node{operand, outputShapeTensor}
	g := validateBuildingGraphFromInputs(inputNodes...)
	inputs := &nodeInputsDynamicReshapeWithBounds{
		operand:     operand,
		outputShape: outputShapeTensor,
		bounds:      bounds,
	}
	result, err := g.builder.DynamicReshapeWithBounds(operand.outputOps[0], outputShapeTensor.outputOps[0], bounds)
	if err != nil {
		panic(err)
	}
	outputShape := shapes.MakeDynamic(operand.DType(), outputDims...)
	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{outputShape},
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

// backendDynamicBroadcastInDimWithBoundsAndShape is like backendDynamicBroadcastInDimWithBounds but allows
// explicitly setting the output shape for GoMLX shape propagation.
func backendDynamicBroadcastInDimWithBoundsAndShape(operand *Node, outputDimensions *Node, broadcastDimensions []int, bounds []int, outputDims []int) (node *Node) {
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
	// Get the shape from the XLA backend result, which tracks logical dimensions
	// when they differ from physical (bounds) dimensions.
	outputShape, err := g.builder.OpShape(result)
	if err != nil {
		// Fallback to computing shape from outputDims if OpShape fails
		hasDynamic := false
		for _, d := range outputDims {
			if d < 0 {
				hasDynamic = true
				break
			}
		}
		if hasDynamic {
			outputShape = shapes.MakeDynamic(operand.DType(), outputDims...)
		} else {
			outputShape = shapes.Make(operand.DType(), outputDims...)
		}
	}

	// CRITICAL FIX: When using DynamicBroadcastInDimWithBounds, XLA creates a bounded dynamic
	// dimension even if the logical dimension is concrete. We MUST mark the output as symbolic
	// so that downstream operations (like Reshape) use dynamic variants instead of static ones.
	// Without this, Reshape will see concrete dims and use static reshape, but XLA will complain
	// that the input is actually dynamic (bounded dynamic).
	//
	// Check if any dimension uses bounded dynamic (logical != physical)
	needsSymbolic := false
	for i, d := range outputDims {
		if d > 0 && i < len(bounds) && d != bounds[i] {
			// This dimension is bounded dynamic: logical size (d) != physical size (bounds[i])
			needsSymbolic = true
			break
		}
	}

	if needsSymbolic {
		// Convert concrete shape to symbolic to signal downstream ops to use dynamic variants
		symbolicDims := make([]int, len(outputDims))
		for i, d := range outputDims {
			if d > 0 && i < len(bounds) && d != bounds[i] {
				// Mark as symbolic by using negative value (XLA tracks this as bounded dynamic)
				symbolicDims[i] = -1
			} else {
				symbolicDims[i] = d
			}
		}
		outputShape = shapes.MakeDynamic(operand.DType(), symbolicDims...)
	}

	node = &Node{
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{outputShape},
		graph:        g,
		inputs:       inputs,
		inputNodes:   inputNodes,
	}
	g.registerNode(node)
	return
}
