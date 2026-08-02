package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
)

// ReshapeDimensionSpec specifies how the dimension of an axis is to be reshaped.
// It is used with DynamiceReshape, and can be created with one of:
//
//   - StaticDim(int): the dimension is a constant known at graph building time.
//
//   - DynamicDim(*Node): the dimension is given by the input node, and the resulting axis has a dynamic dimension
//     that doesn't match anything statically (during build time).
//
//   - NamedDynamicDim(name, *Node): the dimension is given by the input node and is named.
//
//   - InferredDim(): the dimension is inferred from the input node with the given name. There can only be one
//     axis whose dimension is automatically inferred.
//
//   - NamedInferredDim(name): the dimension is inferred from the input node and named. There can only be one
//     axis whose dimension is automatically inferred.
type ReshapeDimensionSpec struct {
	static   int
	axisName string
	dynamic  *Node
}

// StaticDim specifies a static dimension. The dimension is a constant known at graph building time.
//
// Use this with DynamicReshape.
func StaticDim(dim int) ReshapeDimensionSpec {
	return ReshapeDimensionSpec{static: dim}
}

// DynamicDim specifies a dynamic dimension whose value is given by the input node.
//
// Use this with DynamicReshape.
func DynamicDim(dim *Node) ReshapeDimensionSpec {
	return ReshapeDimensionSpec{axisName: shapes.AnonymousAxis, dynamic: dim}
}

// NamedDynamicDim specifies a dynamic dimension whose value is given by the input node and is named.
//
// Use this with DynamicReshape.
func NamedDynamicDim(name string, dim *Node) ReshapeDimensionSpec {
	return ReshapeDimensionSpec{axisName: name, dynamic: dim}
}

// InferredDim specifies that the dimension of the axis is inferred from the input node with the same name.
// There can be only one axis with inferred dimension.
//
// Use this with DynamicReshape.
func InferredDim() ReshapeDimensionSpec {
	return ReshapeDimensionSpec{axisName: shapes.AnonymousAxis}
}

// NamedInferredDim specifies that the dimension of the axis is inferred from the input node with the given name.
// There can be only one axis with inferred dimension.
//
// Use this with DynamicReshape.
func NamedInferredDim(name string) ReshapeDimensionSpec {
	return ReshapeDimensionSpec{axisName: name}
}

// DynamicReshape reshapes the operand according to dimension specifications for each axis.
// Because the shape is not known in graph building time, it is only supported in backends that
// support dynamic axes (shapes), see backend.Capabilities().
//
// Each axis can have its dimension specified in one of 4 ways:
//
//   - StaticDim(dim): the dimension is a constant known at graph building time.
//   - DynamicDim(name, dim): the dimension is given by the input node with a given name that should always resolve
//     to the same value for a particular execution of the graph.
//   - AnonymousDynamicDim(dim) creates a dynamic dimension that is not named (name is generated automatically)
//     for dimensions that are not reused in the graph.
//   - InferredDim(name): the dimension is inferred from the input node with the given name. There can only be one
//     axis whose dimension is automatically inferred.
//
// Example: Flattening a dynamically shaped input with rank > 2, while preserving the dynamic "batch_size" axis:
//
//	batchSizeNode := Slice(DynamicShape(inputs)
//	batchSizeSpec := NamedDynamicDim("batch_size", batchSizeNode)
//	flatInputs := DynamicReshape(batchSizeSpec, NamedInferredDim("features"))
func DynamicReshape(operand *Node, axisSpecs ...ReshapeDimensionSpec) *Node {
	inputsToValidate := make([]*Node, 1, 1+len(axisSpecs))
	inputsToValidate[0] = operand
	for _, spec := range axisSpecs {
		if spec.dynamic != nil {
			inputsToValidate = append(inputsToValidate, spec.dynamic)
		}
	}
	_ = validateBuildingGraphFromInputs(inputsToValidate...)

	dimensions := make([]compute.DynamicDimensionSpec, len(axisSpecs))
	for i, spec := range axisSpecs {
		var name string
		var static int
		var value compute.Value

		if spec.dynamic != nil {
			name = spec.axisName
			value = spec.dynamic.outputOps[0]
		} else if spec.axisName != "" {
			name = spec.axisName
		} else {
			static = spec.static
		}

		dimensions[i] = compute.DynamicDimensionSpec{
			Static: static,
			Name:   name,
			Value:  value,
		}
	}
	return backendDynamicReshape(operand, dimensions...)
}
