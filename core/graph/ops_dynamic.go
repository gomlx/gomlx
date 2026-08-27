package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
)

// DimensionSpec specifies how the dimension of an axis is defined in dynamic shape operations
// like DynamicReshape and DynamicBroadcastInDim.
//
// It can be created with one of:
//   - StaticDim(int): the dimension is a constant known at graph building time.
//   - DynamicDim(*Node): the dimension is given by the input node, and the resulting axis has an anonymous dynamic dimension.
//   - NamedDynamicDim(name, *Node): the dimension is given by the input node and is named.
//   - InferredDim(): the dimension is inferred from the input node (only supported in DynamicReshape).
//   - NamedInferredDim(name): the dimension is inferred and named, or refers to an already known named dynamic axis.
type DimensionSpec struct {
	static   int
	axisName string
	dynamic  *Node
}

// ReshapeDimensionSpec is an alias to DimensionSpec for backwards compatibility.
//
// Deprecated: use DimensionSpec instead.
type ReshapeDimensionSpec = DimensionSpec

// StaticDim specifies a static dimension. The dimension is a constant known at graph building time.
//
// Use this with DynamicReshape and DynamicBroadcastInDim.
func StaticDim(dim int) DimensionSpec {
	return DimensionSpec{static: dim}
}

// DynamicDim specifies a dynamic dimension whose value is given by the input node.
//
// Use this with DynamicReshape and DynamicBroadcastInDim.
func DynamicDim(dim *Node) DimensionSpec {
	return DimensionSpec{axisName: shapes.AnonymousAxis, dynamic: dim}
}

// NamedDynamicDim specifies a dynamic dimension whose value is given by the input node and is named.
//
// Use this with DynamicReshape and DynamicBroadcastInDim.
func NamedDynamicDim(name string, dim *Node) DimensionSpec {
	return DimensionSpec{axisName: name, dynamic: dim}
}

// InferredDim specifies that the dimension of the axis is inferred from the input node with the same name.
// There can be only one axis with inferred dimension in DynamicReshape.
//
// Use this with DynamicReshape.
func InferredDim() DimensionSpec {
	return DimensionSpec{axisName: shapes.AnonymousAxis}
}

// NamedInferredDim specifies that the dimension of the axis is inferred from the input node with the given name,
// or refers to an existing known dynamic axis.
//
// Use this with DynamicReshape or DynamicBroadcastInDim.
func NamedInferredDim(name string) DimensionSpec {
	return DimensionSpec{axisName: name}
}

// DynamicReshape reshapes the operand according to dimension specifications for each axis.
// Because the shape is not known in graph building time, it is only supported in backends that
// support dynamic axes (shapes), see backend.Capabilities().
//
// If both the operand and all of the axis specs have static dimensions, this uses the static Reshape instead.
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
func DynamicReshape(operand *Node, axisSpecs ...DimensionSpec) *Node {
	inputsToValidate := make([]*Node, 1, 1+len(axisSpecs))
	inputsToValidate[0] = operand
	for _, spec := range axisSpecs {
		if spec.dynamic != nil {
			inputsToValidate = append(inputsToValidate, spec.dynamic)
		}
	}
	_ = validateBuildingGraphFromInputs(inputsToValidate...)

	// Check if operand and all axisSpecs are completely static (or inferred from a static operand shape).
	if !operand.Shape().IsDynamic() {
		staticDims := make([]int, len(axisSpecs))
		allStatic := true
		inferredIdx := -1
		for i, spec := range axisSpecs {
			if spec.dynamic != nil {
				allStatic = false
				break
			}
			if spec.static > 0 {
				staticDims[i] = spec.static
			} else if spec.static == 0 {
				// Inferred dimension spec (static == 0 and dynamic == nil)
				if inferredIdx != -1 {
					allStatic = false // More than one inferred dim
					break
				}
				inferredIdx = i
				staticDims[i] = -1
			} else {
				allStatic = false
				break
			}
		}
		if allStatic {
			return Reshape(operand, staticDims...)
		}
	}

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

// DynamicReshapeLike reshapes the operand to the same dynamic shape (and axis names) as the reference node.
// The DType of reference is ignored; the returned node will retain operand's DType.
func DynamicReshapeLike(operand, reference *Node) *Node {
	refShape := reference.Shape()
	refRank := refShape.Rank()
	specs := make([]DimensionSpec, refRank)

	if !operand.Shape().IsDynamic() && !refShape.IsDynamic() {
		// Both are static.
		return ReshapeWithShape(operand, reference.Shape())
	}

	for axis := range refRank {
		dim := refShape.Dimensions[axis]
		name := refShape.AxisName(axis)
		if dim != shapes.DynamicDim {
			// Static dimension.
			specs[axis] = StaticDim(dim)
		} else {
			// Dynamic dimension.
			dimSizeNode := DynamicDimensionSize(reference, axis)
			specs[axis] = NamedDynamicDim(name, dimSizeNode)
		}
	}
	return DynamicReshape(operand, specs...)
}

// DynamicBroadcastInDim broadcasts the operand to an output with target dimensions specified by axisSpecs.
//
// broadcastAxes maps operand axes to the corresponding output axes (len(broadcastAxes) == operand.Shape().Rank()),
// where the i-th axis of operand is mapped to broadcastAxes[i]-th dimension of the output.
// broadcastAxes must also be strictly increasing: this operation cannot be used to transpose axes.
//
// If both the operand and all axisSpecs are completely static, this falls back to static BroadcastInDim instead.
func DynamicBroadcastInDim(operand *Node, broadcastAxes []int, axisSpecs ...DimensionSpec) *Node {
	inputsToValidate := make([]*Node, 1, 1+len(axisSpecs))
	inputsToValidate[0] = operand
	for _, spec := range axisSpecs {
		if spec.dynamic != nil {
			inputsToValidate = append(inputsToValidate, spec.dynamic)
		}
	}
	_ = validateBuildingGraphFromInputs(inputsToValidate...)

	// Check if operand and all axisSpecs are completely static.
	if !operand.Shape().IsDynamic() {
		staticDims := make([]int, len(axisSpecs))
		allStatic := true
		for i, spec := range axisSpecs {
			if spec.dynamic != nil || spec.static <= 0 {
				allStatic = false
				break
			}
			staticDims[i] = spec.static
		}
		if allStatic {
			outShape := shapes.Make(operand.DType(), staticDims...)
			return backendBroadcastInDim(operand, outShape, broadcastAxes)
		}
	}

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
	return backendDynamicBroadcastInDim(operand, broadcastAxes, dimensions...)
}

// DynamicBroadcastLike broadcasts the operand to the dynamic shape of the reference node along the specified broadcastAxes.
// If broadcastAxes is omitted, it defaults to trailing (NumPy-style) axes.
func DynamicBroadcastLike(operand, reference *Node, broadcastAxes ...int) *Node {
	refShape := reference.Shape()
	refRank := refShape.Rank()
	if len(broadcastAxes) == 0 {
		if operand.Rank() > refRank {
			exceptions.Panicf("cannot broadcast operand with rank %d to reference with rank %d", operand.Rank(), refRank)
		}
		offset := refRank - operand.Rank()
		broadcastAxes = make([]int, operand.Rank())
		for i := range broadcastAxes {
			broadcastAxes[i] = offset + i
		}
	}

	if !operand.Shape().IsDynamic() && !refShape.IsDynamic() {
		return backendBroadcastInDim(operand, refShape, broadcastAxes)
	}

	specs := make([]DimensionSpec, refRank)
	for axis := range refRank {
		dim := refShape.Dimensions[axis]
		name := refShape.AxisName(axis)
		if dim != shapes.DynamicDim {
			specs[axis] = StaticDim(dim)
		} else {
			dimSizeNode := DynamicDimensionSize(reference, axis)
			specs[axis] = NamedDynamicDim(name, dimSizeNode)
		}
	}
	return DynamicBroadcastInDim(operand, broadcastAxes, specs...)
}

// DynamicBroadcastToShape broadcasts x to the given shape (which can be static or dynamic).
// x must have an equal or lower rank than shape, and if shape has more dimensions than x rank,
// x will be expanded at the end (so new axes will be appended to x).
// Dimensions of x must either match the corresponding dimension in shape, or they must be 1, in which case they are broadcast.
//
// If shape is not dynamic (and x is not dynamic), it calls BroadcastToShape.
// Conversely, BroadcastToShape calls DynamicBroadcastToShape if shape or x is dynamic.
func DynamicBroadcastToShape(x *Node, shape shapes.Shape) *Node {
	if !x.Shape().IsDynamic() && !shape.IsDynamic() {
		return BroadcastToShape(x, shape)
	}

	_ = validateBuildingGraphFromInputs(x)
	xShape := x.Shape()
	if xShape.Rank() > shape.Rank() {
		exceptions.Panicf("DynamicBroadcastToShape: rank mismatch: x shape %s has rank %d, target shape %s has rank %d",
			xShape, xShape.Rank(), shape, shape.Rank())
	}

	specs := make([]DimensionSpec, shape.Rank())
	for i := range shape.Rank() {
		dim := shape.Dimensions[i]
		name := shape.AxisName(i)
		if dim != shapes.DynamicDim {
			specs[i] = StaticDim(dim)
		} else {
			if i < xShape.Rank() && xShape.Dimensions[i] == shapes.DynamicDim {
				dimVal := DynamicDimensionSize(x, i)
				specs[i] = NamedDynamicDim(name, dimVal)
			} else if name != "" {
				specs[i] = NamedInferredDim(name)
			} else {
				specs[i] = InferredDim()
			}
		}
	}

	broadcastDims := make([]int, xShape.Rank())
	for ii := range xShape.Rank() {
		broadcastDims[ii] = ii
	}
	return DynamicBroadcastInDim(x, broadcastDims, specs...)
}

// DynamicBroadcastToDims broadcasts operand to the target dimensions using prefix axis alignment
// (appending new axes at the end), matching BroadcastToDims.
func DynamicBroadcastToDims(operand *Node, axisSpecs ...DimensionSpec) *Node {
	outputRank := len(axisSpecs)
	if operand.Rank() > outputRank {
		exceptions.Panicf("DynamicBroadcastToDims: rank mismatch: operand rank %d > target rank %d", operand.Rank(), outputRank)
	}
	broadcastAxes := make([]int, operand.Rank())
	for i := range operand.Rank() {
		broadcastAxes[i] = i
	}
	return DynamicBroadcastInDim(operand, broadcastAxes, axisSpecs...)
}

