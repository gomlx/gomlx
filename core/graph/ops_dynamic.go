package graph

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/support/exceptions"
)

// DimensionSpec specifies how the dimension of an axis is defined in dynamic shape operations
// like DynamicReshape and DynamicBroadcastInDim.
//
// It can be created with one of:
//   - StaticDim(int): the dimension is a constant known at graph building time.
//   - DynamicDim(*Node): the dimension is given by the input node, and the resulting axis has an anonymous dynamic dimension.
//   - NamedDynamicDim(name, *Node): the dimension is given by the input node and is named.
//   - InferredDim(): the dimension is automatically inferred to preserve total tensor size (can ONLY be used for at most one axis in DynamicReshape, and nowhere else).
//   - NamedInferredDim(name): the dimension is automatically inferred and named (can ONLY be used for at most one axis in DynamicReshape, and nowhere else).
//   - DimensionSpecFor(x, axis): extracts the dimension specification for the given axis of x.
//   - DimensionSpecsFor(x): extracts dimension specifications for all axes of x.
type DimensionSpec struct {
	static   int
	axisName string
	dynamic  *Node
}

// IsStatic returns true if the dimension is a constant known at graph building time.
func (s DimensionSpec) IsStatic() bool { return s.dynamic == nil && s.static > 0 }

// IsDynamic returns true if the dimension is given by a dynamic *Node.
func (s DimensionSpec) IsDynamic() bool { return s.dynamic != nil }

// IsInferred returns true if the dimension is automatically inferred.
func (s DimensionSpec) IsInferred() bool { return s.dynamic == nil && s.static <= 0 }

// Static returns the static dimension size, or shapes.DynamicDim (-1) if dynamic or inferred.
func (s DimensionSpec) Static() int {
	if s.IsStatic() {
		return s.static
	}
	return shapes.DynamicDim
}

// Dynamic returns the *Node representing the dynamic dimension, or nil if static or inferred.
func (s DimensionSpec) Dynamic() *Node { return s.dynamic }

// AxisName returns the name associated with this dimension (or "" / shapes.AnonymousAxis).
func (s DimensionSpec) AxisName() string { return s.axisName }

// WithName returns a copy of the DimensionSpec with the specified axis name.
func (s DimensionSpec) WithName(name string) DimensionSpec {
	s.axisName = name
	return s
}

// Clone returns a copy of the DimensionSpec.
func (s DimensionSpec) Clone() DimensionSpec { return s }

// String implements fmt.Stringer for nice debug and error printing.
func (s DimensionSpec) String() string {
	if s.IsDynamic() {
		if s.axisName != "" && s.axisName != shapes.AnonymousAxis {
			return fmt.Sprintf("DynamicDim(%q, %s)", s.axisName, s.dynamic)
		}
		return fmt.Sprintf("DynamicDim(%s)", s.dynamic)
	}
	if s.IsInferred() {
		if s.axisName != "" && s.axisName != shapes.AnonymousAxis {
			return fmt.Sprintf("InferredDim(%q)", s.axisName)
		}
		return "InferredDim()"
	}
	if s.axisName != "" {
		return fmt.Sprintf("StaticDim(%q, %d)", s.axisName, s.static)
	}
	return fmt.Sprintf("StaticDim(%d)", s.static)
}

// ReshapeDimensionSpec is an alias to DimensionSpec for backwards compatibility.
//
// Deprecated: use DimensionSpec instead.
type ReshapeDimensionSpec = DimensionSpec

// StaticDim specifies a static dimension. The dimension is a constant known at graph building time.
//
// Use this with DynamicReshape, DynamicBroadcastInDim, and DynamicIota.
func StaticDim(dim int) DimensionSpec {
	return DimensionSpec{static: dim}
}

// DynamicDim specifies a dynamic dimension whose value is given by the input node.
//
// Use this with DynamicReshape, DynamicBroadcastInDim, and DynamicIota.
func DynamicDim(dim *Node) DimensionSpec {
	return DimensionSpec{axisName: shapes.AnonymousAxis, dynamic: dim}
}

// NamedDynamicDim specifies a dynamic dimension whose value is given by the input node and is named.
//
// Use this with DynamicReshape, DynamicBroadcastInDim, and DynamicIota.
func NamedDynamicDim(name string, dim *Node) DimensionSpec {
	return DimensionSpec{axisName: name, dynamic: dim}
}

// InferredDim specifies that the dimension of the axis is automatically inferred to preserve
// the total size of the tensor.
//
// Note: Inferred dimensions can ONLY be used for at most one axis in DynamicReshape.
// They cannot be used with DynamicBroadcast operations or DynamicIota.
func InferredDim() DimensionSpec {
	return DimensionSpec{axisName: shapes.AnonymousAxis}
}

// NamedInferredDim specifies that the dimension of the axis is automatically inferred and named
// to preserve the total size of the tensor.
//
// Note: Inferred dimensions can ONLY be used for at most one axis in DynamicReshape.
// They cannot be used with DynamicBroadcast operations or DynamicIota.
func NamedInferredDim(name string) DimensionSpec {
	return DimensionSpec{axisName: name}
}

// DimensionSpecFor returns the DimensionSpec corresponding to the given axis of x.
// If the axis has a static dimension, it returns StaticDim(dim).
// If the axis has a dynamic dimension, it extracts DimensionSize(x, axis) and returns NamedDynamicDim(name, sizeNode).
func DimensionSpecFor(x *Node, axis int) DimensionSpec {
	_ = validateBuildingGraphFromInputs(x)
	axis = MustAdjustAxis(axis, x)
	dim := x.Shape().Dimensions[axis]
	name := x.Shape().AxisName(axis)
	if dim == shapes.DynamicDim {
		return NamedDynamicDim(name, DimensionSize(x, axis))
	}
	return StaticDim(dim)
}

// DimensionSpecsFor returns a slice of DimensionSpecs for all axes of x.
// Static dimensions produce StaticDim, and dynamic dimensions produce NamedDynamicDim with their runtime *Node values.
func DimensionSpecsFor(x *Node) []DimensionSpec {
	_ = validateBuildingGraphFromInputs(x)
	specs := make([]DimensionSpec, x.Rank())
	for axis := range x.Rank() {
		specs[axis] = DimensionSpecFor(x, axis)
	}
	return specs
}

// DimensionSize returns a scalar Int32 *Node representing the dimension size of the given axis.
// If the dimension is static, it returns a constant Scalar node.
// If the dimension is dynamic, it queries the backend for the dynamic dimension size.
func DimensionSize(x *Node, axis int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	axis = MustAdjustAxis(axis, x)
	dim := x.Shape().Dimensions[axis]
	if dim == shapes.DynamicDim {
		return backendDynamicDimensionSize(x, axis)
	}
	return Scalar(x.Graph(), dtypes.Int32, dim)
}

// DynamicReshape reshapes the operand according to dimension specifications for each axis.
// Because the shape is not known in graph building time, it is only supported in backends that
// support dynamic axes (shapes), see backend.Capabilities().
//
// If both the operand and all of the axis specs have static dimensions, this uses the static Reshape instead.
//
// Each axis can have its dimension specified in one of:
//
//   - StaticDim(dim): the dimension is a constant known at graph building time.
//   - DynamicDim(dimNode): the dimension is given by the scalar node dimNode (anonymous dynamic axis).
//   - NamedDynamicDim(name, dimNode): the dimension is given by the scalar node dimNode and named.
//   - InferredDim(): the dimension is automatically inferred from the total size of the operand.
//   - NamedInferredDim(name): the dimension is automatically inferred and named.
//
// At most one axis can be inferred (`InferredDim()` or `NamedInferredDim(name)`).
//
// Example: Flattening a dynamically shaped input with rank > 2, while preserving the dynamic "batch_size" axis:
//
//	batchSizeNode := DynamicDimensionSize(inputs, 0)
//	batchSizeSpec := NamedDynamicDim("batch_size", batchSizeNode)
//	flatInputs := DynamicReshape(inputs, batchSizeSpec, NamedInferredDim("features"))
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

// DynamicReshapeLike reshapes the operand to the same shape (and axis names) as the reference node.
// The DType of reference is ignored; the returned node will retain operand's DType.
// It works seamlessly for both static and dynamic shapes.
func DynamicReshapeLike(operand, reference *Node) *Node {
	specs := DimensionSpecsFor(reference)
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
		} else if spec.static > 0 {
			name = spec.axisName
			static = spec.static
		} else {
			exceptions.Panicf("DynamicBroadcastInDim: axis %d requires a positive Static dimension or a dynamic *Node (inferred dimensions are not supported in broadcast; use DynamicBroadcastLike or DynamicDim(*Node))", i)
		}

		dimensions[i] = compute.DynamicDimensionSpec{
			Static: static,
			Name:   name,
			Value:  value,
		}
	}
	return backendDynamicBroadcastInDim(operand, broadcastAxes, dimensions...)
}

// DynamicBroadcastLike broadcasts the operand to the shape of the reference node along the specified broadcastAxes.
// If broadcastAxes is omitted, it defaults to trailing (NumPy-style) axes.
//
// It works seamlessly for both static and dynamic shapes (extracting dynamic sizes automatically).
func DynamicBroadcastLike(operand, reference *Node, broadcastAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(operand, reference)
	refRank := reference.Rank()
	opRank := operand.Rank()

	if len(broadcastAxes) == 0 {
		if opRank > refRank {
			exceptions.Panicf("DynamicBroadcastLike: operand rank %d > reference rank %d", opRank, refRank)
		}
		diff := refRank - opRank
		broadcastAxes = make([]int, opRank)
		for i := range opRank {
			broadcastAxes[i] = diff + i
		}
	}

	specs := DimensionSpecsFor(reference)
	return DynamicBroadcastInDim(operand, broadcastAxes, specs...)
}

// DynamicBroadcastToShape broadcasts x to the given shape (which can be static or dynamic).
// x must have an equal or lower rank than shape, and if shape has more dimensions than x rank,
// x will be expanded at the end (so new axes will be appended to x).
// Dimensions of x must either match the corresponding dimension in shape, or they must be 1, in which case they are broadcast.
//
// Dynamic shape rules for target axes:
//  1. Preserved dynamic axis: If operand x at axis i is already dynamic (shapes.DynamicDim),
//     the dynamic size is taken directly from x (DynamicDimensionSize(x, i)). The target dynamic axis must have a compatible name (or empty).
//  2. Broadcast 1 -> Dynamic: Broadcasting dimension 1 to a dynamic dimension is not supported by DynamicBroadcastToShape,
//     because no reference tensor is available to provide the runtime dynamic size. Use BroadcastLike or DynamicBroadcastLike
//     with a reference node instead, or DynamicBroadcastInDim with explicit DimensionSpec(*Node) values.
//  3. Static dimension > 1 to Dynamic: An exception is raised.
//
// If neither shape nor x is dynamic, it delegates to BroadcastToShape.
// Conversely, BroadcastToShape delegates to DynamicBroadcastToShape if shape or x is dynamic.
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
		targetDim := shape.Dimensions[i]
		targetName := shape.AxisName(i)
		if targetDim != shapes.DynamicDim {
			if i < xShape.Rank() {
				xDim := xShape.Dimensions[i]
				if xDim == shapes.DynamicDim {
					exceptions.Panicf("DynamicBroadcastToShape: cannot broadcast dynamic axis %d (%q) of %s to static dimension %d",
						i, xShape.AxisName(i), xShape, targetDim)
				}
				if xDim != targetDim && xDim != 1 {
					exceptions.Panicf("DynamicBroadcastToShape: incompatible dimensions at axis %d: x has %d, target has %d (shapes %s vs %s)",
						i, xDim, targetDim, xShape, shape)
				}
			}
			specs[i] = StaticDim(targetDim)
		} else {
			// targetDim == shapes.DynamicDim
			if i < xShape.Rank() && xShape.Dimensions[i] == shapes.DynamicDim {
				// Rule 1: Preserved dynamic axis from x.
				unifiedName, err := shapes.UnifyAxisName(xShape.AxisName(i), targetName)
				if err != nil {
					exceptions.Panicf("DynamicBroadcastToShape: incompatible dynamic axis names at axis %d: %v", i, err)
				}
				dimVal := DimensionSize(x, i)
				specs[i] = NamedDynamicDim(unifiedName, dimVal)
			} else if i >= xShape.Rank() || xShape.Dimensions[i] == 1 {
				// Rule 2: Cannot broadcast dimension 1 to dynamic without reference node.
				exceptions.Panicf("DynamicBroadcastToShape: cannot broadcast axis %d (dimension 1) to dynamic dimension (use BroadcastLike, DynamicBroadcastLike, or DynamicBroadcastInDim with a reference *Node instead)", i)
			} else {
				// Rule 3: Static dimension > 1 to Dynamic.
				exceptions.Panicf("DynamicBroadcastToShape: cannot broadcast static dimension %d (> 1) at axis %d of %s to dynamic target dimension",
					xShape.Dimensions[i], i, xShape)
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

// DynamicIota creates a tensor with the given dynamic dimensions and dtype, filled with
// increasing numbers (starting from 0) along the specified iotaAxis.
//
// If all dimension specs are static, it delegates to static Iota.
func DynamicIota(g *Graph, dtype dtypes.DType, iotaAxis int, axisSpecs ...DimensionSpec) *Node {
	g.AssertBuilding()
	if len(axisSpecs) == 0 {
		exceptions.Panicf("DynamicIota requires at least one dimension spec")
	}

	inputsToValidate := make([]*Node, 0, len(axisSpecs))
	allStatic := true
	staticDims := make([]int, len(axisSpecs))
	for i, spec := range axisSpecs {
		if spec.dynamic != nil {
			allStatic = false
			inputsToValidate = append(inputsToValidate, spec.dynamic)
		} else if spec.axisName != "" {
			allStatic = false
		} else {
			staticDims[i] = spec.static
		}
	}

	if len(inputsToValidate) > 0 {
		_ = validateBuildingGraphFromInputs(inputsToValidate...)
	}

	if allStatic {
		return Iota(g, shapes.Make(dtype, staticDims...), iotaAxis)
	}

	dimensions := make([]compute.DynamicDimensionSpec, len(axisSpecs))
	for i, spec := range axisSpecs {
		var name string
		var static int
		var value compute.Value

		if spec.dynamic != nil {
			name = spec.axisName
			value = spec.dynamic.outputOps[0]
		} else if spec.static > 0 {
			name = spec.axisName
			static = spec.static
		} else {
			exceptions.Panicf("DynamicIota: axis %d requires a positive Static dimension or a dynamic *Node (inferred dimensions are not supported in iota; use DynamicDim(*Node))", i)
		}

		dimensions[i] = compute.DynamicDimensionSpec{
			Static: static,
			Name:   name,
			Value:  value,
		}
	}

	return backendDynamicIota(g, dtype, iotaAxis, dimensions...)
}

// DynamicPadAxisSpec specifies the padding configuration for an axis in DynamicPad.
// It can specify static integer padding amounts and/or dynamic scalar *Node amounts.
type DynamicPadAxisSpec struct {
	Start, End, Interior             int
	StartNode, EndNode, InteriorNode *Node
	TargetAxisName                   string
}

// DynamicPadAxis creates a DynamicPadAxisSpec with static padding amounts.
func DynamicPadAxis(start, end, interior int) DynamicPadAxisSpec {
	return DynamicPadAxisSpec{
		Start:    start,
		End:      end,
		Interior: interior,
	}
}

// DynamicPad injects padding on the start, end, or interior of the given operand using static
// and/or dynamic scalar padding amounts.
//
// If all padding specifications and operand shape are static, it delegates to static Pad.
func DynamicPad(x, fillValue *Node, axesConfig ...DynamicPadAxisSpec) *Node {
	inputNodes := []*Node{x, fillValue}
	for _, cfg := range axesConfig {
		if cfg.StartNode != nil {
			inputNodes = append(inputNodes, cfg.StartNode)
		}
		if cfg.EndNode != nil {
			inputNodes = append(inputNodes, cfg.EndNode)
		}
		if cfg.InteriorNode != nil {
			inputNodes = append(inputNodes, cfg.InteriorNode)
		}
	}
	_ = validateBuildingGraphFromInputs(inputNodes...)

	isAllStatic := !x.Shape().IsDynamic()
	for _, cfg := range axesConfig {
		if cfg.StartNode != nil || cfg.EndNode != nil || cfg.InteriorNode != nil || cfg.TargetAxisName != "" {
			isAllStatic = false
			break
		}
	}

	if isAllStatic {
		staticConfigs := make([]compute.PadAxis, len(axesConfig))
		for i, cfg := range axesConfig {
			staticConfigs[i] = compute.PadAxis{
				Start:    cfg.Start,
				End:      cfg.End,
				Interior: cfg.Interior,
			}
		}
		return Pad(x, fillValue, staticConfigs...)
	}

	dynConfigs := make([]compute.DynamicPadAxis, len(axesConfig))
	for i, cfg := range axesConfig {
		var startVal, endVal, interiorVal compute.Value
		if cfg.StartNode != nil {
			startVal = cfg.StartNode.outputOps[0]
		}
		if cfg.EndNode != nil {
			endVal = cfg.EndNode.outputOps[0]
		}
		if cfg.InteriorNode != nil {
			interiorVal = cfg.InteriorNode.outputOps[0]
		}
		dynConfigs[i] = compute.DynamicPadAxis{
			Start:          cfg.Start,
			End:            cfg.End,
			Interior:       cfg.Interior,
			StartValue:     startVal,
			EndValue:       endVal,
			InteriorValue:  interiorVal,
			TargetAxisName: cfg.TargetAxisName,
		}
	}

	return backendDynamicPad(x, fillValue, dynConfigs...)
}
