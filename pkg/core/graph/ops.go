package graph

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// PadAxis defines the amount of padding preceding one axis (Start), at the end of axis (End)
// or in between the inputNodes (Interior).
// This is used as a parameter for the Pad function.
// This is an alias to backends.PadAxis
type PadAxis = backends.PadAxis

// nodeInputsParameter holds the inputs used for the call to backends.Parameter.
type nodeInputsParameter struct {
	name     string
	shape    shapes.Shape
	sharding *distributed.ShardingSpec
	handle   ParameterHandle
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsParameter) Type() NodeType {
	return NodeTypeParameter
}

// String implements the interface NodeInputs.
func (ni *nodeInputsParameter) String() string {
	return fmt.Sprintf("%s(name=%q, shape=%s)", ni.Type(), ni.name, ni.shape)
}

// mustNoError converts an error to a panic.
func mustNoError[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// Parameter registers an input parameter for a computation Graph (e.g: a feature used as input).
//
// When created they get a handle (a plain index) but they can also be accessed
// It can be used in two different ways: as a Node when building the Graph, so when defining a
// function that uses the parameter, or as the key in the map of the inputNodes when executing
// the computation Graph (see Backend.RunWithMap).
func Parameter(g *Graph, name string, shape shapes.Shape) (node *Node) {
	return ShardedParameter(g, name, shape, nil)
}

// ShardedParameter works like Parameter but takes as an optional (it can be nil) argument the sharding
// specification of how this input is going to be fed across devices.
// This is used for distributed computations -- see Graph.SetAutoSharding.
func ShardedParameter(g *Graph, name string, shape shapes.Shape, sharding *distributed.ShardingSpec) (node *Node) {
	g.AssertBuilding()
	if sharding != nil && g.distStrategy == distributed.None {
		exceptions.Panicf(
			"cannot set sharding spec for parameter %q in graph %q when distributed strategy is %q",
			name, g.name, g.distStrategy,
		)
	}
	if sharding != nil && slices.Index(g.deviceMeshes, sharding.Mesh) == -1 {
		exceptions.Panicf(
			"sharding spec for parameter %q in graph %q specifies mesh %q that was not registered with the "+
				"Graph -- all meshes used in sharding specifications must be declared in the Graph upfront when "+
				"configuring AutoSharding",
			name, g.name, sharding.Mesh,
		)
	}
	handle := ParameterHandle(len(g.parameters))
	if name == "" {
		name = fmt.Sprintf("parameter_#%d", handle)
	}
	if _, ok := g.parameterNameToHandle[name]; ok {
		exceptions.Panicf("requested parameter with name %q for graph %q already exists", name, g.name)
	}
	nodeInputs := &nodeInputsParameter{
		name:     name,
		shape:    shape,
		sharding: sharding, // it can be nil.
		handle:   handle,
	}
	result, err := g.builder.Parameter(nodeInputs.name, nodeInputs.shape, sharding.ToBackendsSpec())
	if err != nil {
		panic(errors.WithMessagef(err, "failed to create parameter %q", name))
	}
	node = &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       nodeInputs,
	}
	g.registerNode(node)
	g.parameters = append(g.parameters, node)
	g.parameterNameToHandle[name] = handle
	return
}

// nodeInputsSplitNode holds the inputs used for the call to backends.Parameter.
type nodeInputsSplitNode struct {
	multiOutputNode *Node
	index           int
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsSplitNode) Type() NodeType {
	return NodeTypeSplitNode
}

// String implements the interface NodeInputs.
func (ni *nodeInputsSplitNode) String() string {
	return fmt.Sprintf("%s(multiOutputNode=[#%d], index=%d)", ni.Type(), ni.multiOutputNode.Id(), ni.index)
}

// splitNode splits a Node that has multiple outputs into multiple nodes with one output only.
// Internal only: users should only need to handle nodes with one output, so any method that returns many outputs
// must call splitNode before returning to the user.
func splitNode(multiOutputNode *Node) (splitNodes []*Node) {
	if multiOutputNode.NumOutputs() == 1 {
		// Identity: no need to split.
		// This happens when an operation returns a slice of outputs, but the slice contains only one element.
		return []*Node{multiOutputNode}
	}
	if multiOutputNode.NumOutputs() == 0 {
		exceptions.Panicf("splitNode expects at least one node as input -- got %d outputs",
			multiOutputNode.NumOutputs())
	}
	g := multiOutputNode.Graph()
	g.AssertBuilding()
	splitNodes = make([]*Node, 0, multiOutputNode.NumOutputs())
	for ii, op := range multiOutputNode.outputOps {
		inputs := &nodeInputsSplitNode{
			multiOutputNode: multiOutputNode,
			index:           ii,
		}
		inputNodes := []*Node{multiOutputNode}
		node := &Node{
			outputOps:    []backends.Op{op},
			outputShapes: []shapes.Shape{multiOutputNode.outputShapes[ii]},
			graph:        g,
			inputs:       inputs,
			inputNodes:   inputNodes,
		}
		g.registerNode(node)
		splitNodes = append(splitNodes, node)
	}
	return
}

// MinConstValueSizeToKeep defines a size below which constant values (see Const, ConstTensor) are kept in the Node/Graph
// for printing/debugging purposes
//
// If set to 0, no value is kept.
var MinConstValueSizeToKeep = 32

// nodeInputsConstant holds the inputs used for the call to backends.Parameter.
type nodeInputsConstant struct {
	shape  shapes.Shape
	tensor *tensors.Tensor // Only saved for values < MinConstValueSizeToKeep
}

// Type implements the interface NodeInputs.
func (ni *nodeInputsConstant) Type() NodeType {
	return NodeTypeConstant
}

// String implements the interface NodeInputs.
func (ni *nodeInputsConstant) String() string {
	if ni.tensor == nil {
		return fmt.Sprintf("%s(%s)", ni.Type(), ni.shape)
	} else {
		return fmt.Sprintf("%s(%s: %v)", ni.Type(), ni.shape, ni.tensor.Value())
	}
}

// ConstTensor returns a newly created constant node for the tensor t.
//
// The value of t is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
//
// See also ConstCachedTensor if you think you'll use the same tensor multiple times in the same graph.
func ConstTensor(g *Graph, t *tensors.Tensor) (node *Node) {
	g.AssertBuilding()
	nodeInputs := &nodeInputsConstant{
		shape: t.Shape(),
	}
	if t.Size() < MinConstValueSizeToKeep {
		var err error
		nodeInputs.tensor, err = t.LocalClone()
		if err != nil {
			panic(errors.WithMessagef(err,
				"ConstTensor failed to create a local clone of the tensor in the graph"))
		}
	}
	var result backends.Op
	var err error
	t.MustConstFlatData(func(flat any) {
		result, err = g.builder.Constant(flat, nodeInputs.shape.Dimensions...)
	})
	if err != nil {
		panic(errors.WithMessagef(err, "ConstTensor failed to create a constant in the backend"))
	}
	node = &Node{
		graph:        g,
		outputOps:    []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
		inputs:       nodeInputs,
	}
	g.registerNode(node)
	return
}

// ConstCachedTensor returns a constant node for the tensor t.
// If it's the first time the tensor is used in this graph, a new node is created.
// Otherwise, a previously created node is reused.
//
// The caching of the tensor has the side effect of keeping the tensor alive (and its memory resources) util
// the Graph itself is garbage collected. If this is a concern, use ConstTensor instead.
//
// TODO:this can be made default (ConstTensor) once weak references land into Go and the issue of keeping the
// tensor alive is resolved.
// See discussion in https://github.com/golang/go/issues/67552 and cache with
// weak references example in
// https://github.com/golang/go/issues/67552#issuecomment-2200755798
func ConstCachedTensor(g *Graph, t *tensors.Tensor) *Node {
	g.AssertBuilding()
	node, found := g.tensorConstants[t]
	if found {
		return node
	}
	node = ConstTensor(g, t)
	g.tensorConstants[t] = node
	return node
}

// Const creates constant nodes in the Graph. It can take a tensor as well as
// multidimensional slices (or scalars).
//
// It uses tensor.FromAnyValue to figure out the shape given a Go scalar/slice/array.
// If the value is unsupported, it panics.
//
// A tensor.Device (e.g., generated by another computation) will be converted to local first.
// If you are creating very large constants that don't need to be materialized locally, consider
// instead storing them as variables in the context, or as a side parameter.
func Const(g *Graph, x any) *Node {
	if _, ok := x.(*Node); ok {
		exceptions.Panicf(
			"Const(g, x) can only take actual values, not another computation graph `*Node` -- " +
				"for that you don't need Const(), just use it directly.")
	}
	tensor := tensors.FromAnyValue(x)
	return ConstTensor(g, tensor)
}

// ConstAsDType creates a constant of the given DType. It adds the convenience
// of converting x (slice or scalar) to the appropriate type.
// E.g.:
//
//	Pi := ConstAsDType(g, myDType, math.Pi)
//	PiAndE := ConstAsDType(g, myDType, []float64{math.Pi, math.E})
func ConstAsDType(g *Graph, dtype dtypes.DType, x any) *Node {
	if dtype == dtypes.InvalidDType {
		exceptions.Panicf("invalid DType given for ConstAsDType")
	}
	return Const(g, shapes.CastAsDType(x, dtype))
}

// ConstAs creates a constant (slice or scalar) of the same DType and on the same Graph as
// the given base.
func ConstAs(base *Node, x any) *Node {
	return ConstAsDType(base.Graph(), base.DType(), x)
}

// Infinity returns the positive/negative (depending on the value of sign, which must be 1 or -1) for the given dtype.
// For integer dtypes, it returns the highest/lowest values.
func Infinity(g *Graph, dtype dtypes.DType, sign int) *Node {
	switch sign {
	case 1:
		return Const(g, dtype.HighestValue())
	case -1:
		return Const(g, dtype.LowestValue())
	default:
		exceptions.Panicf("Infinity's sign must be 1 or -1, got %d", sign)
		panic(nil) // Disable lint warning.
	}
}

// StopGradient creates an identity node (see Identity), through which gradients don't back-propagate.
//
// No new XLA outputOps is created, so there are no costs to the computation execution speed.
func StopGradient(x *Node) *Node {
	n := Identity(x)
	n.stopGradient = true
	return n
}

// IdentityWithCustomGradient returns x unchanged, but sets a custom gradient function to be applied when
// doing the reverse autograd (gradient) calculation.
//
// The `gradientFn` will be called during auto-grad and will be passed `x` and `v`, the "adjoint", which represents
// the gradient of the loss (typically, but of whatever we are calculating the gradient of) with respect to `x`,
// and we should return the updated `v`, that is, the customized gradient with respect to `x`.
func IdentityWithCustomGradient(x *Node, gradientFn func(x, v *Node) *Node) *Node {
	n := Identity(x)
	n.customVJP = func(node *Node, vjpForOutputs []*Node, _ shapes.Shape) []*Node {
		return []*Node{gradientFn(node, vjpForOutputs[0])}
	}
	return n
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
//
// See also IotaFull.
func Iota(g *Graph, shape shapes.Shape, iotaAxis int) *Node {
	if shape.IsScalar() {
		exceptions.Panicf("cannot Iota a scalar shape, shape=%s", shape)
	}
	adjustedAxis := adjustAxisToRank(iotaAxis, shape.Rank())
	if adjustedAxis < 0 || adjustedAxis >= shape.Rank() {
		exceptions.Panicf("invalid axis #%d for Iota, when shape is rank %d", iotaAxis, shape.Rank())
	}
	return backendIota(g, shape, adjustedAxis)
}

// IotaFull creates a constant of the given shape with increasing numbers for all values.
// So `IotaFull([2,2])` returns `[[0 1][2 3]]`.
func IotaFull(g *Graph, shape shapes.Shape) *Node {
	if !shape.Ok() {
		panic(errors.New("invalid shape"))
	}
	return ReshapeWithShape(Iota(g, shapes.Make(shape.DType, shape.Size()), 0), shape)
}

// validateBuildingGraphFromInputs checks that all inputNodes are of the same Graph and that
// the Graph is valid for building.
// It panics with a corresponding error message in case of issues.
// Otherwise, it returns the Graph common to all inputNodes.
func validateBuildingGraphFromInputs(inputs ...*Node) (g *Graph) {
	if len(inputs) == 0 {
		exceptions.Panicf("no input nodes provided, at least one is required")
	}

	// Checks that all inputNodes are of the same graph.
	for ii, n := range inputs {
		if err := exceptions.TryCatch[error](n.AssertValid); err != nil {
			panic(errors.WithMessagef(err, "invalid input[%d]", ii))
		}
		if n.NumOutputs() != 1 {
			exceptions.Panicf(
				"input[%d](%s) has multiple-outputs, it has to be split with selectOutput calls first",
				ii,
				n,
			)
		}
		if g == nil {
			g = n.Graph()
			g.AssertBuilding()
		} else {
			if n.Graph() != g {
				exceptions.Panicf("combining nodes from different graphs not allowed: "+
					"input[0] graph is %q, input[%d] graph is %q", g.Name(), ii, n.Graph().Name())
			}
		}
	}
	return
}

// Log1P is an alias to Log1p. It returns log(1+x).
func Log1P(x *Node) *Node { return Log1p(x) }

// Sigmoid returns the expression $1/(1+exp(-x)). It is an alias to the Logistic function.
func Sigmoid(x *Node) *Node { return Logistic(x) }

// Sign returns element-wise +1, +/-0 or -1 depending on the sign of x. It returns NaN if the input is NaN.
// The gradient of Sign is assumed to be zero everywhere.
func Sign(x *Node) *Node {
	y := backendSign(x)
	y.stopGradient = true
	return y
}

// Mod adds to the graph the module (remainder) operation on the two input nodes x and y. It's an alias to Rem.
// Standard broadcasting rules apply (see documentation).
func Mod(x, y *Node) *Node {
	if x.DType().IsComplex() || y.DType().IsComplex() {
		exceptions.Panicf("cannot take the remainder (Mod) of a complex number: Mod(%s, %s)",
			x.Shape(), y.Shape())
	}
	return Rem(x, y)
}

// BroadcastPrefix adds dimensions to an array by duplicating the data in the array.
//
// The new dimensions dims are inserted on the left, i.e., if
// broadcast_sizes has values `{a0, ..., aN}` and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
//
// The new dimensions id into copies of the operand, i.e.
//
//	output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
func BroadcastPrefix(x *Node, prefixDims ...int) *Node {
	shape := x.Shape()
	newDims := make([]int, shape.Rank()+len(prefixDims))
	copy(newDims, prefixDims)
	copy(newDims[len(prefixDims):], shape.Dimensions)
	broadcastAxes := make([]int, shape.Rank())
	for i := range shape.Rank() {
		broadcastAxes[i] = i + len(prefixDims)
	}
	outputShape := shapes.Make(x.DType(), newDims...)
	return backendBroadcastInDim(x, outputShape, broadcastAxes)
}

// ExpandAndBroadcast combines ExpandAxes and broadcast of axes of `x`, the returned shape will be newDimensions.
// Only newly expanded axes can be broadcast.
//
//   - newDimensions should have a rank larger than the rank of x, and the new axes in newDimensions
//     should be listed in expandedAxes. In other words: `x.Rank() + len(expandedAxes) == len(newDimensions)`.
//
//   - expandedAxes refer to the axes in newDimensions that are expanded and going to be broadcast. The reminder
//     dimensions in newDimensions much match the corresponding in x.
//
// Example:
//
//	x = Const(g, []int32{10, 20})
//	ExpandAndBroadcast(x, []int{2, 2}, []int{0})  // -> [][]int32{{10, 20}, {10, 20}}
//	ExpandAndBroadcast(x, []int{2, 2}, []int{1})  // -> [][]int32{{10, 10}, {20, 20}}
func ExpandAndBroadcast(x *Node, newDimensions []int, expandedAxes []int) (output *Node) {
	_ = validateBuildingGraphFromInputs(x)
	if x.Rank()+len(expandedAxes) != len(newDimensions) {
		exceptions.Panicf(
			"there must be exactly one expandedAxes (%v) for each new axis in newDimensions (%v) -- x.shape=%s",
			expandedAxes,
			newDimensions,
			x.Shape(),
		)
	}

	// Verify that the values of expandedAxis and create a map of the expanded axis.
	expandedSet := sets.Make[int](len(expandedAxes))
	for ii, axis := range expandedAxes {
		axis = adjustAxisToRank(axis, len(newDimensions))
		if axis < 0 || axis >= len(newDimensions) {
			exceptions.Panicf(
				"expandedAxes (%v) defines a value out-of-range (%d-th value -> %d), they must be between 0 and len(newDimensions)=%d",
				expandedAxes,
				ii,
				axis,
				len(newDimensions),
			)
		}
		if expandedSet.Has(axis) {
			exceptions.Panicf(
				"expandedAxes (%v) repeats axis %d (expandedAxes[%d]), they must be all unique and between 0 and len(newDimensions)=%d",
				expandedAxes,
				axis,
				ii,
				len(newDimensions),
			)
		}
		expandedSet.Insert(axis)
	}

	var preservedAxes []int
	if !x.Shape().IsScalar() {
		preservedAxes = make([]int, 0, x.Rank())
		axisInX := 0
		for axis, dim := range newDimensions {
			if !expandedSet.Has(axis) {
				preservedAxes = append(preservedAxes, axis)
				if x.Shape().Dimensions[axisInX] != dim && x.Shape().Dimensions[axisInX] != 1 {
					exceptions.Panicf("the values of newDimensions (%v) that are not expanded (not in expandedAxes) "+
						"must match the corresponding value in x shape (%s) or be 1 (if broadcasting), "+
						"but the value of newDimensions[%d]=%d does not match the value in x.Shape().Dimensions[%d]=%d",
						newDimensions, x.Shape(), axis, dim, axisInX, x.Shape().Dimensions[axisInX])
				}
				axisInX++
			}
		}
	}

	return backendBroadcastInDim(x, shapes.Make(x.DType(), newDimensions...), preservedAxes)
}

// BroadcastToShape broadcasts x to the given shape.
// x must have an equal or lower rank than shape, and if shape has higher rank, x will be expanded at the end (so new axes will be appended to x).
// Dimensions of x must either match the corresponding dimension in shape, or they must be 1, in which case they are broadcast.
//
// It works as expected if x is a scalar.
//
// Notice that the dtype of shape is ignored, the returned value preserves the dtype of x.
//
// This is equivalent to BroadcastToDims(x, shape.Dimensions...).
func BroadcastToShape(x *Node, shape shapes.Shape) *Node {
	_ = validateBuildingGraphFromInputs(x)
	return BroadcastToDims(x, shape.Dimensions...)
}

// BroadcastToDims broadcasts x to the given dimensions.
// x must have an equal or lower rank than the given dimensions, and if there are more dimensions than x rank,
// x will be expanded at the end (so new axes will be appended to x).
// Dimensions of x must either match the corresponding value in dimensions, or they must be 1, in which case they
// are broadcast.
//
// It works as expected if x is a scalar.
//
// See also the equivalent BroadcastToShape.
func BroadcastToDims(x *Node, dimensions ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	shape := shapes.Make(x.DType(), dimensions...)
	if x.Shape().IsScalar() && shape.IsScalar() {
		// Assume nothing to do.
		return x
	}
	if x.Shape().IsScalar() {
		return backendBroadcastInDim(x, shape, nil)
	}
	broadcastDims := make([]int, x.Rank())
	for ii := range x.Rank() {
		broadcastDims[ii] = ii
	}
	return backendBroadcastInDim(x, shape, broadcastDims)
}

// ConvertType is an alias to ConvertDType.
// Deprecated: use ConvertDType instead.
func ConvertType(x *Node, dtype dtypes.DType) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if !dtype.IsSupported() {
		exceptions.Panicf("converting to an unsupported dtype %s", dtype)
	}
	return ConvertDType(x, dtype)
}

// Where takes element-wise values from onTrue or onFalse depending on the value of condition (expected to be boolean).
//
// Usual implicit broadcasting rules don't apply. But it will broadcast in the following cases:
//
//  1. If either onTrue or onFalse are a scalar, they are broadcast to the other (onFalse or onTrue respectively).
//     If both are scalars, they will be broadcast to the shape of condition.
//  2. If condition is a prefix to the shapes of onTrue/onFalse then condition is expanded to match.
//     This is useful for masking of embeddings for instance.
func Where(condition, onTrue, onFalse *Node) *Node {
	_ = validateBuildingGraphFromInputs(condition)
	if condition.DType() != dtypes.Bool {
		exceptions.Panicf("Where(condition, onTrue, onFalse) requires condition to be of dtype Bool, got %s instead",
			condition.Shape())
	}

	// Find output shape:
	outputShape := onTrue.Shape()
	if outputShape.IsScalar() {
		outputShape = onFalse.Shape()
		if outputShape.IsScalar() {
			outputShape = condition.Shape()
		}
	}

	// Broadcast onTrue and onFalse to the outputShape if needed.
	if !onTrue.Shape().IsScalar() && !onFalse.Shape().IsScalar() && !onTrue.Shape().Equal(onFalse.Shape()) {
		exceptions.Panicf("Where() requires onTrue (%s) and onFalse (%s) to either be the same shape or be a scalar",
			onTrue.Shape(), onFalse.Shape())
	}

	// Broadcasting of condition when it's a prefix to one of the operands:
	if !condition.IsScalar() {
		if condition.Rank() > outputShape.Rank() {
			exceptions.Panicf(
				"Where() requires the condition shape (%s) to be a prefix (or equal) to the output shape (%s), onTrue is %s and onFalse is %s",
				condition.Shape(),
				outputShape,
				onTrue.Shape(),
				onFalse.Shape(),
			)
		}
		for axis, dim := range condition.Shape().Dimensions {
			if outputShape.Dimensions[axis] != dim {
				exceptions.Panicf(
					"Where() requires the condition shape to be a prefix (or equal) to the output shape, but condition is %s and output shape is %s",
					condition.Shape(),
					outputShape,
				)
			}
		}
		if condition.Rank() != outputShape.Rank() {
			// Broadcast condition.
			extraAxes := outputShape.Rank() - condition.Rank()
			condition = InsertAxes(condition, xslices.SliceWithValue(extraAxes, -1)...)
			condition = BroadcastToDims(condition, outputShape.Dimensions...)
		}
	}

	// Broadcasting of scalar onTrue or onFalse is done by the backend.
	return backendWhere(condition, onTrue, onFalse)
}

// Reshape x to the given dimensions. Total size cannot change. One dimension can be left as -1,
// in which case it will be set to match the size, if possible.
func Reshape(x *Node, dimensions ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	totalSize := x.Shape().Size()
	newSize := 1
	missingIdx := -1
	for idx, dim := range dimensions {
		if dim != -1 {
			newSize *= dim
		} else {
			if missingIdx != -1 {
				exceptions.Panicf("only one dimension can be missing (that is, set to -1) for Reshape, %v given",
					dimensions)
			}
			missingIdx = idx
		}
	}
	if missingIdx != -1 {
		tmpDim := slices.Clone(dimensions)
		tmpDim[missingIdx] = totalSize / newSize
		newSize *= tmpDim[missingIdx]
		if newSize != totalSize {
			exceptions.Panicf(
				"cannot find new dimension for axis %d that will make new dimensions %v match original the input size %d (dimensions %v)",
				missingIdx,
				dimensions,
				totalSize,
				x.Shape().Dimensions,
			)
		}
		dimensions = tmpDim
	} else {
		if newSize != totalSize {
			exceptions.Panicf("total requested size %d (dimensions=%v) doesnt match original size %d (dimensions %v)",
				newSize, dimensions, totalSize, x.Shape().Dimensions)
		}
	}
	return backendReshape(x, dimensions...)
}

// ReshapeWithShape reshapes x to the dimensions given by shape.
// Total size cannot change, neither the DType is allowed to change.
// Conceptually, this is a limited form of "shape casting."
func ReshapeWithShape(x *Node, shape shapes.Shape) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if shape.DType != x.DType() {
		exceptions.Panicf("cannot change dtype (from %s to %s) with ReshapeWithShape",
			x.DType(), shape.DType)
	}
	if shape.Size() != x.Shape().Size() {
		exceptions.Panicf(
			"shapes (x.shape=%s, shape=%s) have different total sizes (from %d to %d), reshape not possible",
			x.Shape(),
			shape,
			x.Shape().Size(),
			shape.Size(),
		)
	}
	return backendReshape(x, shape.Dimensions...)
}

// InsertAxes expands x creating new axes just before the axes given -- beforeAxes points to positions on the original
// tensor x, and they can be repeated, in case one wants to insert more than one new axis in the given position.
//
// If beforeAxes[ii] < 0, then they are counted from the end — -1 represents a new axis after the end of the original shape.
//
// The new axes will be of dimension 1 (so the total size of and contents of the tensor remains the same),
// and the rank is increased by `len(axes)`.
//
// See also ExpandAxes, where the new axes are given as positions in the target shape.
func InsertAxes(x *Node, beforeAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if len(beforeAxes) == 0 {
		// Trivial case, noop.
		return x
	}

	// Ranks.
	fromRank := x.Rank()
	toRank := fromRank + len(beforeAxes)

	// Copy dimensions, so we don't change the callers' values, and replace negatives.
	newAxes := make([]int, len(beforeAxes))
	copy(newAxes, beforeAxes)
	beforeAxes = newAxes
	for ii, axis := range beforeAxes {
		if axis < 0 {
			beforeAxes[ii] = fromRank + 1 + axis
		}
	}
	slices.Sort(beforeAxes)

	// Create new target shape.
	toShape := shapes.Shape{DType: x.DType(), Dimensions: make([]int, toRank)}
	iiOriginal, iiNewAxes := 0, 0
	for ii := range toShape.Dimensions {
		if iiNewAxes < len(beforeAxes) && beforeAxes[iiNewAxes] <= iiOriginal || iiOriginal == fromRank {
			toShape.Dimensions[ii] = 1
			iiNewAxes += 1
		} else {
			toShape.Dimensions[ii] = x.Shape().Dimensions[iiOriginal]
			iiOriginal += 1
		}
	}
	return ReshapeWithShape(x, toShape)
}

// ExpandDims is an alias to InsertAxes.
//
// Deprecated: this will be removed at the next release! Notice this has a different semantics than the more common numpy.expand_dims (which is matched by ExpandAxes). Please use InsertAxes or ExpandAxes instead.
func ExpandDims(x *Node, beforeAxes ...int) *Node {
	return InsertAxes(x, beforeAxes...)
}

// ExpandAxes expands x creating new axes at the positions given by newAxes -- the positions are given at the target shape.
//
// The list newAxes represent the positions in the returned shape.
// If newAxes[ii] < 0, then they are counted from the end of the new shape — -1 represents the last axis in the new shape.
//
// There should be no repeated values in newAxes -- since they represent the positions in the returned shape, it wouldn't make sense.
//
// See also InsertAxes, where the new axes are given as positions in the target shape.
func ExpandAxes(x *Node, newAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	if len(newAxes) == 0 {
		// Trivial case, noop.
		return x
	}

	// Ranks.
	fromRank := x.Rank()
	toRank := fromRank + len(newAxes)

	// Adjust new axes and check they are unique.
	adjustedNewAxes := make([]int, len(newAxes))
	copy(adjustedNewAxes, newAxes)
	for ii, axis := range newAxes {
		if axis < 0 {
			adjustedNewAxes[ii] = toRank + axis
		}
	}
	slices.Sort(adjustedNewAxes)
	for ii := range adjustedNewAxes {
		if ii > 0 && adjustedNewAxes[ii] == adjustedNewAxes[ii-1] {
			exceptions.Panicf(
				"ExpandedAxes(x, newAxes=%v...) got repeated new axis %d which doesn't make sense -- likely an error",
				newAxes,
				adjustedNewAxes[ii],
			)
		}
	}

	// Create new target shape.
	toShape := shapes.Shape{DType: x.DType(), Dimensions: make([]int, toRank)}
	iiOriginal, iiNewAxes := 0, 0
	for axis := range toShape.Dimensions {
		if iiNewAxes < len(adjustedNewAxes) && adjustedNewAxes[iiNewAxes] == axis {
			toShape.Dimensions[axis] = 1
			iiNewAxes += 1
		} else {
			toShape.Dimensions[axis] = x.Shape().Dimensions[iiOriginal]
			iiOriginal += 1
		}
	}
	return ReshapeWithShape(x, toShape)
}

// ExpandLeftToRank prepend axes of dimension 1 to x, until it reaches rank `newRank`.
func ExpandLeftToRank(x *Node, newRank int) (output *Node) {
	_ = validateBuildingGraphFromInputs(x)
	if newRank < x.Rank() {
		exceptions.Panicf("ExpandLeftToRank(newRank=%d), but x already has rank %d", newRank, x.Rank())
	}
	if newRank == x.Rank() {
		// Already the correct rank.
		output = x
		return
	}
	newDims := make([]int, 0, newRank)
	for ii := 0; ii < newRank-x.Rank(); ii++ {
		newDims = append(newDims, 1)
	}
	newDims = append(newDims, x.Shape().Dimensions...)
	output = Reshape(x, newDims...)
	return
}

// Squeeze removes `axes` of dimension 1. If `axes` is not set, all axes of dimension 1 are removed.
// Otherwise, only the provided `axes` are removed. If any of the given `axes` is not of dimension 1,
// an error is raised in the Graph and an invalid node is returned.
//
// If all dimensions are reduced, it returns a scalar.
func Squeeze(x *Node, axes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)

	newDims := make([]int, x.Rank())
	copy(newDims, x.Shape().Dimensions)
	if len(axes) == 0 {
		for ii, dim := range newDims {
			if dim == 1 {
				newDims[ii] = 0
			}
		}
	} else {
		for axisIdx, axis := range axes {
			if axis < 0 {
				axis = x.Rank() + axis
			}
			if axis < 0 || axis >= x.Rank() {
				exceptions.Panicf("Squeeze() for x.shape=%s, axis %d is out-of-range", x.Shape(), axes[axisIdx])
			}
			if newDims[axis] == 0 {
				exceptions.Panicf("Squeeze() for x.shape=%s, axis %d was selected twice!?", x.Shape(), axes[axisIdx])
			}
			if newDims[axis] != 1 {
				exceptions.Panicf("Squeeze() for x.shape=%s, axis %d does not have dimension 1", x.Shape(), axes[axisIdx])
			}
			newDims[axis] = 0
		}
	}

	tgtAxisIdx := 0
	for _, dim := range newDims {
		if dim > 0 {
			newDims[tgtAxisIdx] = dim
			tgtAxisIdx++
		}
	}
	newDims = newDims[:tgtAxisIdx] // May reduce to a scalar.
	return Reshape(x, newDims...)
}

// ArgMax returns the index of the largest element across the given axis.
//
// The selected axis is reduced, and the output has one fewer axes (rank `x.Rank() - 1`).
// The output `DType`, if not given, is `dtypes.Int32`.
//
// Ties are resolved by returning the smallest index.
func ArgMax(x *Node, axis int, outputDType ...dtypes.DType) (output *Node) {
	_ = validateBuildingGraphFromInputs(x)
	dtype := dtypes.Int32
	if len(outputDType) > 1 {
		exceptions.Panicf("ArgMax takes at most one outputDType, %d values given", len(outputDType))
	} else if len(outputDType) == 1 {
		dtype = outputDType[0]
	}
	axis = adjustAxisToRank(axis, x.Rank())
	return backendArgMinMax(x, axis, dtype, false)
}

// ArgMin returns the index of the smallest element across the given axis.
//
// The selected axis is reduced, and the output has one fewer axes (rank `x.Rank() - 1`).
// The output `DType`, if not given, is `dtypes.Int32`.
//
// Ties are resolved by returning the smallest index.
func ArgMin(x *Node, axis int, outputDType ...dtypes.DType) (output *Node) {
	_ = validateBuildingGraphFromInputs(x)
	dtype := dtypes.Int32
	if len(outputDType) > 1 {
		exceptions.Panicf("ArgMin takes at most one outputDType, %d values given", len(outputDType))
	} else if len(outputDType) == 1 {
		dtype = outputDType[0]
	}
	axis = adjustAxisToRank(axis, x.Rank())
	return backendArgMinMax(x, axis, dtype, true)
}

// adjustAxesToRank not-inplace, it returns an adjusted copy of the given `axesWithNegatives`.
// An axis set to -1 is converted to `rank - 1`.
// It panics if any of the axes is out-of-range for given rank.
func adjustAxesToRank(rank int, axesWithNegatives []int, paramName string) []int {
	axes := slices.Clone(axesWithNegatives)
	for ii := range axes {
		if axes[ii] < 0 {
			axes[ii] = rank + axes[ii]
		}
		if axes[ii] < 0 || axes[ii] > rank {
			exceptions.Panicf("%s's axis #%d of %v = %v given is out-of-range for rank %d",
				paramName, ii, axesWithNegatives, axesWithNegatives[ii], rank)
		}
	}
	return axes
}

// adjustAxesToRankAndSort not-inplace, it returns an adjusted copy of the given `axesWithNegatives`.
// Finally, it sorts the axes -- careful not to use it where the order matters.
// An axis set to -1 is converted to `rank - 1`.
// It panics if any of the axes is out-of-range for given rank.
func adjustAxesToRankAndSort(rank int, axesWithNegatives []int, paramName string) []int {
	axes := adjustAxesToRank(rank, axesWithNegatives, paramName)
	slices.Sort(axes)
	return axes
}

// ReduceSum reduces by summing over X elements over the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
// See ReduceAndKeep for a version to preserve the reduced axes.
func ReduceSum(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceSum(x, axes...)
}

// ReduceAllSum reduces all dimensions to a scalar by summing.
func ReduceAllSum(x *Node) *Node {
	return ReduceSum(x)
}

// MaskedReduceSum reduces by summing the `x` elements over the selected axes.
// If `reduceAxes` is nil, reduce over all dimensions to a scalar.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
//
// It ignores values for which the corresponding mask is false.
// The `mask` and `x` values must have the same shape.
// If mask is nil, it behaves like ReduceSum.
func MaskedReduceSum(x, mask *Node, reduceAxes ...int) *Node {
	if mask == nil {
		return ReduceSum(x, reduceAxes...)
	}
	maskedX := Where(mask, x, ZerosLike(x))
	return ReduceSum(maskedX, reduceAxes...)
}

// MaskedReduceAllSum reduces all dimensions to a scalar by summing.
//
// It ignores values for which the corresponding mask is false.
// The `mask` and `x` values must have the same shape.
func MaskedReduceAllSum(x, mask *Node) *Node {
	return MaskedReduceSum(x, mask)
}

// ReduceMean reduces by taking the mean over the elements of the selected axes.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
// See ReduceAndKeep for a version to preserve the reduced axes.
func ReduceMean(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	sum := ReduceSum(x, reduceAxes...)
	denominator := x.Shape().Size() / sum.Shape().Size()
	return MulScalar(sum, 1.0/float64(denominator))
}

// ReduceAllMean reduces all dimensions to a scalar by taking the mean.
func ReduceAllMean(x *Node) *Node {
	return ReduceMean(x)
}

// MaskedReduceMean reduces by taking the mean over the elements of the selected axes.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
//
// It first applies a mask to x, converting masked values to the neutral value of the operation (0).
// For reduction dimensions that are completely masked, it returns 0.
// If mask is nil, it behaves like ReduceMean.
func MaskedReduceMean(x, mask *Node, reduceAxes ...int) *Node {
	if mask == nil {
		return ReduceMean(x, reduceAxes...)
	}

	if mask.Rank() < x.Rank() {
		// Mask must have a prefix rank to X, in which case we need to expand it to get the count of masked elements right.
		mask = BroadcastToDims(mask, x.Shape().Dimensions...)
	}
	zeros := ZerosLike(x)
	maskedX := Where(mask, x, zeros)
	sum := ReduceSum(maskedX, reduceAxes...)
	denominator := ConvertDType(mask, x.DType())
	denominator = ReduceSum(denominator, reduceAxes...)
	denominator = Max(denominator, OnesLike(denominator))
	denominator.stopGradient = true
	return Div(sum, denominator)
}

// MaskedReduceAllMean reduces all dimensions to a scalar by taking the mean.
// It ignores entries where mask is false.
func MaskedReduceAllMean(x, mask *Node) *Node {
	return MaskedReduceMean(x, mask)
}

// ReduceMultiply reduces by summing over the elements of the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
// See ReduceAndKeep for a version to preserve the reduced axes.
func ReduceMultiply(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceProduct(x, axes...)
}

// ReduceAllMultiply reduces all dimensions to a scalar by multiplying.
func ReduceAllMultiply(x *Node) *Node {
	return ReduceMultiply(x)
}

// ReduceMax reduces by taking the max over the elements of the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
// See ReduceAndKeep for a version to preserve the reduced axes.
func ReduceMax(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceMax(x, axes...)
}

// ReduceAllMax reduces all dimensions to a scalar by taking the max.
func ReduceAllMax(x *Node) *Node {
	return ReduceMax(x)
}

// MaskedReduceMax reduces by taking the max of `x` elements over the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// It ignores values for which the corresponding mask is false.
// The shapes of `mask and x must be the same.
// If mask is nil, it behaves like ReduceMax.
func MaskedReduceMax(x, mask *Node, reduceAxes ...int) *Node {
	if mask == nil {
		return ReduceMax(x, reduceAxes...)
	}
	g := x.Graph()
	lowest := lowestForDType(g, x.DType())
	broadcastLowest := BroadcastToDims(lowest, x.Shape().Dimensions...)
	maskedX := Where(mask, x, broadcastLowest)
	return ReduceMax(maskedX, reduceAxes...)
}

// MaskedReduceAllMax reduces all dimensions to a scalar by taking the max.
//
// It ignores values for which the corresponding mask is false.
// The shapes of `mask and x must be the same.
func MaskedReduceAllMax(x, mask *Node) *Node {
	return MaskedReduceMax(x, mask)
}

// ReduceMin reduces by taking the min over the elements of the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// The reduced axes of `x` are removed in the output -- so the rank is reduced.
// See ReduceAndKeep for a version to preserve the reduced axes.
func ReduceMin(x *Node, reduceAxes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	axes := adjustAxesToRankAndSort(x.Rank(), reduceAxes, "x")
	return backendReduceMin(x, axes...)
}

// ReduceAllMin reduces all dimensions to a scalar by taking the min.
func ReduceAllMin(x *Node) *Node {
	return ReduceMin(x)
}

// MaskedReduceMin reduces by taking the min of `x` elements over the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// It ignores values for which the corresponding mask is false.
// The shapes of `mask and x must be the same.
// If mask is nil, it behaves like ReduceMin.
func MaskedReduceMin(x, mask *Node, reduceAxes ...int) *Node {
	if mask == nil {
		return ReduceMin(x, reduceAxes...)
	}
	g := x.Graph()
	lowest := highestForDType(g, x.DType())
	broadcastHighest := BroadcastToDims(lowest, x.Shape().Dimensions...)
	maskedX := Where(mask, x, broadcastHighest)
	return ReduceMin(maskedX, reduceAxes...)
}

// MaskedReduceAllMin reduces all dimensions to a scalar by taking the min.
//
// It ignores values for which the corresponding mask is false.
// The shapes of `mask and x must be the same.
// If mask is nil, it behaves like ReduceAllMin.
func MaskedReduceAllMin(x, mask *Node) *Node {
	return MaskedReduceMin(x, mask)
}

// SliceAxisSpec specifies the range and stride of an axis to include in a Slice.
//
// The recommendation is to use AxisRange or AxisElem (defined below) to create it.
//
// Full means to include the whole range (and ignore Start/End), and
// NoEnd means from Start to the full dimension of the axis.
//
// Optional (if Stride != 0) it can set the stride for the axis as well.
//
// Spacer means this AxisRange should be the generic definition for all
// undefined axes -- useful when the rank of the node is not known.
//
// Consider using function AxisRange below to construct SliceAxisSpec values.
//
// TODO: Add strides.
type SliceAxisSpec struct {
	Start, End, StrideValue int
	Full, NoEnd             bool
	IsSpacer                bool
}

// Stride returns a copy of the SliceAxisSpec with Stride set to the given stride.
func (ar SliceAxisSpec) Stride(stride int) SliceAxisSpec {
	ar2 := ar
	ar2.StrideValue = stride
	return ar2
}

// Spacer marks this SliceAxisSpec to be a generic filler range to use on the undefined
// axes in Slice -- similar to a "*" in a path definition.
//
// It works with any SliceAxisSpec, so it can be used with the return of any call to
// AxisRange or AxisElem.
//
// Example: let's say we want to get just the last example of a batch, and just the first
// element of the embedding. Assume x is shaped `[batch_size, ..., embedding_size]` and
// we want something like `x[-1, ..., 0:1]`
//
// sample := Slice(x, AxisElem(-1), AxisRange().Spacer(), AxisElem(0))
//
// Notice that "spacer" ranges also matches zero dimensions. So if x is shaped `[5, 5]`,
// calling `Slice(x, AxisElem(0), AxisRange().Spacer(), AxisElem(0))` would return
// a node of shape `[1, 1]` and the spacer would be ignored.
func (ar SliceAxisSpec) Spacer() SliceAxisSpec {
	ar2 := ar
	ar2.IsSpacer = true
	return ar2
}

// AxisRange defines a range to take for an axis in Slice.
// It returns an `SliceAxisSpec` object.
//
// The indices can have 0, 1 or 2 elements:
// - If `len(indices) == 0`, it's assumed to be the full range of the axis.
// - If `len(indices) == 1`, it's assumed to be the start, and the range should be taken to the end.
// - If `len(indices) == 2`, they should be the start and end indices for the axis.
// - If `len(indices) > 2`, an error is raised with panic.
//
// See also AxisElem if you want to define only one element of the range.
func AxisRange(indices ...int) SliceAxisSpec {
	if len(indices) == 0 {
		return SliceAxisSpec{Full: true}
	}
	if len(indices) == 1 {
		return SliceAxisSpec{Start: indices[0], NoEnd: true}
	}
	if len(indices) > 2 {
		exceptions.Panicf("AxisRange(%v): more than 2 indices provided, that's not supported", indices)
	}
	return SliceAxisSpec{Start: indices[0], End: indices[1]}
}

// AxisRangeToEnd defines a range from the given value to the end of the axis.
// It's return value is to be used by Slice to specify one axis.
func AxisRangeToEnd(from int) SliceAxisSpec {
	return SliceAxisSpec{Start: from, NoEnd: true}
}

// AxisRangeFromStart defines a range from the start (0) to the given value for the axis.
// It's return value is to be used by Slice to specify one axis.
func AxisRangeFromStart(to int) SliceAxisSpec {
	return SliceAxisSpec{Start: 0, End: to}
}

// AxisElem defines a range of one element to take for an axis in Slice.
// It returns an `SliceAxisSpec` object.
func AxisElem(index int) SliceAxisSpec {
	if index == -1 {
		// Take the last element: since we don't know the dimensions of the axes yet, just
		// take it to the end, it will be only one element.
		return SliceAxisSpec{Start: index, NoEnd: true}
	}
	return SliceAxisSpec{Start: index, End: index + 1}
}

// adjustAxisToRank converts negative axes to a value starting from the end.
func adjustAxisToRank(axis, rank int) int {
	if axis >= 0 {
		return axis
	}
	return rank + axis
}

// Slice take slices of the operand.
//
// Each axis can have a range defined as (start, end) pairs. Any axis for which a range
// is not specified is assumed to be taken in full. Consider using the shortcut AxisRange to define
// the ranges.
//
// Examples:
//
// - For `x = {10, 20, 30, 40}`:
//   - `Slice(x) = {10, 20, 30, 40}`  // SliceAxisSpec not given is taken in full.
//   - `Slice(x, AxisRange()) = {10, 20, 30, 40}`  // Default for AxisRange is the full range.
//   - `Slice(x, AxisRange(1,-1)) = {20, 30}`  // Negative values are taken from the end of the axis dimension.
//   - `Slice(x, AxisRangeFromStart(-2)) = {10, 20}`  // Negative values are taken from the end of the axis dimension.
//   - `Slice(x, AxisRangeToEnd(2)) = {30, 40}`  // Negative values are taken from the end of the axis dimension.
//   - `Slice(x, AxisElem(2)) = {3}`  // Take only one element of an axis.
//
// - For `x = {{1, 2, 3}, {4, 5, 6}}`:
//   - `Slice(x, AxisRange(), AxisElem(0)) = {{1}, {4}}` // First axis taken in full, second axis only the first element.
//   - `Slice(x, AxisElem(1)) = {{4, 5, 6}}`  // Missing second SliceAxisSpec, assumed to be taken in full.
//
// If Slice is called with `x.shape = [5, 5, 5, 5]` and `axesRanges=AxisElem(1), AxisRange(), AxisRange(2), AxisRange(0,2)`
// would return a node shaped `[1, 5, 3, 2]`.
//
// It also supports "spacers" (like "*" in paths), that fill the unknown axes.
// Example: let's say we want to get just the last example of a batch, and just the first
// element of the embedding. Assume x is shaped `[batch_size, ..., embedding_size]` and
// we want something like `x[-1, ..., 0:1]`.
//
// sample := Slice(x, AxisElem(-1), AxisRange().Spacer(), AxisElem(0))
//
// It also works with strides, use the SliceAxisSpec.Stride() method to conveniently set it.
//
// Example:
//
// - For `x = {1, 2, 3, 4}`:
//   - `Slice(x, AxisRange().Stride(2)) = {1, 3}`  // The whole range, but with a stride of 2.
//
// - For `x = {{1, 2, 3}, {4, 5, 6}}`:
//   - `Slice(x, AxisRange().Stride(2), AxisRange(-1)) = {{3}}`  // Take every 2nd row (so only the 1st here), the last column.
func Slice(x *Node, axesSpec ...SliceAxisSpec) *Node {
	_ = validateBuildingGraphFromInputs(x)
	rank := x.Rank()

	// Convert spacers
	var numSpacers int
	for _, spec := range axesSpec {
		if spec.IsSpacer {
			numSpacers++
		}
	}
	if numSpacers > 1 {
		exceptions.Panicf(
			"Only one \"spacer\" range is allowed in Slice, but %d were given: axesSpec=%+v",
			numSpacers,
			axesSpec,
		)
	}
	if numSpacers == 1 {
		// Replace spacer spec with as many copies as needed to fill the axesSpec to match the rank.
		newAxesSpec := make([]SliceAxisSpec, 0, rank)
		copies := rank - len(axesSpec) + numSpacers
		if copies < 0 {
			exceptions.Panicf(
				"Slice was given %d ranges (not counting spacer), but x only has (rank) %d axes",
				len(axesSpec)-1,
				rank,
			)
		}
		for _, spec := range axesSpec {
			if !spec.IsSpacer {
				newAxesSpec = append(newAxesSpec, spec)
			} else {
				spec.IsSpacer = false
				for ii := 0; ii < copies; ii++ {
					newAxesSpec = append(newAxesSpec, spec)
				}
			}
		}
		axesSpec = newAxesSpec
	}

	if len(axesSpec) > rank {
		exceptions.Panicf("Slice was given %d ranges, but x only has (rank) %d axes", len(axesSpec), rank)
	}
	starts := make([]int, rank)
	limits := make([]int, rank)
	strides := make([]int, rank)
	for ii, dim := range x.Shape().Dimensions {
		// Start with the full range.
		starts[ii] = 0
		limits[ii] = dim
		strides[ii] = 1
		if len(axesSpec) > ii && !axesSpec[ii].Full {
			starts[ii] = adjustAxisToRank(axesSpec[ii].Start, dim)
			if !axesSpec[ii].NoEnd {
				limits[ii] = adjustAxisToRank(axesSpec[ii].End, dim)
			}
		}
		if len(axesSpec) > ii && axesSpec[ii].StrideValue > 0 {
			strides[ii] = axesSpec[ii].StrideValue
		}
	}
	return backendSlice(x, starts, limits, strides)
}

// SliceAxis is similar to Slice, but take a slice of one axis only, and preserve all others.
//
// Example:
//
//	x.Shape() == [5, 4, 3]
//	SliceAxis(x, 1, AxisElem(1)) -> shape [5, 1 (sliced axis), 3]
func SliceAxis(x *Node, axis int, axisSpec SliceAxisSpec) *Node {
	specs := make([]SliceAxisSpec, x.Rank())
	adjustedAxis := AdjustAxisToOperandRank(x, axis)
	for ii := range specs {
		if ii == adjustedAxis {
			specs[ii] = axisSpec
		} else {
			specs[ii] = AxisRange()
		}
	}
	return Slice(x, specs...)
}

// Split splits x on the given axis in numSplits equally shaped values.
// x.Shape().Dimensions[axis] must be divisible by numSplits.
//
// Example:
//
//	x := IotaFull(g, shapes.Make(dtypes.Int32, 2, 3)) // Creates [[0 1 2][3 4 5]]
//	splits := Split(x, 1, 3) // Split along axis 1 into 3 parts
//	// Now splits[0] is [[0][3]], splits[1] is [[1][4]], splits[2] is [[2][5]]
func Split(x *Node, axis int, numSplits int) []*Node {
	axis = AdjustAxisToOperandRank(x, axis)
	dim := x.Shape().Dimensions[axis]
	if dim%numSplits != 0 {
		exceptions.Panicf(
			"Split: x.Shape().Dimensions[%d] (=%d) must be divisible by numSplits (=%d)",
			axis,
			dim,
			numSplits,
		)
	}

	// Trivial case of one split:
	if numSplits == 1 {
		return []*Node{x}
	}

	splits := make([]*Node, numSplits)
	splitDim := dim / numSplits
	for ii := 0; ii < numSplits; ii++ {
		start := ii * splitDim
		end := start + splitDim
		if ii == numSplits-1 {
			end = dim
		}
		splits[ii] = SliceAxis(x, axis, AxisRange(start, end))
	}
	return splits
}

// Concatenate results on the given axis. A negative axis will be counted from
// the end -- so `axis==-1` means the last axis.
//
// If operands are scalars, they will be concatenated to a vector (just use `axis=0`).
func Concatenate(operands []*Node, axis int) *Node {
	_ = validateBuildingGraphFromInputs(operands...)
	if len(operands) == 0 {
		exceptions.Panicf("cannot Concatenate with 0 operands")
	}
	rank := operands[0].Rank()
	if rank == 0 {
		// Scalars will be converted to [1] before concatenating.
		operands = xslices.Map(operands, func(x *Node) *Node { return InsertAxes(x, 0) })
		rank = 1
	}
	if len(operands) == 1 {
		// Trivial solution.
		return operands[0]
	}
	baseShape := operands[0].Shape()
	adjustedAxis := adjustAxisToRank(axis, rank)
	for ii, node := range operands[1:] {
		if node.DType() != baseShape.DType {
			exceptions.Panicf("Concatenate operand #%d has different dtype (%s) than operand 0's dtype (%s)",
				ii+1, node.DType(), baseShape.DType)
		}
		if node.Rank() != rank {
			exceptions.Panicf("Concatenate operand #%d has different rank (%d) than operand 0's rank (%d)",
				ii+1, node.Rank(), operands[0].Rank())
		}
		for ii, nodeDim := range node.Shape().Dimensions {
			if ii == adjustedAxis {
				// Dimension being concatenated can be different.
				continue
			}
			if baseShape.Dimensions[ii] != nodeDim {
				exceptions.Panicf(
					"Concatenate(axis=%d) operand #%d has incompatible shape (%s) with operand 0's shape (%s) "+
						"-- except for axis %d, the dimensions on all other axes must match",
					axis,
					ii+1,
					node.Shape(),
					baseShape,
					axis,
				)
			}
		}
	}
	return backendConcatenate(adjustedAxis, operands...)
}

// Stack puts together many values (*Node) with the exact same shape by creating a new axis and concatenating them.
//
// Axis is relative to returning shape.
//
// The returned value increased the rank by 1: output.Rank() = 1+operands[i].Rank()
func Stack(operands []*Node, axis int) *Node {
	_ = validateBuildingGraphFromInputs(operands...)
	operands = xslices.Map(operands, func(x *Node) *Node { return ExpandAxes(x, axis) })
	return Concatenate(operands, axis)
}

// concatenateVJP implements a VJP function for the ConcatenateNode operation.
func concatenateVJP(node, v *Node, _ shapes.Shape) []*Node {
	vjps := make([]*Node, 0, len(node.inputNodes))
	params := node.inputs.(*nodeInputsConcatenate)
	concatAxis := params.axis
	shape := node.Shape()

	// Set starts and limits for slices that are shared among all concatenated inputNodes.
	starts, limits := make([]int, shape.Rank()), make([]int, shape.Rank())
	ranges := make([]SliceAxisSpec, shape.Rank())
	for dim := 0; dim < shape.Rank(); dim++ {
		if dim == concatAxis {
			continue
		}
		ranges[dim] = AxisRange()
		starts[dim], limits[dim] = 0, shape.Dimensions[dim]
	}

	// Take slice for each concatenated input.
	concatDimStart := 0
	for _, input := range node.inputNodes {
		concatDimEnd := concatDimStart + input.Shape().Dimensions[concatAxis]
		ranges[concatAxis] = AxisRange(concatDimStart, concatDimEnd)
		concatDimStart = concatDimEnd
		vjps = append(vjps, Slice(v, ranges...))
	}
	return vjps
}

// Reverse returns x with the values for the given dimensions reversed, that is,
// the value indexed at `i` will be swapped with the value at indexed `(dimension_size - 1 - i)`.
// The shape remains the same.
func Reverse(x *Node, axes ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	rank := x.Rank()
	adjustedAxes := slices.Clone(axes)
	for ii, axis := range adjustedAxes {
		adjustedAxes[ii] = adjustAxisToRank(axis, rank)
		if adjustedAxes[ii] > rank || adjustedAxes[ii] < 0 {
			exceptions.Panicf("in Reverse(x, axes=%v), passed axis %d which is out-of-limits for x rank %d",
				axes, axis, rank)
		}
	}
	return backendReverse(x, adjustedAxes...)
}

// ConvertDType of x to dtype.
// If x is already of the given dtype, it's a no-op.
func ConvertDType(x *Node, dtype dtypes.DType) (node *Node) {
	if x.DType() == dtype {
		return x
	}
	return backendConvertDType(x, dtype)
}

// Transpose returns x with the axes axisA and axisB transposed.
func Transpose(x *Node, axisA, axisB int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	rank := x.Rank()
	dims := []int{axisA, axisB}
	for ii, dim := range dims {
		if dim < 0 {
			dims[ii] = rank + dim
		}
		if dims[ii] > rank || dims[ii] < 0 {
			exceptions.Panicf("in Transpose(x, %d, %d), passed dimension %d which is out-of-limits for x rank %d",
				axisA, axisB, dim, rank)
		}
	}
	permutation := make([]int, x.Rank())
	for dimIdx := range permutation {
		permutation[dimIdx] = dimIdx
	}
	permutation[dims[0]], permutation[dims[1]] = dims[1], dims[0]
	return TransposeAllAxes(x, permutation...)
}

// TransposeAllAxes allows one to transpose any or all dimensions.
// It permutes the operand axes with the given permutation, so ∀ i, 0 ≤ i < rank ⇒ input_dimensions[permutations[i]] = output_dimensions[i].
func TransposeAllAxes(x *Node, permutations ...int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	rank := x.Rank()
	if len(permutations) != rank {
		exceptions.Panicf(
			"in TransposeAllAxes(x, %v), there must be one permutations per dimension in x, but x rank %d",
			permutations,
			rank,
		)
	}
	used := make([]bool, rank)
	for ii, idx := range permutations {
		if idx < 0 {
			idx = rank + idx
			permutations[ii] = idx
		}
		if idx >= rank || idx < 0 {
			exceptions.Panicf(
				"in TransposeAllAxes(x, %v), element %d id is %d which is out-of-limits for x rank %d",
				permutations,
				ii,
				idx,
				rank,
			)
		}
		if used[idx] {
			exceptions.Panicf("in TransposeAllAxes(x, %v), id %d appears more than once", permutations, idx)
		}
	}
	return backendTranspose(x, permutations...)
}

// TransposeAllDims is a deprecated alias to TransposeAllAxes.
//
// Deprecated: use TransposeAllDims instead.
func TransposeAllDims(x *Node, permutations ...int) *Node {
	return TransposeAllAxes(x, permutations...)
}

// Einsum evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions.
// This version uses a textual description on how to manipulate the axes.
// See EinsumAxes for a version where the axes are given numerically.
//
// This is inspired on numpy Einsum, a description of which can be seen in
// https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428.
//
// The equation string describes what to do with each dimension, for each operand,
// separated by ",", and the format of the result after the "->" describes what is to be made
// for each dimension.
//
// Examples:
//
// * `Einsum("ij,jk->ik", matrixA, matrixB)` performs the usual matrix multiplication.
// * `Einsum("bij,bjk->bik", batchedMatrixA, batchedMatrixB)` performs a batched matrix multiplication.
// * `Einsum("i,i->", vectorA, vectorB)` performs a dot product.
// * `Einsum("i,j->ij", vectorA, vectorB)` performs an outer (cross) product between two vectors.
//
// It also works for higher dimension tensors. Dimensions missing on the output (after "->") are
// reduce-summed.
//
// More examples in TensorFlow documentation:
// https://www.tensorflow.org/api_docs/python/tf/einsum
//
// Notice though that this Einsum is only defined for operations between 2 operands:
//
// - `lhs`: left-hand-side operand.
// - `rhs`: right-hand-side operand.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// consider trying both sides.
func Einsum(equation string, lhs, rhs *Node) *Node {
	_ = validateBuildingGraphFromInputs(lhs, rhs)

	// Parse equation.
	inOutParts := strings.Split(equation, "->")
	if len(inOutParts) != 2 {
		exceptions.Panicf(
			"Einsum(%q) missing or too many \"->\" separating inputNodes from outputs, there must be only one",
			equation,
		)
	}
	outputDesc, err := newEinsumOperandDesc(inOutParts[1])
	if err != nil {
		panic(err)
	}
	equationInputs := strings.Split(inOutParts[0], ",")
	if len(equationInputs) != 2 {
		exceptions.Panicf(
			"Einsum(%q) equation describes %d operands (separated by \",\"), but 2 operands (lhs and rhs) required",
			equation,
			len(equationInputs),
		)
	}
	operandsDesc := make([]einsumOperandDesc, 2)
	for ii, str := range equationInputs {
		operandsDesc[ii], err = newEinsumOperandDesc(str)
		if err != nil {
			panic(errors.WithMessagef(err, "when parsing operand %d", ii))
		}
	}

	// First, independently contract axes that only appear in one operand and not in the output.
	for opIdx, opPtr := range []**Node{&lhs, &rhs} {
		var newDesc einsumOperandDesc
		var contracting []int
		thisDesc, otherDesc := operandsDesc[opIdx], operandsDesc[1-opIdx]
		for axisIdx, axis := range thisDesc {
			if otherDesc.hasAxis(axis) || outputDesc.hasAxis(axis) {
				newDesc = append(newDesc, axis)
				continue
			}
			contracting = append(contracting, axisIdx)
		}
		if len(contracting) > 0 {
			//operandNames := []string{"lhs", "rhs"}
			//fmt.Printf("\tEinsum: independently contracting dimensions (%s): %v\n", operandNames[opIdx], contracting)
			// Contract dimensions.
			*opPtr = ReduceSum(*opPtr, contracting...)
			operandsDesc[opIdx] = newDesc
		}
	}

	// Calculate parameters for the dotGeneralXLA, and the order of its output — if
	// the order of `DotGeneral`'s output is different from the requested in `outputDesc`
	// we need to do a final transposition of the axes.
	lhsDesc := operandsDesc[0]
	rhsDesc := operandsDesc[1]
	var lhsBatchAxes, lhsContractingAxes, rhsBatchAxes, rhsContractingAxes []int
	var outputBatchAxes, outputCrossAxes einsumOperandDesc // dotGeneralXLA order of outputs.

	// Start from lhs: all axes that feature in both `lhs` and `rhs` are already taken care in
	// this loop.
	for lhsAxisIdx, axis := range lhsDesc {
		if rhsDesc.hasAxis(axis) {
			rhsAxisIdx := rhsDesc.axisIndex(axis)
			if outputDesc.hasAxis(axis) {
				// Batch dimension.
				lhsBatchAxes = append(lhsBatchAxes, lhsAxisIdx)
				rhsBatchAxes = append(rhsBatchAxes, rhsAxisIdx)
				outputBatchAxes = append(outputBatchAxes, axis)
			} else {
				// Contracting dimension.
				lhsContractingAxes = append(lhsContractingAxes, lhsAxisIdx)
				rhsContractingAxes = append(rhsContractingAxes, rhsAxisIdx)
			}
		} else {
			// Axis only exists on lhs and in the output: because axes that only
			// exist in one operand and nowhere else have already been contracted
			// earlier.
			//
			// This is a cross/outer product axes, the default for dotGeneralXLA.
			outputCrossAxes = append(outputCrossAxes, axis)
		}
	}

	// Loop in rhs: only missing those axes that only feature in rhs.
	for _, axis := range rhsDesc {
		if !lhsDesc.hasAxis(axis) {
			// This is a cross/outer product axes, the default for dotGeneralXLA.
			outputCrossAxes = append(outputCrossAxes, axis)
		}
	}

	// dotGeneralXLA will calculate the einsum, but the output may still be on the wrong
	// order.
	dotOutputDesc := outputBatchAxes
	if len(outputCrossAxes) > 0 {
		dotOutputDesc = append(dotOutputDesc, outputCrossAxes...)
	}

	output := DotGeneral(lhs, lhsContractingAxes, lhsBatchAxes,
		rhs, rhsContractingAxes, rhsBatchAxes)

	// Calculate the target permutation.
	permutation := make([]int, 0, output.Rank())
	hasPermutation := false
	for toAxisIdx, axis := range outputDesc {
		fromAxisIdx := dotOutputDesc.axisIndex(axis)
		permutation = append(permutation, fromAxisIdx)
		if fromAxisIdx != toAxisIdx {
			hasPermutation = true
		}
	}
	if hasPermutation {
		output = TransposeAllAxes(output, permutation...)
	}
	return output
}

type einsumOperandDesc []rune

func newEinsumOperandDesc(str string) (einsumOperandDesc, error) {
	e := make(einsumOperandDesc, 0, len(str))
	for _, r := range str {
		if e.hasAxis(r) {
			return nil, errors.Errorf("operands description (%q) has axis %q appearing more than once", str, r)
		}
		e = append(e, r)
	}
	return e, nil
}

func (e einsumOperandDesc) hasAxis(axis rune) bool {
	for _, r := range e {
		if r == axis {
			return true
		}
	}
	return false
}

func (e einsumOperandDesc) axisIndex(axis rune) int {
	for ii, r := range e {
		if r == axis {
			return ii
		}
	}
	return -1
}

// EinsumAxes evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions. Similar to Einsum, but it uses the explicit numeric
// axis, as opposed to a textual description.
//
// There are two operands: `lhs` (left-hand-side) and `rhs` (right-hand-side). The default for
// every axis is to do a cross-product, and the resulting tensor will have the concatenated shape (`lhs`
// dimensions first then `rhs` dimensions).
//
// One can specify contractionAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be contracted: these dimensions will multiplied and summed one at a time. That's what happens in
// the usual "dot product."
//
// One can also specify batchAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be considered as independently, as a batch dimension. These dimensions will show up in the same
// position as the `lhs`.
//
// Examples:
//
//   - `EinsumAxes(matrixA, matrixB, [][2]int{{1, 0}}, nil)` performs the usual matrix multiplication, where
//     we contract axis 1 of `matrixA` with axis 0 of `matrixB`.
//   - `EinsumAxes(batchedMatrixA, batchedMatrixB, [][2]int{{2, 1}}, [][2]int{{0, 0}})` is similar, but we
//     use axis 0 of both inputNodes as a batch, and following 2 axes as a matrix multiplication.
//   - `EinsumAxes(vectorA, vectorB, nil, nil)` performs an outer (cross) product -- no contractions, no batch.
//   - `EinsumAxes(vectorA, vectorB, [][2]int{{0, 0}}, nil)` performs a dot product and returns a scalar.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// Consider trying both sides.
func EinsumAxes(lhs, rhs *Node, contractingAxes, batchAxes [][2]int) (output *Node) {
	_ = validateBuildingGraphFromInputs(lhs, rhs)
	lhsRank := lhs.Rank()
	rhsRank := rhs.Rank()

	// Create function to process both, contractingAxes and batchAxes.
	lhsSeen := sets.Make[int](lhsRank)
	rhsSeen := sets.Make[int](rhsRank)
	normalizePairs := func(name string, pairs [][2]int) (lhsAxes, rhsAxes []int) {
		if len(pairs) == 0 {
			return
		}
		lhsAxes = make([]int, 0, len(contractingAxes))
		rhsAxes = make([]int, 0, len(contractingAxes))
		for _, pair := range pairs {
			lhsAxis, rhsAxis := adjustAxisToRank(pair[0], lhsRank), adjustAxisToRank(pair[1], rhsRank)

			if lhsAxis < 0 || lhsAxis >= lhs.Rank() {
				exceptions.Panicf("EinsumAxes %s has out-of-bound axis for left-hand-side operand: %v",
					name, pairs)
			}
			if lhsSeen.Has(lhsAxis) {
				exceptions.Panicf(
					"EinsumAxes %s axis for left-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v",
					name,
					pairs,
				)
			}
			lhsSeen.Insert(lhsAxis)

			if rhsAxis < 0 || rhsAxis >= rhs.Rank() {
				exceptions.Panicf("EinsumAxes %s has out-of-bound axis for right-hand-side operand: %v", name, pairs)
			}
			if rhsSeen.Has(rhsAxis) {
				exceptions.Panicf(
					"EinsumAxes %s axis for right-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v",
					name,
					pairs,
				)
			}
			rhsSeen.Insert(rhsAxis)

			lhsAxes = append(lhsAxes, lhsAxis)
			rhsAxes = append(rhsAxes, rhsAxis)
		}
		return
	}

	lhsContractingAxes, rhsContractingAxes := normalizePairs("contractingAxes", contractingAxes)
	lhsBatchAxes, rhsBatchAxes := normalizePairs("batchAxes", batchAxes)

	// Execute DotGeneral with parameters.
	return backendDotGeneral(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
func DotGeneral(
	lhs *Node,
	lhsContractingAxes, lhsBatchAxes []int,
	rhs *Node,
	rhsContractingAxes, rhsBatchAxes []int,
) *Node {
	_ = validateBuildingGraphFromInputs(lhs, rhs)
	lhsContractingAxes = adjustAxesToRank(lhs.Rank(), lhsContractingAxes, "lhsContractingAxes")
	lhsBatchAxes = adjustAxesToRank(lhs.Rank(), lhsBatchAxes, "lhsBatchAxes")
	rhsContractingAxes = adjustAxesToRank(rhs.Rank(), rhsContractingAxes, "rhsContractingAxes")
	rhsBatchAxes = adjustAxesToRank(rhs.Rank(), rhsBatchAxes, "rhsBatchAxes")
	return backendDotGeneral(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
}

// InternalBatchNormForTraining is a wrapper to the backend function.
// Don't use this directly, instead use layers.BatchNormalization.
func InternalBatchNormForTraining(
	operand *Node,
	scale *Node,
	offset *Node,
	epsilon float32,
	axis int,
) (normalized, batchMean, batchVariance *Node) {
	_ = validateBuildingGraphFromInputs(operand, scale, offset)
	dtype := operand.DType()
	if scale.DType() != dtype || offset.DType() != dtype {
		exceptions.Panicf(
			"InternalBatchNormForTraining: operand (%s), scale (%s) and offset (%s) must all have the same DType",
			operand.Shape(),
			scale.Shape(),
			offset.Shape(),
		)
	}
	axis = adjustAxisToRank(axis, operand.Rank())
	return backendBatchNormForTraining(operand, scale, offset, epsilon, axis)
}

// InternalBatchNormForInference is a wrapper to the backend function.
// Don't use this directly, instead use layers.BatchNormalization.
func InternalBatchNormForInference(
	operand *Node,
	scale *Node,
	offset *Node,
	mean *Node,
	variance *Node,
	epsilon float32,
	axis int,
) (node *Node) {
	_ = validateBuildingGraphFromInputs(operand, scale, offset, mean, variance)
	dtype := operand.DType()
	if scale.DType() != dtype || offset.DType() != dtype {
		exceptions.Panicf(
			"InternalBatchNormForInference: operand (%s), scale (%s) and offset (%s) must all have the same DType",
			operand.Shape(),
			scale.Shape(),
			offset.Shape(),
		)
	}
	axis = adjustAxisToRank(axis, operand.Rank())
	return backendBatchNormForInference(operand, scale, offset, mean, variance, epsilon, axis)
}

// InternalBatchNormGradient is a wrapper to the backend function.
// Don't use this directly, instead use layers.BatchNormalization.
func InternalBatchNormGradient(
	operand *Node,
	scale *Node,
	mean *Node,
	variance *Node,
	gradOutput *Node,
	epsilon float32,
	axis int,
) (gradOperand, gradScale, gradOffset *Node) {
	_ = validateBuildingGraphFromInputs(operand, scale, mean, variance)
	dtype := operand.DType()
	if scale.DType() != dtype || mean.DType() != dtype || variance.DType() != dtype {
		exceptions.Panicf(
			"InternalBatchNormForInference: operand (%s), scale (%s), mean (%s) and variance (%s) must all have the same DType",
			operand.Shape(),
			scale.Shape(),
			mean.Shape(),
			variance.DType(),
		)
	}
	axis = adjustAxisToRank(axis, operand.Rank())
	return backendBatchNormGradient(operand, scale, mean, variance, gradOutput, epsilon, axis)
}

// MatMul is the `numpy.matmul` equivalent, for those used to that.
//
// It is similar to Dot but extends to allow for more batch dimensions in lhs or rhs operand, and
// does broadcasting (of all but the last 2 axes) according to the numpy broadcasting rules.
//
// It's popular hence it is here, but full of edge cases, consider using DotGeneral instead.
func MatMul(lhs, rhs *Node) *Node {
	_ = validateBuildingGraphFromInputs(lhs, rhs)
	if lhs.Rank() == 0 || rhs.Rank() == 0 {
		exceptions.Panicf("MatMul expects two tensors with rank > 0, got ranks %d and %d", lhs.Rank(), rhs.Rank())
	}
	if lhs.Rank() <= 2 && rhs.Rank() <= 2 {
		return Dot(lhs, rhs)
	}

	// Special case when one of operands is a vector.
	if lhs.Rank() == 1 {
		return DotGeneral(lhs, []int{0}, nil, rhs, []int{rhs.Rank() - 2}, nil)
	}
	if rhs.Rank() == 1 {
		return DotGeneral(lhs, []int{lhs.Rank() - 1}, nil, rhs, []int{0}, nil)
	}

	// Trivial and most common case: right-hand-side is simply a linear transformation (matrix) on the last axis of lhs.
	if rhs.Rank() == 2 {
		return DotGeneral(lhs, []int{lhs.Rank() - 1}, nil, rhs, []int{rhs.Rank() - 2}, nil)
	}

	// Generic version, that will include broadcasting: we will use Einsum (and not DotGeneral) because it will do
	// the final transposing of the axes, where needed.
	//
	// . All axes before the last 2 are "batch":
	// . If one of the axes is not present, it should be effectively broadcast on the other:
	// . Batch axes appear first, then axis lhs[rank-2] and then rhs[rank-1].
	const lhsIdx, rhsIdx = 0, 1
	rhsRemap := make(map[int]int)              // Maps a rhs axis to match a lhs axis.
	rhsRemap[rhs.Rank()-2] = lhs.Rank() - 1    // The contracting axes.
	var lhsSqueezedAxes, rhsSqueezedAxes []int // Axes with dimension 1 that will be broadcast, they are squeezed away.
	letterForAxis := func(side int, axis int) string {
		var offset int
		if side == lhsIdx {
			if slices.Index(lhsSqueezedAxes, axis) >= 0 {
				// Axis has been dropped.
				return ""
			}
		} else {
			if slices.Index(rhsSqueezedAxes, axis) >= 0 {
				// Axis has been dropped.
				return ""
			}
			if lhsAxis, found := rhsRemap[axis]; found {
				// Use the lhs axis letter for the rhsAxis: either they are contracting or they are a batch dimension.
				axis = lhsAxis
				offset = 0
			} else {
				offset = lhs.Rank()
			}
		}
		return string('a' + rune(offset+axis))
	}
	var outputAxesLetters string

	var lhsBatchAxes, rhsBatchAxes []int
	minRank := min(rhs.Rank(), lhs.Rank())
	for axis := lhs.Rank() - minRank; axis < lhs.Rank()-2; axis++ {
		lhsBatchAxes = append(lhsBatchAxes, axis)
	}
	for axis := rhs.Rank() - minRank; axis < rhs.Rank()-2; axis++ {
		rhsBatchAxes = append(rhsBatchAxes, axis)
	}

	// First axes of the output are the "batch" axes not present in the other side.
	// Only one of the two for-loop belows will run.
	for axis := range lhs.Rank() - minRank {
		outputAxesLetters += letterForAxis(lhsIdx, axis)
	}
	for axis := range rhs.Rank() - minRank {
		outputAxesLetters += letterForAxis(rhsIdx, axis)
	}

	// Process common batch axes:
	for idx := range len(lhsBatchAxes) {
		leftAxis := lhsBatchAxes[idx]
		rightAxis := rhsBatchAxes[idx]
		leftAxisDim := lhs.Shape().Dimensions[leftAxis]
		rightAxisDim := rhs.Shape().Dimensions[rightAxis]
		if leftAxisDim == rightAxisDim {
			// Same batch axis on both sides:
			rhsRemap[rightAxis] = leftAxis
			outputAxesLetters += letterForAxis(lhsIdx, leftAxis)
			continue
		}
		if leftAxisDim != 1 && rightAxisDim != 1 {
			exceptions.Panicf("MatMul cannot match batch dimensions of lhs (left-hand-side) axis #%d (dim=%d) "+
				" and rhs (right-hand-side) axis #%d (dim=%d), for lhs.shape=%s and rhs.shape=%s",
				leftAxis, leftAxisDim, rightAxis, rightAxisDim, lhs.Shape(), rhs.Shape())
		}
		if leftAxisDim == 1 {
			lhsSqueezedAxes = append(lhsSqueezedAxes, leftAxis)
			outputAxesLetters += letterForAxis(rhsIdx, rightAxis)
		} else { // rightAxisDim == 1
			rhsSqueezedAxes = append(rhsSqueezedAxes, rightAxis)
			outputAxesLetters += letterForAxis(lhsIdx, leftAxis)
		}
	}

	// Final output axes
	outputAxesLetters += letterForAxis(lhsIdx, lhs.Rank()-2)
	outputAxesLetters += letterForAxis(rhsIdx, rhs.Rank()-1)

	// List lhs and rhs axes as letters:
	var lhsLetters, rhsLetters string
	for axis := range lhs.Rank() {
		lhsLetters += letterForAxis(lhsIdx, axis)
	}
	for axis := range rhs.Rank() {
		rhsLetters += letterForAxis(rhsIdx, axis)
	}

	// Squeeze unused 1-dimensional axes:
	lhsSqueezed := lhs
	if len(lhsSqueezedAxes) > 0 {
		lhsSqueezed = Squeeze(lhs, lhsSqueezedAxes...)
	}
	rhsSqueezed := rhs
	if len(rhsSqueezedAxes) > 0 {
		rhsSqueezed = Squeeze(rhs, rhsSqueezedAxes...)
	}

	equation := fmt.Sprintf("%s,%s->%s", lhsLetters, rhsLetters, outputAxesLetters)
	return Einsum(equation, lhsSqueezed, rhsSqueezed)
}
