/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph

import (
	"fmt"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
	"reflect"
	"sort"
	"strings"
)

const (
	MaxSizeToPrint = 5
)

// Node implements Node and is a standard node implementation that using a xla.SerializedNode definition
// can be used by most ops (node types).
//
// Almost every new node type implementation will rely on the Node.
type Node struct {
	graph          *Graph
	shape          shapes.Shape
	id             NodeId // id within graph.
	xlaHandle      NodeXlaHandle
	serializedNode *xla.SerializedNode
	inputs         []*Node
	logMessage     string // if != "" the node is marked for logging.
	stopGradient   bool   // if true, no gradient is passed through.

	trace error // Stack-trace error of where Node was created. Stored if graph.traced is true.
}

// NodeType identify the operation performed by the node.
func (n *Node) NodeType() xla.NodeType { return n.serializedNode.Type }

// Graph that holds this Node.
func (n *Node) Graph() *Graph {
	if n == nil {
		return nil
	}
	return n.graph
}

// Shape of the output of the Node. It can be nil, for nodes that simply have
// a side effect, like a "Print" Node.
func (n *Node) Shape() shapes.Shape {
	if n == nil {
		return shapes.Shape{}
	}
	return n.shape
}

// DType returns the DType of the node's shape.
func (n *Node) DType() shapes.DType {
	return n.shape.DType
}

// Rank returns the rank fo the node's shape.
func (n *Node) Rank() int {
	return n.shape.Rank()
}

// XlaHandle is used internally to refer to the node counterpart in the XLA implementation.
func (n *Node) XlaHandle() NodeXlaHandle { return n.xlaHandle }

// Id is the unique id of this node within the Graph.
func (n *Node) Id() NodeId {
	return n.id
}

// ParameterHandle returns the parameter id in the graph. Returns InvalidParameterHandle if node is not a parameter.
func (n *Node) ParameterHandle() ParameterHandle {
	if !n.Ok() {
		return InvalidParameterHandle
	}
	if n.NodeType() != xla.ParameterNode {
		return InvalidParameterHandle
	}
	return ParameterHandle(n.serializedNode.Int)
}

const NotAParameterStr = "NOT_A_PARAMETER"

// ParameterName returns the parameter name, if this node is a parameter.
func (n *Node) ParameterName() string {
	if !n.Ok() {
		return NotAParameterStr
	}
	if n.NodeType() != xla.ParameterNode {
		return NotAParameterStr
	}
	return n.serializedNode.Str

}

// Inputs are the other nodes that are direct inputs to the node. This doesn't include static inputs for some
// operations, that are not given by other nodes.
func (n *Node) Inputs() []*Node { return n.inputs }

// Ok indicates whether the Node was created successfully.
func (n *Node) Ok() bool {
	return n != nil && n.graph != nil && n.graph.Ok() && n.id >= 0
}

// SetLogged indicates that a node should be logged by executors.
func (n *Node) SetLogged(message string) {
	n.logMessage = message
}

// IsLogged returns whether node is marked to be logged.
func (n *Node) IsLogged() bool {
	return n.logMessage != ""
}

// LogMessage associated with node, if any.
func (n *Node) LogMessage() string {
	return n.logMessage
}

// Trace returns stack-trace in form of an error, of when the node was created.
// Only available if enabled by `Graph.SetTraced(true)`.
func (n *Node) Trace() error {
	return n.trace
}

// String implements Stringer interface. Logged nodes are marked with (*).
func (n *Node) String() (str string) {
	if !n.Ok() {
		return "InvalidNode"
	}
	if n.serializedNode == nil {
		return fmt.Sprintf("???(id=%d, xlaHandle=%d)", n.id, n.xlaHandle)
	}
	if n.serializedNode.Type == xla.InvalidNode {
		str = "NoOp"
	} else {
		str = n.serializedNode.Type.String()
	}
	inputIds := make([]NodeId, 0, len(n.inputs))
	for _, inputNode := range n.inputs {
		inputIds = append(inputIds, inputNode.Id())
	}
	str = fmt.Sprintf("%s(id=%d, xlaHandle=%d, inputs=%v", str, n.id, n.xlaHandle, inputIds)
	if !n.serializedNode.Literal.IsNil() {
		dataV := reflect.ValueOf(n.serializedNode.Literal.Data())
		var ellipsis string
		if dataV.Len() > MaxSizeToPrint {
			ellipsis = "..."
			dataV = dataV.Slice(0, MaxSizeToPrint)
		}
		str = fmt.Sprintf("%s, literal=%v%s", str, dataV.Interface(), ellipsis)
	}
	str = fmt.Sprintf("%s) -> %s", str, n.shape)
	if n.logMessage != "" {
		str = str + "\t[Logged]"
	}
	if n.stopGradient {
		str = str + "\t[StopGradient]"
	}
	return
}

// newNode creates a Node based on the serializedNode information, registers it in the Graph and
// adds it into the underlying xla computation (Const++'s xla::XlaBuilder). Returns a Node which implements
// the Node interface.
//
// Almost every new node type implementation will rely on the Node.
func newNode(graph *Graph, serializedNode *xla.SerializedNode, inputs []*Node) (node *Node) {
	if !graph.Ok() {
		return graph.InvalidNode()
	}
	node = &Node{
		graph:          graph,
		serializedNode: serializedNode,
		xlaHandle:      InvalidNodeXlaHandle,
	}
	node.id = graph.registerNode(node)
	node.setInputs(inputs)
	var err error
	var opsNum int
	opsNum, node.shape, err = graph.comp.AddOp(serializedNode)
	if err != nil {
		fmt.Printf("Failed to add op: %+v\n", err)
		graph.SetError(errors.WithStack(err))
	} else {
		node.xlaHandle = NodeXlaHandle(opsNum)
	}
	if graph.traced {
		node.trace = errors.Errorf("Stack-trace")
	}
	return
}

func (n *Node) setInputs(inputs []*Node) {
	n.inputs = inputs
	if len(inputs) > 0 {
		handles := make([]int32, 0, len(inputs))
		for _, input := range inputs {
			handles = append(handles, int32(input.XlaHandle()))
		}
		n.serializedNode.NodeInputs = handles
	} else {
		n.serializedNode.NodeInputs = nil
	}
}

// newNoOpNode is similar to newNode, but creates a node with no associated XLA operation,
// whose output is its input.
func newNoOpNode(graph *Graph, serializedNode *xla.SerializedNode, input *Node) (node *Node) {
	if !graph.Ok() {
		return graph.InvalidNode()
	}
	node = &Node{
		graph:          graph,
		serializedNode: serializedNode,
	}
	node.id = graph.registerNode(node)
	node.xlaHandle = input.XlaHandle()
	node.setInputs([]*Node{input})
	node.shape = input.shape.Copy()
	if graph.traced {
		node.trace = errors.Errorf("Stack-trace")
	}
	return
}

// ConstLocal returns a newly created constant node for the tensor x.
func ConstLocal(g *Graph, x *tensor.Local) *Node {
	if !g.Ok() {
		return (*Node)(nil)
	}
	literal := x.Literal()
	if literal.IsNil() {
		g.SetErrorf("undefined tensor given to ConstLocal")
		return (*Node)(nil)
	}
	return newNode(g, &xla.SerializedNode{
		Type:    xla.ConstantNode,
		Literal: literal,
	}, nil)
}

// Const creates constant nodes in the Graph. It can take a tensor as well as
// multidimensional slices (or scalars).
// It uses tensor.FromAnyValue to figure out the shape given a Go scalar/slice/array.
// If the value is unsupported, it sets the error in the Graph.
//
// A tensor.Device (e.g., generated by another computation) will be converted to local first.
// If you are creating very large constants that don't need to be materialized locally, consider
// instead storing them as variables in the context, or as a side parameter.
func Const(g *Graph, x any) *Node {
	if !g.Ok() {
		return g.InvalidNode()
	}
	if _, ok := x.(*Node); ok {
		g.SetErrorf("Const(g, x) can only take actual values, not another computation graph `*Node` -- " +
			"for that you don't need Const(), just use it directly.")
		return g.InvalidNode()
	}
	valueT := tensor.FromAnyValue(x)
	if valueT.Error() != nil {
		g.SetErrorf("failed converting value to Tensor: %w", valueT.Error())
		return g.InvalidNode()
	}
	return ConstLocal(g, valueT.Local())
}

// ConstAsDType creates a constant of the given DType. It adds the convenience
// of converting x (slice or scalar) to the appropriate type.
// E.g:
//
//	Pi := ConstScalar(g, myDType, math.Pi)
//	PiAndE := ConstScalar(g, myDType, []float64{math.Pi, math.E})
func ConstAsDType(g *Graph, dtype shapes.DType, x interface{}) *Node {
	if !g.Ok() {
		return g.InvalidNode()
	}
	if dtype == shapes.InvalidDType {
		g.SetErrorf("invalid DType given for ConstAsDType")
		return g.InvalidNode()
	}
	return Const(g, shapes.CastAsDType(x, dtype))
}

// ConstAs creates a constant (slice or scalar) of the same DType and on the same Graph as
// the given base.
func ConstAs(base *Node, x interface{}) *Node {
	return ConstAsDType(base.Graph(), base.DType(), x)
}

// NoOp creates a new Node whose output equals the input. No new XLA op is created, so no
// costs are actually impose.
func NoOp(x *Node) *Node {
	g := x.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNoOpNode(g, &xla.SerializedNode{
		Type: xla.InvalidNode,
	}, x)
}

// StopGradient creates a new NoOp Node, through which gradients don't back-propagate. No new XLA op is created, so no
// costs are actually impose.
func StopGradient(x *Node) *Node {
	g := x.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	n := NoOp(x)
	n.stopGradient = true
	return n
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given dimension. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func Iota(g *Graph, shape shapes.Shape, iotaDimension int) *Node {
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.IotaNode,
		Shape: shape,
		Int:   iotaDimension,
	}, nil)
}

// IotaFull creates a constant of the given shape with increasing numbers for all values.
// So `IotaFull([2,2])` returns `[[0 1][2 3]]`.
func IotaFull(g *Graph, shape shapes.Shape) *Node {
	if !g.Ok() {
		return g.InvalidNode()
	}
	return ReshapeWithShape(Iota(g, shapes.Make(shape.DType, shape.Size()), 0), shape)
}

// validateGraphFromInputs checks that all inputs are of the same Graph and that
// the Graph has no error. Returns the Graph, or set the error on the graphs
// and returns nil.
func validateGraphFromInputs(inputs ...*Node) (g *Graph) {
	if len(inputs) == 0 {
		return nil
	}
	if inputs[0] == nil {
		return nil
	}

	// Checks that all inputs are of the same graph.
	var err error
	for ii, n := range inputs {
		if g == nil {
			g = n.Graph()
		} else {
			if n.Graph() != g {
				err = errors.WithStack(errors.Errorf("combining nodes from different graphs not allowed"))
				break
			}
		}
		if !n.Ok() {
			err = errors.WithStack(errors.Errorf("invalid node for input %d", ii))
			continue
		}
	}
	if err != nil {
		for _, n := range inputs {
			if n.Ok() {
				n.Graph().SetError(err)
			}
		}
	}
	return
}

// oneArgNode is a helper function that implements ops that simply take 1 input.
func oneArgNode(nodeType xla.NodeType, x *Node) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{Type: nodeType}, []*Node{x})
}

// Abs adds to the graph the corresponding operation on the input node x.
func Abs(x *Node) *Node { return oneArgNode(xla.AbsNode, x) }

// Neg adds to the graph negative of the given node x.
func Neg(x *Node) *Node { return oneArgNode(xla.NegNode, x) }

// Exp adds to the graph the corresponding operation on the input node x.
func Exp(x *Node) *Node { return oneArgNode(xla.ExpNode, x) }

// Expm1 adds to the graph the corresponding operation on the input node x.
func Expm1(x *Node) *Node { return oneArgNode(xla.Expm1Node, x) }

// Floor adds to the graph the corresponding operation on the input node x.
func Floor(x *Node) *Node { return oneArgNode(xla.FloorNode, x) }

// Ceil adds to the graph the corresponding operation on the input node x.
func Ceil(x *Node) *Node { return oneArgNode(xla.CeilNode, x) }

// Round adds to the graph the corresponding operation on the input node x.
func Round(x *Node) *Node { return oneArgNode(xla.RoundNode, x) }

// Log adds to the graph the corresponding operation on the input node x.
func Log(x *Node) *Node { return oneArgNode(xla.LogNode, x) }

// Log1p adds to the graph the corresponding operation on the input node x.
func Log1p(x *Node) *Node { return oneArgNode(xla.Log1pNode, x) }

// Not adds to the graph the corresponding operation on the input node x.
func Not(x *Node) *Node { return oneArgNode(xla.LogicalNotNode, x) }

// Logistic returns a node with $1/(1+exp(-x))$. Alias to the Sigmoid function.
func Logistic(x *Node) *Node { return oneArgNode(xla.LogisticNode, x) }

// Sigmoid returns a node with $1/(1+exp(-x))$. Alias to the Logistic function.
func Sigmoid(x *Node) *Node { return Logistic(x) }

// Sign adds to the graph the corresponding operation on the input node x.
func Sign(x *Node) *Node { return oneArgNode(xla.SignNode, x) }

// Clz adds to the graph the "count leading zeroes" operation on the input node x.
func Clz(x *Node) *Node { return oneArgNode(xla.ClzNode, x) }

// Cos adds to the graph the corresponding operation on the input node x.
func Cos(x *Node) *Node { return oneArgNode(xla.CosNode, x) }

// Sin adds to the graph the corresponding operation on the input node x.
func Sin(x *Node) *Node { return oneArgNode(xla.SinNode, x) }

// Tanh adds to the graph the corresponding operation on the input node x.
func Tanh(x *Node) *Node { return oneArgNode(xla.TanhNode, x) }

// Sqrt adds to the graph the corresponding operation on the input node x.
func Sqrt(x *Node) *Node { return oneArgNode(xla.SqrtNode, x) }

// RSqrt adds the 1/sqrt(x) operation to the graph.
func RSqrt(x *Node) *Node { return oneArgNode(xla.RsqrtNode, x) }

// twoArgsNode is a helper function that implements ops that simply take 2 inputs.
func twoArgsNode(nodeType xla.NodeType, x, y *Node) *Node {
	g := validateGraphFromInputs(x, y)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if x.shape.DType != y.shape.DType {
		g.SetErrorf("operands of %s have different dtypes (%s and %s)", nodeType, x.shape.DType, y.shape.DType)
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{Type: nodeType}, []*Node{x, y})
}

// Add adds a node that sums the two nodes.
// Standard broadcasting rules apply (see documentation).
func Add(x, y *Node) *Node { return twoArgsNode(xla.AddNode, x, y) }

// Mul adds a node that multiplies the two nodes.
// Standard broadcasting rules apply (see documentation).
func Mul(x, y *Node) *Node { return twoArgsNode(xla.MulNode, x, y) }

// Sub adds to the graph the corresponding operation on the two input nodes x and y.
// Standard broadcasting rules apply (see documentation).
func Sub(x, y *Node) *Node { return twoArgsNode(xla.SubNode, x, y) }

// Div adds to the graph the corresponding operation on the two input nodes x and y.
// Standard broadcasting rules apply (see documentation).
func Div(x, y *Node) *Node { return twoArgsNode(xla.DivNode, x, y) }

// Mod adds to the graph the module operation on the two input nodes x and y.
// Standard broadcasting rules apply (see documentation).
func Mod(x, y *Node) *Node { return twoArgsNode(xla.RemNode, x, y) }

// And adds to the graph the corresponding operation on the two input nodes x and y.
// Only integer types.
// Standard broadcasting rules apply (see documentation).
func And(x, y *Node) *Node { return twoArgsNode(xla.AndNode, x, y) }

// Or adds to the graph the corresponding operation on the two input nodes x and y.
// Only integer types.
// Standard broadcasting rules apply (see documentation).
func Or(x, y *Node) *Node { return twoArgsNode(xla.OrNode, x, y) }

// Xor adds to the graph the corresponding operation on the two input nodes x and y.
// Only integer types.
// Standard broadcasting rules apply (see documentation).
func Xor(x, y *Node) *Node { return twoArgsNode(xla.XorNode, x, y) }

// Max returns element-wise the max from lhs and rhs.
// Standard broadcasting rules apply (see documentation).
func Max(lhs, rhs *Node) *Node { return twoArgsNode(xla.MaxNode, lhs, rhs) }

// Min returns the min from lhs and rhs for each element.
// Standard broadcasting rules apply (see documentation).
func Min(lhs, rhs *Node) *Node { return twoArgsNode(xla.MinNode, lhs, rhs) }

// Pow adds lhs^(rhs) to the graph.
// Standard broadcasting rules apply (see documentation).
func Pow(lhs, rhs *Node) *Node { return twoArgsNode(xla.PowNode, lhs, rhs) }

// Equal returns the element-wise operation to the graph.
//
//	// Standard broadcasting rules apply (see documentation).                                                                                                                //
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func Equal(x, y *Node) *Node { return twoArgsNode(xla.EqualNode, x, y) }

// NotEqual returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func NotEqual(x, y *Node) *Node { return twoArgsNode(xla.NotEqualNode, x, y) }

// GreaterOrEqual returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func GreaterOrEqual(x, y *Node) *Node { return twoArgsNode(xla.GreaterOrEqualNode, x, y) }

// GreaterThan returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func GreaterThan(x, y *Node) *Node { return twoArgsNode(xla.GreaterThanNode, x, y) }

// LessOrEqual returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func LessOrEqual(x, y *Node) *Node { return twoArgsNode(xla.LessOrEqualNode, x, y) }

// LessThan returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func LessThan(x, y *Node) *Node { return twoArgsNode(xla.LessThanNode, x, y) }

// EqualTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func EqualTotalOrder(x, y *Node) *Node { return twoArgsNode(xla.EqualTotalOrderNode, x, y) }

// NotEqualTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func NotEqualTotalOrder(x, y *Node) *Node { return twoArgsNode(xla.NotEqualTotalOrderNode, x, y) }

// GreaterOrEqualTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func GreaterOrEqualTotalOrder(x, y *Node) *Node {
	return twoArgsNode(xla.GreaterOrEqualTotalOrderNode, x, y)
}

// GreaterThanTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func GreaterThanTotalOrder(x, y *Node) *Node { return twoArgsNode(xla.GreaterThanTotalOrderNode, x, y) }

// LessOrEqualTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func LessOrEqualTotalOrder(x, y *Node) *Node { return twoArgsNode(xla.LessOrEqualTotalOrderNode, x, y) }

// LessThanTotalOrder returns the element-wise operation to the graph.
//
// Standard broadcasting rules apply (see documentation).
//
// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
func LessThanTotalOrder(x, y *Node) *Node { return twoArgsNode(xla.LessThanTotalOrderNode, x, y) }

// Dot adds to the graph the corresponding operation on the two input nodes x and y.
// The exact semantics of this operation depend on the ranks of the operands:
//
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
//
// lhs -> left-hand-side; rhs -> right-hand-side
// The operation performs sum of products over the second dimension of lhs (or the first if it has rank 1) and
// the first dimension of rhs. These are the "contracted" dimensions. The contracted dimensions
// of lhs and rhs must be of the same size. In practice, it can be used to perform dot products between vectors,
// vector/matrix multiplications or matrix/matrix multiplications.
func Dot(lhs, rhs *Node) *Node { return twoArgsNode(xla.DotNode, lhs, rhs) }

// BroadcastPrefix adds dimensions to an array by duplicating the data in the array.
//
// The new dimensions dims are inserted on the left, i.e. if
// broadcast_sizes has values {a0, ..., aN} and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
//
// The new dimensions id into copies of the operand, i.e.
//
//	output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
func BroadcastPrefix(x *Node, dims []int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.BroadcastNode,
		Ints: dims,
	}, []*Node{x})
}

// ExpandAndBroadcast combines ExpandDims and Broadcast of `x`, broadcasting it to the shape
// given in newDimensions. Only new expanded axes or axes with dimension 1 can be broadcast
// to new dimensions.
//
// newDimensions should have a rank larger than the rank of x, and the new axes in newDimensions
// should be listed in expandedAxes. In other words: `x.Rank() + len(expandedAxes) == len(newDimensions)`.
//
// For example:
//
//		  x = Const(g, []int32{10, 20})
//	   ExpandAndBroadcast(x, []int{2, 2}, []int{0})  // -> [][]int32{{10, 20}, {10, 20}}
//	   ExpandAndBroadcast(x, []int{2, 2}, []int{0})  // -> [][]int32{{10, 10}, {20, 20}}
func ExpandAndBroadcast(x *Node, newDimensions []int, expandedAxes []int) (output *Node) {
	g := validateGraphFromInputs(x)
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}
	if x.Rank()+len(expandedAxes) != len(newDimensions) {
		g.SetErrorf("there must be exactly one expandedAxes (%v) for each new axis in newDimensions (%v) -- x.shape=%s",
			expandedAxes, newDimensions, x.shape)
		return
	}

	// Verify values of expandedAxis and create a map of the expanded axis.
	expandedMap := make([]bool, len(newDimensions))
	for ii, axis := range expandedAxes {
		if axis < 0 {
			axis = len(newDimensions) + axis
		}
		if axis < 0 || axis >= len(newDimensions) {
			g.SetErrorf("expandedAxes (%v) defines a value out-of-range (%d-th value -> %d), they must be between 0 and len(newDimensions)=%d",
				expandedAxes, ii, axis, len(newDimensions))
			return
		}
		if expandedMap[axis] {
			g.SetErrorf("expandedAxes (%v) repeats an axis (%d-th value -> %d), they must be all unique and between 0 and len(newDimensions)=%d",
				expandedAxes, ii, axis, len(newDimensions))
			return
		}
		expandedMap[axis] = true
	}

	var preservedAxes []int
	if !x.Shape().IsScalar() {
		preservedAxes = make([]int, 0, x.Rank())
		for axis := 0; axis < len(newDimensions); axis++ {
			if !expandedMap[axis] {
				preservedAxes = append(preservedAxes, axis)
			}
		}
	}

	return broadcastInDim(x, shapes.Make(x.DType(), newDimensions...), preservedAxes)
}

// broadcastInDim broadcasts x to an output with the given shape.
// broadcastDims are the dimensions to be broadcasting into, i.e., the
// i-th dimension of x is mapped to the broadcastDim[i]-th dimension of the output.
// This also requires that the i-th input dimension is either 1 or is the same as the
// output dimension it's broadcasting into.
//
// This is part the XLA API, prefer using BroadcastAndExpand instead.
//
// For example, say operand `x = (s32)[2]{1, 2}`; shape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcast_dimension will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcast_dimension
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
//
// This interface is cumbersome, so instead we expose
func broadcastInDim(x *Node, shape shapes.Shape, broadcastDims []int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if x.Rank() != len(broadcastDims) {
		g.SetErrorf("there must be one broadcastDim for each axis of x in broadcastInDim, but x.shape=%s and broadcastDims=%v",
			x.shape, broadcastDims)
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.BroadcastInDimNode,
		Shape: shape,
		Ints:  broadcastDims,
	}, []*Node{x})
}

// BroadcastToShape broadcasts x to the given shape. They must both have the
// same rank, and the dimensions in x being broadcast (that is, where its corresponding
// dimension is shape is different) must be of size 1.
//
// One exception is if x is a scalar, in which case it can be broadcast to any shape.
func BroadcastToShape(x *Node, shape shapes.Shape) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if shape.DType != x.shape.DType {
		g.SetErrorf("cannot change dtype (from %s to %s) with BroadcastWithShape", x.shape.DType, shape.DType)
		return g.InvalidNode()
	}
	if x.shape.IsScalar() && shape.IsScalar() {
		// Assume nothing to do.
		return x
	}

	if x.shape.IsScalar() {
		return broadcastInDim(x, shape, nil)
	}
	broadcastDims := make([]int, shape.Rank())
	for ii := 0; ii < shape.Rank(); ii++ {
		broadcastDims[ii] = ii
	}
	return broadcastInDim(x, shape, broadcastDims)
}

// BroadcastToDims broadcasts x to the given dimensions. They must both have the
// same rank, and the dimensions in x being broadcast (that is, where it's corresponding requested
// dimension is different) must be of size 1.
//
// This is a convenient wrapper for BroadcastToShape.
func BroadcastToDims(x *Node, dimensions ...int) *Node {
	shape := shapes.Make(x.DType(), dimensions...)
	return BroadcastToShape(x, shape)
}

// ConvertType converts x to a different primitive type. See shapes.Supported for the supported types.
func ConvertType(x *Node, dtype shapes.DType) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() || !x.Ok() {
		return g.InvalidNode()
	}
	if !dtype.IsSupported() {
		g.SetErrorf("converting to an unsupported type %s", dtype)
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.ConvertTypeNode,
		Int:  int(dtype),
	}, []*Node{x})
}

// Where takes element-wise values from onTrue or onFalse depending on the value of condition (expected to be boolean).
func Where(condition, onTrue, onFalse *Node) *Node {
	g := validateGraphFromInputs(condition)
	if !g.Ok() || !condition.Ok() {
		return g.InvalidNode()
	}
	if condition.DType() != shapes.Bool {
		g.SetErrorf("Where(condition, onTrue, onFalse) requires condition to be of dtype Bool, got %s instead", condition.Shape())
		return g.InvalidNode()
	}
	if !onTrue.Shape().Eq(onFalse.Shape()) {
		g.SetErrorf("Where(condition, onTrue, onFalse) requires onTrue (%s) and onFalse (%s) to be the same shape", onTrue.Shape(), onFalse.Shape())
		return g.InvalidNode()
	}
	if condition.Rank() > onTrue.Rank() || !reflect.DeepEqual(condition.Shape().Dimensions, onTrue.Shape().Dimensions[:condition.Rank()]) {
		g.SetErrorf("Where(condition, onTrue, onFalse) requires condition (%s) dimensions to be the same or a prefix to onTrue (%s) and onFalse (%s) dimensions", condition.Shape(), onTrue.Shape(), onFalse.Shape())
		return g.InvalidNode()
	}
	if condition.Rank() < onTrue.Rank() {
		// If condition's shape is a prefix to onTrue and onFalse, simply broadcast to
		// their shape. This allows masks to work for embeddings which has one extra
		// axis.
		extraAxes := onTrue.Rank() - condition.Rank()
		condition = ExpandDims(condition, slices.SliceWithValue(extraAxes, -1)...)
		condition = BroadcastToDims(condition, onTrue.Shape().Dimensions...)
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.WhereNode,
	}, []*Node{condition, onTrue, onFalse})
}

// Reshape x to the given dimensions. Total size cannot change. One dimension can be left as -1,
// in which case it will be set to match the size, if possible.
func Reshape(x *Node, dimensions ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() || !x.Ok() {
		return g.InvalidNode()
	}

	totalSize := x.Shape().Size()
	newSize := 1
	missingIdx := -1
	for idx, dim := range dimensions {
		if dim != -1 {
			newSize *= dim
		} else {
			if missingIdx != -1 {
				g.SetErrorf("only one dimension can be missing (that is, set to -1) for Reshape, %v given", dimensions)
				return g.InvalidNode()
			}
			missingIdx = idx
		}
	}
	if missingIdx != -1 {
		// Make copy of dimensions, so original is not changed.
		tmpDim := make([]int, len(dimensions))
		copy(tmpDim, dimensions)
		dimensions = tmpDim
		dimensions[missingIdx] = totalSize / newSize
		newSize *= dimensions[missingIdx]
	}
	if newSize != totalSize {
		g.SetErrorf("total requested size %d (dimensions=%v) doesnt match original size %d (dimensions %v)",
			newSize, dimensions, totalSize, x.Shape().Dimensions)
	}
	return ReshapeWithShape(x, shapes.Make(x.shape.DType, dimensions...))
}

// ReshapeWithShape reshapes x to the dimensions given by shape. Total size cannot change, neither
// the DType is allowed to change. Conceptually, this is a limited form of "shape casting".
func ReshapeWithShape(x *Node, shape shapes.Shape) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() || !x.Ok() || !shape.Ok() {
		return g.InvalidNode()
	}
	if shape.DType != x.shape.DType {
		g.SetErrorf("cannot change dtype (from %s to %s) with ReshapeWithShape", x.shape.DType, shape.DType)
		return g.InvalidNode()
	}
	if shape.Size() != x.shape.Size() {
		g.SetErrorf("shapes have different total sizes (from %d to %d), reshape not possible", x.shape.Size(), shape.Size())
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.ReshapeNode,
		Shape: shape,
	}, []*Node{x})
}

// ExpandDims expands x creating new axes just before the axes given. If axes[ii] < 0, then they
// are counted from the end -- -1 represents a new axis after the end of the original shape. The new axes
// will be of dimension 1 (so the total size of and contents of the tensor remains the same), and the rank
// is increased by `len(axes)`.
//
// Maybe it should be called ExpandAxes ... but to follow Tensorflow nomenclature.
func ExpandDims(x *Node, axes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if len(axes) == 0 {
		// Trivial case, noop.
		return x
	}

	// Ranks.
	fromRank := x.shape.Rank()
	toRank := fromRank + len(axes)

	// Copy dimensions, so we don't change callers values, and replace negatives.
	newAxes := make([]int, len(axes))
	copy(newAxes, axes)
	axes = newAxes
	for ii, axis := range axes {
		if axis < 0 {
			axes[ii] = fromRank + 1 + axis
		}
	}
	sort.Ints(axes)

	// Create new target shape.
	toShape := shapes.Shape{DType: x.shape.DType, Dimensions: make([]int, toRank)}
	iiOriginal, iiNewAxes := 0, 0
	for ii := range toShape.Dimensions {
		if iiNewAxes < len(axes) && axes[iiNewAxes] <= iiOriginal || iiOriginal == fromRank {
			toShape.Dimensions[ii] = 1
			iiNewAxes += 1
		} else {
			toShape.Dimensions[ii] = x.shape.Dimensions[iiOriginal]
			iiOriginal += 1
		}
	}
	return ReshapeWithShape(x, toShape)
}

// ExpandLeftToRank prepend axes of dimension 1 to x, until it reaches rank `newRank`.
func ExpandLeftToRank(x *Node, newRank int) (output *Node) {
	g := validateGraphFromInputs(x)
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}
	if newRank < x.Rank() {
		g.SetErrorf("ExpandLeftToRank(newRank=%d), but x already has rank %d", newRank, x.Rank())
		return
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
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}

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
				g.SetErrorf("Squeeze() for x.shape=%s, axis %d is out-of-range", x.Shape(), axes[axisIdx])
				return g.InvalidNode()
			}
			if newDims[axis] == 0 {
				g.SetErrorf("Squeeze() for x.shape=%s, axis %d was selected twice!?", x.Shape(), axes[axisIdx])
				return g.InvalidNode()
			}
			if newDims[axis] != 1 {
				g.SetErrorf("Squeeze() for x.shape=%s, axis %d does not have dimension 1", x.Shape(), axes[axisIdx])
				return g.InvalidNode()
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

// Tuple creates a tuple of several values. It's the means to returns several values from one Graph computation.
func Tuple(nodes ...*Node) *Node {
	g := validateGraphFromInputs(nodes...)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.TupleNode,
	}, nodes)
}

// GetTupleElement extracts one element from a Tuple.
func GetTupleElement(tuple *Node, index int) *Node {
	g := validateGraphFromInputs(tuple)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.GetTupleElementNode,
		Int:  index,
	}, []*Node{tuple})
}

// SplitTuple is a convenience wrapper around GetTupleElement, it will return an array with all the nodes.
func SplitTuple(tuple *Node) []*Node {
	g := validateGraphFromInputs(tuple)
	numElements := tuple.Shape().TupleSize()
	nodes := make([]*Node, numElements)
	for ii := 0; ii < numElements; ii++ {
		if g.Ok() {
			nodes[ii] = GetTupleElement(tuple, ii)
		} else {
			nodes[ii] = g.InvalidNode()
		}
	}
	return nodes
}

// reduceHelper helps implements all the Reduce<X> functions.
func reduceHelper(x, init *Node, reduceAxes []int, nodeType xla.NodeType) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}

	// Convert negative dimensions.
	return newNode(g, &xla.SerializedNode{
		Type: nodeType,
		Ints: convertNegativeDimensionsAndSort(x.shape.Rank(), reduceAxes),
	}, []*Node{x, init})
}

// ArgMax returns the index of the largest element across the given axis.
//
// The selected axis is reduced, and the output has one fewer axes (rank `x.Rank() - 1`).
// The output `DType`, if not given, is `shapes.I32`.
//
// Ties are resolved by returning the smallest index.
func ArgMax(x *Node, axis int, outputDType ...shapes.DType) (output *Node) {
	g := validateGraphFromInputs(x)
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}

	dtype := shapes.I32
	if len(outputDType) > 1 {
		g.SetErrorf("ArgMax takes at most one outputDType, %d values given", len(outputDType))
		return
	} else if len(outputDType) == 1 {
		dtype = outputDType[0]
	}
	return argMinMax(x, axis, dtype, false)
}

// ArgMin returns the index of the smallest element across the given axis.
//
// The selected axis is reduced, and the output has one fewer axes (rank `x.Rank() - 1`).
// The output `DType`, if not given, is `shapes.I32`.
//
// Ties are resolved by returning the smallest index.
func ArgMin(x *Node, axis int, outputDType ...shapes.DType) (output *Node) {
	g := validateGraphFromInputs(x)
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}

	dtype := shapes.I32
	if len(outputDType) > 1 {
		g.SetErrorf("ArgMin takes at most one outputDType, %d values given", len(outputDType))
		return
	} else if len(outputDType) == 1 {
		dtype = outputDType[0]
	}
	return argMinMax(x, axis, dtype, true)
}

func argMinMax(x *Node, axis int, outputDType shapes.DType, isMin bool) (output *Node) {
	g := validateGraphFromInputs(x)
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}

	adjustedAxis := AdjustAxis(x, axis)
	if !g.Ok() {
		return
	}

	output = newNode(g, &xla.SerializedNode{
		Type: xla.ArgMinMaxNode,
		Int:  adjustedAxis,
		Ints: []int{boolToInt(isMin), int(outputDType)},
	}, []*Node{x})

	// We don't define a gradient for the ArgMax result. It's a discrete quantity, not
	// something one usually wants to differentiate from anyway.
	// Presumably, it's either 0 or undefined if there are another more than one element with the max value.
	return StopGradient(output)
}

// convertNegativeDimensionsAndSort in a copy of dimsWithNegatives.
func convertNegativeDimensionsAndSort(rank int, dimsWithNegatives []int) []int {
	copyDims := make([]int, len(dimsWithNegatives))
	copy(copyDims, dimsWithNegatives)
	for ii := range copyDims {
		if copyDims[ii] < 0 {
			copyDims[ii] = rank + copyDims[ii]
		}
	}
	sort.Ints(copyDims)
	return copyDims
}

// ReduceSum reduces by summing over the elements of the selected axes of the x.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
func ReduceSum(x *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	zero := ScalarZero(g, x.DType())
	return reduceHelper(x, zero, reduceAxes, xla.ReduceSumNode)
}

// ReduceAllSum reduces all dimensions to a scalar by summing.
func ReduceAllSum(x *Node) *Node {
	return ReduceSum(x)
}

// ReduceMaskedSum reduces by summing over the elements of the selected axes of the x.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// It ignores values for which the corresponding mask is false. mask and x must have the
// same shape.
func ReduceMaskedSum(x, mask *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x, mask)
	if !g.Ok() {
		return g.InvalidNode()
	}
	maskedX := Where(mask, x, ZerosLike(x))
	zero := ScalarZero(g, x.DType())
	return reduceHelper(maskedX, zero, reduceAxes, xla.ReduceSumNode)
}

// ReduceAllMaskedSum reduces all dimensions to a scalar by summing.
//
// It ignores values for which the corresponding mask is false.
// The `mask` and `x` values must have the same shape.
func ReduceAllMaskedSum(x, mask *Node) *Node {
	return ReduceMaskedSum(x, mask)
}

// ReduceMean reduces by taking the mean over the elements of the selected axes.
func ReduceMean(x *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	sum := ReduceSum(x, reduceAxes...)
	denominator := x.Shape().Size() / sum.Shape().Size()
	return Div(sum, ConstAs(sum, denominator))
}

// ReduceAllMean reduces all dimensions to a scalar by taking the mean.
func ReduceAllMean(x *Node) *Node {
	return ReduceMean(x)
}

// ReduceMultiply reduces by summing over the elements of the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
func ReduceMultiply(x *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	one := ScalarOne(g, x.DType())
	return reduceHelper(x, one, reduceAxes, xla.ReduceMultiplyNode)
}

// ReduceAllMultiply reduces all dimensions to a scalar by multiplying.
func ReduceAllMultiply(x *Node) *Node {
	return ReduceMultiply(x)
}

// ReduceMax reduces by taking the max over the elements of the selected axes.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
func ReduceMax(x *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	lowest := lowestForDType(g, x.DType())
	return reduceHelper(x, lowest, reduceAxes, xla.ReduceMaxNode)
}

// ReduceAllMax reduces all dimensions to a scalar by taking the max.
func ReduceAllMax(x *Node) *Node {
	return ReduceMax(x)
}

// ReduceMaskedMax reduces by taking the max over the elements of the selected axes of the x.
// If reduceAxes is nil, reduce over all dimensions to a scalar.
//
// It ignores values for which the corresponding mask is false. mask and x must have the
// same shape.
func ReduceMaskedMax(x, mask *Node, reduceAxes ...int) *Node {
	g := validateGraphFromInputs(x, mask)
	if !g.Ok() {
		return g.InvalidNode()
	}
	lowest := lowestForDType(g, x.DType())
	broadcastLowest := BroadcastToDims(lowest, x.Shape().Dimensions...)
	maskedX := Where(mask, x, broadcastLowest)
	return reduceHelper(maskedX, lowest, reduceAxes, xla.ReduceMaxNode)
}

// ReduceAllMaskedMax reduces all dimensions to a scalar by taking the max.
//
// It ignores values for which the corresponding mask is false. mask and x must have the
// same shape.
func ReduceAllMaskedMax(x, mask *Node) *Node {
	return ReduceMaskedMax(x, mask)
}

// RngNormal constructs an output of a given shape with random numbers generated following the normal
// distribution. The parameters mu and sigma, and output shape have to have a floating point
// elemental type. The parameters furthermore have to be scalar valued.
func RngNormal(mu, sigma *Node, shape shapes.Shape) *Node {
	g := validateGraphFromInputs(mu, sigma)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.RngNormalNode,
		Shape: shape,
	}, []*Node{mu, sigma})
}

// RngUniform constructs an output of a given shape with random numbers generated following the
// uniform distribution over the interval "$[a,b($". The parameters and output element type have
// to be a boolean type, an integral type or a floating point types, and the types have to be
// consistent. Furthermore, the parameters need to be scalar valued. If `b <= a` the result is
// implementation-defined.
func RngUniform(a, b *Node, shape shapes.Shape) *Node {
	g := validateGraphFromInputs(a, b)
	if !g.Ok() {
		return g.InvalidNode()
	}
	return newNode(g, &xla.SerializedNode{
		Type:  xla.RngUniformNode,
		Shape: shape,
	}, []*Node{a, b})
}

// AxisRangeDef defines the range of an axis to include in a Slice.
//
// Use AxisRange below to create it.
//
// Full means to include the whole range (and ignore Start/End), and
// NoEnd means from Start to the full dimension of the axis.
//
// Optional (if Stride != 0) it can set the stride for the axis as well.
//
// Consider using AxisRange below to construct AxisRangeDef values.
//
// TODO: Add strides.
type AxisRangeDef struct {
	Start, End, StrideValue int
	Full, NoEnd             bool
}

// Stride returns a copy of the AxisRangeDef with Stride set to the given stride.
func (ar AxisRangeDef) Stride(stride int) AxisRangeDef {
	ar2 := ar
	ar2.StrideValue = stride
	return ar2
}

// AxisRange creates a AxisRangeDef to be used in Slice. The indices can have 0, 1 or 2 elements:
// - If no elements are given, it's assumed to be full.
// - If one element is given, it's assumed to be the start, and the range should be taken to the end.
// - If two elements are given, they should be the start and end.
// - If more than 2 elements are given, they are ignored.
func AxisRange(indices ...int) AxisRangeDef {
	if len(indices) == 0 {
		return AxisRangeDef{Full: true}
	}
	if len(indices) == 1 {
		return AxisRangeDef{Start: indices[0], NoEnd: true}
	}
	return AxisRangeDef{Start: indices[0], End: indices[1]}
}

// adjustToDimension converts negative indices to a value starting at dimension.
func adjustToDimension(index, dimension int) int {
	if index >= 0 {
		return index
	}
	return dimension + index
}

// Slice take slices of the operand.
//
// Each axis can have a range defined as (start, end) pairs. Any axis for which a range
// is not specified is assumed to be taken in full. Consider using the shortcut AxisRange to define
// the ranges.
//
// Examples:
//
// - For `x = {1, 2, 3, 4}`:
//   - `Slice(x) = {1, 2, 3, 4}`  // AxisRangeDef not given is taken in full.
//   - `Slice(x, AxisRange()) = {1, 2, 3, 4}`  // Default for AxisRange is the full range.
//   - `Slice(x, AxisRange(2)) = {3, 4}`  // If only start is given, it is taken to the end.
//   - `Slice(x, AxisRange(1,-1)) = {2, 3}`  // Negative values are taken from the end of the axis dimension.
//
// - For `x = {{1, 2, 3}, {4, 5, 6}}`:
//   - `Slice(x, AxisRange(), AxisRange(0,1)) = {{1}, {4}}` // First axis taken in full, second axis only the first element.
//   - `Slice(x, AxisRange(1,2)) = {{4, 5, 6}}`  // Missing second AxisRangeDef, assumed to be taken in full.
//
// If Slice is called with `x.shape = [5, 5, 5, 5]` and `axesRanges=AxisRange(1,2), AxisRange(), AxisRange(2), AxisRange(0,2)`
// would return a node shaped `[1, 5, 3, 2]`.
//
// It also works with strides, use the AxisRangeDef.Stride() method to conveniently set it.
//
// Example:
//
// - For `x = {1, 2, 3, 4}`:
//   - `Slice(x, AxisRange().Stride(2)) = {1, 3}`  // The whole range, but with a stride of 2.
//
// - For `x = {{1, 2, 3}, {4, 5, 6}}`:
//   - `Slice(x, AxisRange().Stride(2), AxisRange(-1)) = {{3}}`  // Take every 2nd row (so only the 1st here), the last column.
func Slice(x *Node, axesRanges ...AxisRangeDef) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()

	if len(axesRanges) > rank {
		g.SetErrorf("Slice was given %d ranges, but x only has (rank) %d axes", len(axesRanges), rank)
	}
	starts := make([]int, rank)
	limits := make([]int, rank)
	strides := make([]int, rank)
	for ii, dim := range x.shape.Dimensions {
		// Start with full range.
		starts[ii] = 0
		limits[ii] = dim
		strides[ii] = 1
		if len(axesRanges) > ii && !axesRanges[ii].Full {
			starts[ii] = adjustToDimension(axesRanges[ii].Start, dim)
			if !axesRanges[ii].NoEnd {
				limits[ii] = adjustToDimension(axesRanges[ii].End, dim)
			}
		}
		if len(axesRanges) > ii && axesRanges[ii].StrideValue > 0 {
			strides[ii] = axesRanges[ii].StrideValue
		}
	}
	return SliceWithStridesXLA(x, starts, limits, strides)
}

// PadAxis defines the amount of padding preceding one axis (Start), at the end of axis (End)
// or in between the inputs (Interior). This is used as parameter for the Pad function.
type PadAxis struct {
	Start, End, Interior int
}

// Pad injects padding on the start, end or interior (in between each element) of the given operand.
// There must be at most `operand.Rank()` axesConfig values. Missing PadAxis are assumed to be zeros,
// that is, no padding for those axes.
func Pad(operand, fillValue *Node, axesConfig ...PadAxis) *Node {
	g := validateGraphFromInputs(operand, fillValue)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := operand.Rank()

	ints := make([]int, 0, 3*rank)
	for axis := 0; axis < rank; axis++ {
		var padding PadAxis
		if axis < len(axesConfig) {
			padding = axesConfig[axis]
		}
		ints = append(ints, padding.Start, padding.End, padding.Interior)
	}

	return newNode(g, &xla.SerializedNode{
		Type: xla.PadNode,
		Ints: ints,
	}, []*Node{operand, fillValue})
}

// Gather values in params from pointer in indices. The output are slices of
// `params` selected by `indices`, stitched together.
//
// Let's assume params has shape `[i_0, ..., i_M, s_0, ..., s_o]`, where:
//
//   - `i_0, ..., i_N` are the N "indexed dimensions", that is, the dimensions indexed by `indices`.
//   - `s_0, ..., s_S` are the S dimensions of the slices that are going to be "gathered" (copied over).
//
// And let's assume indices has shape `[o_0,...,o_O, N]`, where:
//
//   - `o_0, ..., o_O` are enumerations of the slices from `params` to gather. E.g: let's say O=1, and o_0=3,
//     that means there will be 3 slices to gather.
//   - Last dimension `N`: this is the number of indices in `params` to point to. `N` is the number of
//     dimensions indexed `i_0, ..., i_N` in `params` above.
//
// The output will have shape `[o_0,...,o_O, s_0, ... s_S]`, where:
//
//   - `o_0, ..., o_O` come from indices, and are enumerations of the slices from params to gather.
//   - `s_0, ..., s_S` are the slice sizes copied from params.
//
// For example:
//
//	params := [][]float32{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
//	indices := [][]int{{1}, {0}}
//	Gather(params, indices) would return {{3, 4, 5}, {0, 1, 2}}
//
// In the case above params shape is interpreted as `[i_0=3, s_0=3]`, and indices' shape is
// `[o_0=2, N=1]`. The output shape is `[o_0=2, s_0=3]`.
func Gather(params, indices *Node) *Node {
	g := validateGraphFromInputs(params, indices)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if params.shape.IsScalar() || params.shape.IsTuple() {
		g.SetErrorf("cannot Gather from scalar or tuple, params shape is %s", params.Shape())
		return g.InvalidNode()
	}
	if !indices.shape.DType.IsInt() {
		g.SetErrorf("Gather requires indices to have an integer type, got shape %q instead", indices.Shape())
		return g.InvalidNode()
	}

	// If indices is a scalar, simply convert it to shape `[1]`.
	if indices.Shape().IsScalar() {
		indices = ExpandDims(indices, 0)
		if !g.Ok() {
			return g.InvalidNode()
		}
	}

	// Check ranks are compatible.
	paramsRank := params.Rank()
	indicesRank := indices.Rank()
	indexedSubRank := indices.Shape().Dimensions[indicesRank-1] // N from documentation.
	slicesSubRank := paramsRank - indexedSubRank                // S from documentation, the slices dimensions.
	if slicesSubRank < 0 {
		g.SetErrorf("Gather params are \"over-indexed\": params has only rank %d and "+
			"indexed rank is %d (last dimension of indices)", paramsRank, indexedSubRank)
		return g.InvalidNode()
	}
	outputSubRank := indicesRank - 1

	// Construct call to gatherXLA:
	// * indexVectorDim is always the last one.
	indexVectorDim := indicesRank - 1
	// * startIndexMap is sequential and sorted
	startIndexMap := make([]int, indexedSubRank)
	for ii := 0; ii < indexedSubRank; ii++ {
		startIndexMap[ii] = ii
	}
	// * sliceSizes are 1 everywhere but on the sliced dimensions.
	// * collapsedSliceDims is set to collapse all dimensions set to 1.
	sliceSizes := make([]int, paramsRank)
	collapsedSliceDims := make([]int, indexedSubRank)
	for ii := 0; ii < paramsRank; ii++ {
		if ii < indexedSubRank {
			sliceSizes[ii] = 1
			collapsedSliceDims[ii] = ii
		} else {
			sliceSizes[ii] = params.Shape().Dimensions[ii]
		}
	}
	// * offsetDims are the dimensions indexed.
	offsetDims := make([]int, paramsRank-indexedSubRank)
	for ii := range offsetDims {
		offsetDims[ii] = outputSubRank + ii
	}

	// Make no assumptions about indices being sorted or unique.
	// TODO: add version where these can be set.
	return gatherXLA(params, indices, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes,
		false)
}

// GatherSlices from inputs. Each axis listed in slicedAxes have corresponding start position and size for each
// slice indexed by `start` (a graph Node, can be dynamically generated in the graph) and `sizes`, which will
// define the output final shape, and must be statically given.
//
// Axes in slicedAxes can be given as negative numbers, which are taken from the the end of the input rank --
// that is axis -1 means the last axis in the input. Axes not given in slicedAxes (and in `start` and `sizes`)
// are taken in full length.
//
// Axes in slicedAxes must be given sorted in increasing order.
//
// The output has a rank equal to the prefixing rank of `start` (== `start.Rank()-1`) plus the rank of `input`.
// And the shape will depend on the sizes of the slices.
//
//   - TODO:
//     Add an option to support batch axes, present in both the input and in the start indices. This
//     will need to automatically concatenate the batch index in the start Node as a iota of each
//     batch example, and add the size 1 slice. This can be done manually today.
//
// Example:
//
//		x := IotaFull(g, shapes.Make(shapes.F64, 3, 10, 10))  // 300 in total.
//		start := Const(g, [][]int32{{0, 3}, {1, 2}})  // 2 slices
//		sizes := []int{1, 3}
//		slices := GatherSlices(x, []int{1,2}, start, sizes)  // Axis=0 is taken in full.
//	    slices.AssertDims(2, 3, 1, 2)  // 2 slices, Axis=0 taken in full (3), and each slice of dimensions (1, 2).
//		// Result would be [][][][]int32{{{0, 1, 2, 3, 4}}, {{30, 31, 32, 33, 34}}, {{40, 41, 42, 43, 44}}}
func GatherSlices(input *Node, slicedAxes []int, start *Node, sizes []int) (gathered *Node) {
	g := validateGraphFromInputs(input, start)
	gathered = g.InvalidNode()
	if !g.Ok() {
		return
	}
	if input.shape.IsScalar() || input.shape.IsTuple() {
		g.SetErrorf("cannot GatherSlices from scalar or tuple, input shape is %s", input.Shape())
		return
	}
	if !start.shape.DType.IsInt() {
		g.SetErrorf("GatherSlices requires start indices to have an integer type, got shape %q instead", start.Shape())
		return
	}
	if start.shape.IsScalar() {
		start = ExpandDims(start, 0)
	}

	// Check ranks are compatible.
	inputRank := input.Rank()
	startRank := start.Rank()
	numSlicedAxes := len(slicedAxes)
	if len(sizes) != numSlicedAxes {
		g.SetErrorf("GatherSlices requires one value in sizes for each axis marked as slicedAxes -- slicedAxes=%v, sizes=%v", slicedAxes, sizes)
		return
	}
	if start.shape.Dimensions[startRank-1] != numSlicedAxes {
		g.SetErrorf("GatherSlices requires the last axis of `start` to be the same dimension as the slicedAxes, "+
			"so it takes one index value per axis to be sliced -- slicedAxes=%v, start.Shape()=%s",
			slicedAxes, start.Shape())
		return
	}
	outputPrefixRank := startRank - 1

	// Validate slicedAxes and normalizes it (replacing negative axis to their corresponding ones).
	{
		seen := types.MakeSet[int](numSlicedAxes)
		normalized := make([]int, 0, numSlicedAxes)
		for ii, axis := range slicedAxes {
			if axis < 0 {
				axis = inputRank + axis
			}
			if axis < 0 || axis >= inputRank {
				g.SetErrorf("GatherSlices got an invalid axis (%d) selected for slicing, input.Shape()=%s, slicedAxes=%v",
					slicedAxes[ii], input.Shape(), slicedAxes)
				return
			}
			if seen.Has(axis) {
				g.SetErrorf("GatherSlices got an axis (%d) selected twice for slicing, input.Shape()=%s, slicedAxes=%v",
					slicedAxes[ii], input.Shape(), slicedAxes)
				return
			}
			seen.Insert(axis)
			if ii > 0 && axis < normalized[ii-1] {
				g.SetErrorf("GatherSlices got an axis (%d) out-of-order, slicedAxes (%v) must be given in increasing order "+
					"(and `sizes` and `start` must match that order)",
					slicedAxes[ii], slicedAxes)
				return
			}
			normalized = append(normalized, axis)
		}
		slicedAxes = normalized
	}

	// Construct call to gatherXLA:
	// * indexVectorDim indicates the axis in the start that has the indices: it's always the last one.
	indexVectorDim := startRank - 1
	// * startIndexMap holds the axis in the input that are pointed by `start`. These are exactly the normalized slicedAxes.
	startIndexMap := slicedAxes
	// * sliceSizes must be defined for each input axis, and are either given in `sizes` or are assumed to be the full dimension
	//   of the input.
	sliceSizes := input.shape.Copy().Dimensions // Start with a copy of the full dimensions of the input.
	for ii, size := range sizes {
		axis := slicedAxes[ii]
		sliceSizes[axis] = size
	}

	// * offsetDims must point for each input axis that is not collapsed, the output Node. Since we don't collapse any of the
	//   input dimensions, all the input axes need to be mapped. Notice that this preserves the order of the axis given by
	//   the input (the order in `slicedAxes` will be ignored).
	offsetDims := make([]int, 0, numSlicedAxes)
	var collapsedSliceDims []int // Left empty.
	for ii := 0; ii < inputRank; ii++ {
		axis := ii + outputPrefixRank
		offsetDims = append(offsetDims, axis)
	}

	// Make no assumptions about indices being sorted or unique.
	// TODO: add version where these can be set.
	return gatherXLA(input, start, indexVectorDim, offsetDims, collapsedSliceDims, startIndexMap, sliceSizes, false)
}

// GatherWithBatchDims values in params from pointer in indices. It works exactly the same as tensorflow's gather_nd
// operation, described in https://www.tensorflow.org/api_docs/python/tf/gather_nd.
//
// Let's assume params has shape `[b_0,...,b_{batchDim}, i_0, ..., i_M, s_0, ..., s_o]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `i_0, ..., i_N` are the N "indexed dimensions", that is, the dimensions indexed by indices.
//   - `s_0, ..., s_S` are the S dimensions of the slices that are going to be "gathered" (copied over).
//
// And let's assume indices has shape `[b_0, ... b_{batchDim}, o_0,...,o_O, N]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `o_0, ..., o_O` are enumerations of the slices from params to gather. E.g: let's say O=1, and o_0=3,
//     that means there will be 3 slices to gather.
//   - Last dimension N: this is the number of indices in params to point to. N is the same value as
//     the dimension `i_0, ..., i_N` in params above.
//
// The output will have shape `[b_0, ... b_{batchDim}, o_0,...,o_O, s_0, ... s_S]`, where:
//
//   - `b_0, ..., b_{batchDim}` are the batchDims batch dimensions, dimensions that are shared
//     in params, indices and will also be present in the output.
//   - `o_0, ..., o_O` come from indices, and are enumerations of the slices from params to gather.
//   - `s_0, ..., s_S` are the slice sizes copied from params.
//
// See some examples in `node_test.go` function `TestGather`.
/*
func GatherWithBatchDims(params, indices *Node, batchDims int) *Node {
	g := validateGraphFromInputs(params, indices)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if params.shape.IsScalar() || params.shape.IsTuple() {
		g.SetErrorf("cannot Gather from scalar or tuple, params shape is %s", params.Shape())
		return g.InvalidNode()
	}
	if !indices.shape.DType.IsInt() {
		g.SetErrorf("Gather requires indices to have an integer type, got shape %q instead", indices.Shape())
	}

	// If indices is a scalar, simply convert it to shape `[1]`.
	if indices.Shape().IsScalar() {
		indices = ReshapeWithShape(indices, shapes.Make(indices.DType(), 1))
		if !g.Ok() {
			return g.InvalidNode()
		}
	}

	// Check ranks are compatible.
	paramsRank := params.Rank()
	indicesRank := indices.Rank()
	indexedSubRank := indices.Shape().Dimensions[indicesRank-1]
	numIndices := indices.shape.Size() / indexedSubRank
	slicesSubRank := paramsRank - batchDims - indexedSubRank
	if slicesSubRank < 0 {
		g.SetErrorf("Gather params are \"over-indexed\": params has only rank %d, batchDims=%d and "+
			"indexed rank is %d (last dimension of indices)", paramsRank, batchDims, indexedSubRank)
		return g.InvalidNode()
	}
	outputSubRank := indicesRank - 1 - batchDims
	if outputSubRank < 0 {
		g.SetErrorf("Gather indices don't have enough batch dimensions: indices rank is %d and "+
			"one dimension is needed for the indices themselves, but batchDims=%d", indicesRank, batchDims)
		return g.InvalidNode()
	}

	// Grow indices to include batch dimensions: "example" here mean one element of the batch
	// dimensions. This is because the underlying gatherXLA doesn't support batch dimensions.
	if batchDims > 0 {
		if !types.DeepSliceCmp(params.shape.Dimensions[0:batchDims], indices.shape.Dimensions[0:batchDims], types.Equal[int]) {
			g.SetErrorf("batch dimensions (first %d dimensions) from params (shape=%s) and indices (shape=%s) don't match",
				batchDims, params.shape, indices.shape)
			return g.InvalidNode()
		}
		batchIndices := IndicesForShape(g, shapes.Make(types.Int64, indices.shape.Dimensions[0:batchDims]))
		// Now batchIndices need to be broadcast to each id for the gather.
		... TODO: flatten batchIndices, broadcast it, and concatenate to a flattenedIndices, and
		... then reshape back. After that, just call the simpler Gather().
		flatIndices := ReshapeWithShape(indices)
	}
	return Gather(params, indices)
}
*/

// IndicesForShape enumerates a list of indices for all elements of the given shape. It will always
// return a node with shape [shape.Size(), shape.Rank()].
// E.g: if shape=[3, 2], it returns `[[0 0] [0 1] [1 0] [1 1] [2 0] [2 1]]`.
func IndicesForShape(g *Graph, shape shapes.Shape) *Node {
	if !g.Ok() {
		return g.InvalidNode()
	}
	if shape.IsScalar() {
		g.SetErrorf("can't generate IndicesForShape for scalars (shape=%s)", shape)
		return g.InvalidNode()
	}
	indices := Iota(g, shapes.Make(shapes.Int64, shape.Size()), 0)                             // [shape.Size()]
	indices = ExpandDims(indices, -1)                                                          // [shape.Size(), 1]
	indices = BroadcastToShape(indices, shapes.Make(shapes.Int64, shape.Size(), shape.Rank())) // [shape.Size(), shape.Rank()]

	dividers := make([]int, shape.Rank())
	dividers[shape.Rank()-1] = 1
	for ii := shape.Rank() - 2; ii >= 0; ii -= 1 {
		dividers[ii] = dividers[ii+1] * shape.Dimensions[ii+1]
	}
	//fmt.Printf("shape=%s, dividers=%v, size=%v\n", shape, dividers, shape.Size())
	indices = Div(indices, Const(g, [][]int{dividers}))
	indices = Mod(indices, Const(g, [][]int{shape.Dimensions}))
	return indices
}

func boolToInt(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

func intToBool(i int) bool {
	return i != 0
}

// Scatter sums up the slices in updates into a new tensor of the given shape, at the locations pointed by indices.
// It does the opposite of Gather.
func Scatter(indices, updates *Node, shape shapes.Shape) *Node {
	g := validateGraphFromInputs(indices, updates)
	if !g.Ok() {
		return g.InvalidNode()
	}
	zeros := Zeros(g, shape)
	return ScatterAdd(zeros, indices, updates)
}

// ScatterAdd adds up the slices in updates into the given operand tensor, at the locations pointed by indices.
// It does the opposite of Gather.
func ScatterAdd(operand, indices, updates *Node) *Node {
	g := validateGraphFromInputs(operand, indices, updates)
	if !g.Ok() {
		return g.InvalidNode()
	}

	if !indices.shape.DType.IsInt() {
		g.SetErrorf("scatter operations require integer indices, instead got shape %s", indices.shape)
		return g.InvalidNode()
	}
	if operand.shape.DType != updates.shape.DType {
		g.SetErrorf("scatter operations require operand and updates to have the same DType, instead got shapes %s (operand) and %s (updates)", operand.shape, updates.shape)
		return g.InvalidNode()
	}
	if indices.shape.IsTuple() || operand.shape.IsTuple() || updates.shape.IsTuple() {
		g.SetErrorf("tuples are not supported in ScatterAdd, operand.shape=%s, indices.shape=%s, updates.shape=%s",
			operand.shape, indices.shape, updates.shape)
		return g.InvalidNode()
	}
	if indices.shape.IsScalar() {
		indices = ExpandDims(indices, 0)
		if !g.Ok() {
			return g.InvalidNode()
		}
	}

	// Check shapes compatibility.
	indicesRank := indices.shape.Rank()
	indexedRank := indices.shape.Dimensions[indicesRank-1]
	updatesRank := updates.shape.Rank()
	if updatesRank < indicesRank-1 || !slices.DeepSliceCmp(updates.shape.Dimensions[:indicesRank-1], indices.shape.Dimensions[:indicesRank-1], slices.Equal[int]) {
		g.SetErrorf("updates rank prefix (shape=%s) must match the first n-1 dimensions of the indices (shape=%s)", updates.shape, indices.shape)
		return g.InvalidNode()
	}
	slicesRank := updatesRank - (indicesRank - 1)
	slicesDims := updates.shape.Dimensions[indicesRank-1:]
	operandRank := operand.shape.Rank()
	if operandRank != indexedRank+slicesRank || !slices.DeepSliceCmp(operand.shape.Dimensions[indexedRank:], slicesDims, slices.Equal[int]) {
		g.SetErrorf("operand shape (%s) has to be a combination of the indexed rank (%d, the last dimension of indices shape %s) and "+
			"the slices coming from updates (the last %d dimensions %v of the updates, shaped %s)",
			operand.shape, indexedRank, indices.shape, slicesRank, slicesDims, updates.shape)
		return g.InvalidNode()
	}

	// Set scatterXLA parameters:
	updateWindowsDims := make([]int, 0, slicesRank)
	for ii := updatesRank - slicesRank; ii < updatesRank; ii++ {
		updateWindowsDims = append(updateWindowsDims, ii)
	}
	insertedWindowDims := make([]int, 0, indexedRank)
	for ii := 0; ii < indexedRank; ii++ {
		insertedWindowDims = append(insertedWindowDims, ii)
	}
	scatterDimsToOperandDims := make([]int, 0, 10)
	for ii := 0; ii < indexedRank; ii++ {
		scatterDimsToOperandDims = append(scatterDimsToOperandDims, ii)
	}
	return scatterXLA(operand, indices, updates, indicesRank-1, updateWindowsDims, insertedWindowDims, scatterDimsToOperandDims,
		true, true)
}

// Concatenate results on the given axis. A negative axis will be counted from
// the end -- so `axis==-1` means the last axis.
func Concatenate(operands []*Node, axis int) *Node {
	g := validateGraphFromInputs(operands...)
	if !g.Ok() {
		return g.InvalidNode()
	}
	if len(operands) == 0 {
		g.SetErrorf("cannot ConcatenateDimensions 0 operands")
		return g.InvalidNode()
	}
	if len(operands) == 1 {
		// Trivial solution.
		return operands[0]
	}
	baseShape := operands[0].shape
	rank := baseShape.Rank()
	if axis >= rank || axis < -rank {
		g.SetErrorf("invalid axis %d for ConcatenateDimensions, where first element has rank %d",
			axis, rank)
		return g.InvalidNode()
	}
	if axis < 0 {
		axis = rank + axis
	}
	for ii, node := range operands[1:] {
		if node.shape.DType != baseShape.DType {
			g.SetErrorf("ConcatenateDimensions operand %d has different dtype (%s) than operand 0's dtype (%s)",
				ii+1, node.shape.DType, baseShape.DType)
			return g.InvalidNode()
		}
		if node.shape.Rank() != rank {
			g.SetErrorf("ConcatenateDimensions operand %d has different rank (%s) than operand 0's rank (%s)",
				ii+1, node.Shape(), operands[0].Shape())
			return g.InvalidNode()
		}
		for ii, nodeDim := range node.shape.Dimensions {
			if ii == axis {
				// Dimension being concatenated can be different.
				continue
			}
			if baseShape.Dimensions[ii] != nodeDim {
				g.SetErrorf("ConcatenateDimensions operand %d has incompatible shape (%s) with operand 0's shape (%s)",
					ii+1, node.shape, baseShape)
				return g.InvalidNode()
			}
		}
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.ConcatenateNode,
		Int:  axis,
	}, operands)
}

// concatenateVJP implements a VJP function for the ConcatenateNode operation.
func concatenateVJP(node, v *Node, _ shapes.Shape) []*Node {
	vjps := make([]*Node, 0, len(node.inputs))
	concatDimension := node.serializedNode.Int
	concatDimStart := 0
	shape := node.shape

	// Set starts and limits for slices that are shared among all concatenated inputs.
	starts, limits := make([]int, shape.Rank()), make([]int, shape.Rank())
	for dim := 0; dim < shape.Rank(); dim++ {
		if dim == concatDimension {
			continue
		}
		starts[dim], limits[dim] = 0, shape.Dimensions[dim]
	}

	// Take slice for each concatenated input.
	for _, input := range node.inputs {
		starts[concatDimension] = concatDimStart
		concatDimStart += input.shape.Dimensions[concatDimension]
		limits[concatDimension] = concatDimStart
		vjps = append(vjps, SliceXLA(v, starts, limits))
	}
	return vjps
}

// Reverse returns x with the values for the given dimensions reversed, that is,
// the value indexed at `i` will be swapped with the value at indexed `(dimension_size - 1 - i)`.
// The shape remains the same.
func Reverse(x *Node, axes ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()
	for ii, dim := range axes {
		if dim < 0 {
			axes[ii] = rank + dim
		}
		if axes[ii] > rank || axes[ii] < 0 {
			g.SetErrorf("in Reverse(x, axes...), passed axis %d which is out-of-limits for x rank %d", dim, rank)
		}
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.ReverseNode,
		Ints: axes,
	}, []*Node{x})
}

// Transpose returns x with the axes axisA and axisB transposed.
func Transpose(x *Node, axisA, axisB int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()
	dims := []int{axisA, axisB}
	for ii, dim := range dims {
		if dim < 0 {
			dims[ii] = rank + dim
		}
		if dims[ii] > rank || dims[ii] < 0 {
			g.SetErrorf("in Transpose(x, %d, %d), passed dimension %d which is out-of-limits for x rank %d", axisA, axisB, dim, rank)
		}
	}
	permutation := make([]int, x.Rank())
	for dimIdx := range permutation {
		permutation[dimIdx] = dimIdx
	}
	permutation[dims[0]], permutation[dims[1]] = dims[1], dims[0]
	return TransposeAllDims(x, permutation...)
}

// TransposeAllDims allows one to transpose any or all dimensions.
// It permutes the operand axes with the given permutation, so ∀ i, 0 ≤ i < rank ⇒ input_dimensions[permutations[i]] = output_dimensions[i].
func TransposeAllDims(x *Node, permutation ...int) *Node {
	g := validateGraphFromInputs(x)
	if !g.Ok() {
		return g.InvalidNode()
	}
	rank := x.shape.Rank()
	if len(permutation) != rank {
		g.SetErrorf("in TransposeAllDims(x, %v), there must be one permutation per dimension in x, but x rank %d", permutation, rank)
		return g.InvalidNode()
	}
	used := make([]bool, rank)
	for ii, idx := range permutation {
		if idx < 0 {
			idx = rank + idx
			permutation[ii] = idx
		}
		if idx >= rank || idx < 0 {
			g.SetErrorf("in TransposeAllDims(x, %v), element %d id is %d which is out-of-limits for x rank %d", permutation, ii, idx, rank)
			return g.InvalidNode()
		}
		if used[idx] {
			g.SetErrorf("in TransposeAllDims(x, %v), id %d appears more than once", permutation, idx)
			return g.InvalidNode()
		}
	}
	return newNode(g, &xla.SerializedNode{
		Type: xla.TransposeNode,
		Ints: permutation,
	}, []*Node{x})
}

// Einsum evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions. This version uses a textual description on
// how to manipulate the axes. See EinsumAxes for a version where the axes are given numerically.
//
// This is inspired on numpy Einsum, a description of which can be seen in
// https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428.
//
// The equation string describes what it to be made with each dimension, for each operand,
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
// reduce summed.
//
// More examples in TensorFlow documentation:
// https://www.tensorflow.org/api_docs/python/tf/einsum
//
// Notice though that this Einsum is only defined for operations between 2 operands:
//
// * lhs -> left-hand-side operand.
// * rhs -> right-hand-side operand.
//
// Important note: the order of the operands can have dramatic impact on the speed of the multiplications.
// consider trying both sides.
func Einsum(equation string, lhs, rhs *Node) *Node {
	//fmt.Printf("Einsum(%s, %s, %s)\n", equation, lhs.Shape(), rhs.Shape())
	g := lhs.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !lhs.Ok() {
		g.SetErrorf("Einsum(%q) left0hand-side operand is no ok", equation)
		return g.InvalidNode()
	}

	// Parse equation.
	inOutParts := strings.Split(equation, "->")
	if len(inOutParts) != 2 {
		g.SetErrorf("Einsum(%q) missing or too many \"->\" separating inputs from outputs, there must be only one", equation)
		return g.InvalidNode()
	}
	outputDesc, err := newEinsumOperandDesc(inOutParts[1])
	if err != nil {
		g.SetError(err)
		return g.InvalidNode()
	}
	equationInputs := strings.Split(inOutParts[0], ",")
	if len(equationInputs) != 2 {
		g.SetErrorf("Einsum(%q) equation describes %d operands (separated by \",\"), but 2 operands (lhs and rhs) required", equation, len(equationInputs))
		return g.InvalidNode()
	}
	operandsDesc := make([]einsumOperandDesc, 2)
	for ii, str := range equationInputs {
		operandsDesc[ii], err = newEinsumOperandDesc(str)
		if err != nil {
			g.SetError(errors.WithMessagef(err, "when parsing operand %d", ii))
			return g.InvalidNode()
		}
	}

	// First independently contract axes that only appear in one operand and not in the output.
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

	// Calculate parameters for the dotGeneralXLA, and the order of its output -- if
	// the order of the output of DotGeneral is different from the requested in outputDesc
	// we need to do a final transposition of the axes.
	lhsDesc := operandsDesc[0]
	rhsDesc := operandsDesc[1]
	var lhsBatchAxes, lhsContractingAxes, rhsBatchAxes, rhsContractingAxes []int
	var outputBatchAxes, outputCrossAxes einsumOperandDesc // dotGeneralXLA order of outputs.

	// Start from lhs: all axis that feature in both lhs and rhs are already taken care in
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
			// Axis only exists on lhs and in the output: because axis that only
			// exist in one operand and nowhere else have already been contracted
			// earlier.
			//
			// This is a cross/outer product axes, the default for dotGeneralXLA.
			outputCrossAxes = append(outputCrossAxes, axis)
		}
	}

	// Loop in rhs: only missing those axis that only feature in rhs.
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

	output := dotGeneralXLA(lhs, lhsContractingAxes, lhsBatchAxes,
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
		output = TransposeAllDims(output, permutation...)
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
// every axis is to do a cross product, and the resulting tensor will have the concatenated shape (`lhs`
// dimensions first then `rhs` dimensions).
//
// One can specify contractionAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be contracted: these dimensions will multiplied and summed one at a time. That's what happens in
// the usual "dot product".
//
// One can also specify batchAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be considered as independently, as a batch dimension. These dimensions will show up in the same
// position as the `lhs`.
//
// Examples:
//
//   - `EinsumAxes(matrixA, matrixB, [][2]int{{1, 0}}, nil)` performs the usual matrix multiplication, where
//     we contract the axis 1 of `matrixA` with the axis 0 of `matrixB`.
//   - `EinsumAxes(batchedMatrixA, batchedMatrixB, [][2]int{{2, 1}}, [][2]int{{0, 0}})` is similar, but we
//     use the axis 0 of both inputs as a batch, and following 2 axes as a matrix multiplication.
//   - `EinsumAxes(vectorA, vectorB, nil, nil)` performs an outer (cross) product -- no contractions, no batch.
//   - `EinsumAxes(vectorA, vectorB, [][2]int{{0, 0}}, nil)` performs a dot product and returns a scalar.
//
// Important note: the order of the operands can have dramatic impact on the speed of the multiplications.
// consider trying both sides.
func EinsumAxes(lhs, rhs *Node, contractingAxes, batchAxes [][2]int) (output *Node) {
	g := lhs.Graph()
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}
	if !lhs.Ok() {
		g.SetErrorf("EinsumAxes left-hand-side operand is in an error state")
		return g.InvalidNode()
	}
	if !rhs.Ok() {
		g.SetErrorf("EinsumAxes right-hand-side operand is in an error state")
		return g.InvalidNode()
	}

	// Create function to process both, contractingAxes and batchAxes.
	lhsSeen := types.MakeSet[int](rhs.Rank())
	rhsSeen := types.MakeSet[int](rhs.Rank())
	normalizePairs := func(name string, pairs [][2]int) (lhsAxes, rhsAxes []int) {
		if len(pairs) == 0 {
			return
		}
		lhsAxes = make([]int, 0, len(contractingAxes))
		rhsAxes = make([]int, 0, len(contractingAxes))
		for _, pair := range pairs {
			lhsAxis, rhsAxis := pair[0], pair[1]
			// Convert negative axis to distance from last (-1 is the last axis).
			if lhsAxis < 0 {
				lhsAxis = lhs.Rank() + lhsAxis
			}
			if rhsAxis < 0 {
				rhsAxis = rhs.Rank() + rhsAxis
			}
			// Check if axis are valid.
			if lhsAxis < 0 || lhsAxis >= lhs.Rank() {
				g.SetErrorf("EinsumAxes %s has out-of-bound axis for left-hand-side operand: %v", name, pairs)
				return
			}
			if lhsSeen.Has(lhsAxis) {
				g.SetErrorf("EinsumAxes %s axis for left-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v", name, pairs)
				return
			}
			lhsSeen.Insert(lhsAxis)
			if rhsAxis < 0 || rhsAxis >= rhs.Rank() {
				g.SetErrorf("EinsumAxes %s has out-of-bound axis for right-hand-side operand: %v", name, pairs)
				return
			}
			if rhsSeen.Has(rhsAxis) {
				g.SetErrorf("EinsumAxes %s axis for right-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v", name, pairs)
				return
			}
			rhsSeen.Insert(rhsAxis)

			lhsAxes = append(lhsAxes, lhsAxis)
			rhsAxes = append(rhsAxes, rhsAxis)
		}
		return
	}

	lhsContractingAxes, rhsContractingAxes := normalizePairs("contractingAxes", contractingAxes)
	if !g.Ok() {
		return
	}
	lhsBatchAxes, rhsBatchAxes := normalizePairs("batchAxes", batchAxes)
	if !g.Ok() {
		return
	}

	// Execute dotGeneralXLA
	output = dotGeneralXLA(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
	return
}
