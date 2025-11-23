package simplego

import (
	"reflect"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	compiled bool

	// nodes are only created when their inputs have already been created. So this is a natural DAG (Directed Acyclic Graph)
	// ordering of the graph. The executor rely on this invariance.
	nodes []*Node

	// inputs will have nodeParameter as data.
	inputs []*Node

	// outputs can be any type of node.
	outputs []*Node
}

// Compile-time check.
var _ backends.Builder = (*Builder)(nil)

// Name implements backends.Builder.
func (b *Builder) Name() string {
	return b.name
}

// Compile implements backends.Builder.
func (b *Builder) Compile(outputs []backends.Op, shardings []*backends.ShardingSpec) (backends.Executable, error) {
	if len(shardings) != 0 {
		return nil, errors.Errorf("sharding or distributed execution are not supported by SimpleGo backend")
	}
	var err error
	b.outputs, err = b.checkOps("Compile", outputs...)
	if err != nil {
		return nil, err
	}
	nodeSet := sets.MakeWith(b.outputs...)
	if len(nodeSet) != len(b.outputs) {
		return nil, errors.Errorf("*** Repeated outputs: %d outputs, %d unique outputs", len(b.outputs), len(nodeSet))
	}
	for _, node := range b.outputs {
		if len(node.multiOutputsShapes) != 0 {
			return nil, errors.Errorf(
				"%s node %q is internal (with multiple-outputs) and cannot be used for output",
				b.Name(),
				node.opType,
			)
		}
	}
	b.compiled = true
	return newExecutable(b), nil
}

// Finalize immediately release the resources associated with the Builder.
func (b *Builder) Finalize() {
	b.inputs = nil
	b.outputs = nil
	b.nodes = nil
}

// Node in the SimpleGo computation graph.
type Node struct {
	// builderIdx in Builder.nodes
	builderIdx int
	inputs     []*Node

	// shape of the output.
	opType  backends.OpType
	shape   shapes.Shape
	builder *Builder

	// multiOutputsShapes are set for a few specialized nodes.
	// For most nodes this is set to nil.
	multiOutputsShapes []shapes.Shape
	multiOutputsNodes  []*Node
	isNodeSelectOutput bool
	selectOutputIdx    int

	// data for the specific node type.
	data any
}

// newNode adds a new node of the given opType and shape to the Builder graph.
// It's used by the other ops when creating new nodes.
func (b *Builder) newNode(opType backends.OpType, shape shapes.Shape, inputs ...*Node) *Node {
	n := &Node{
		builder:    b,
		opType:     opType,
		builderIdx: len(b.nodes),
		shape:      shape,
		inputs:     slices.Clone(inputs),
	}
	b.nodes = append(b.nodes, n)
	return n
}

// newMultiOutputsNode create the multi-outputs node, and its "select nodes", one per output.
// The node.multiOutputsNodes will be set with the individual outputs and can be used by the Builder to return
// to the user.
func (b *Builder) newMultiOutputsNode(
	opType backends.OpType,
	outputShapes []shapes.Shape,
	inputs ...*Node,
) (node *Node) {
	node = b.newNode(opType, shapes.Invalid(), inputs...)
	node.multiOutputsShapes = outputShapes
	node.multiOutputsNodes = make([]*Node, len(outputShapes))
	for idx, shape := range outputShapes {
		node.multiOutputsNodes[idx] = &Node{
			builder:            b,
			opType:             opType,
			builderIdx:         len(b.nodes),
			shape:              shape,
			inputs:             []*Node{node},
			isNodeSelectOutput: true,
			selectOutputIdx:    idx,
		}
		b.nodes = append(b.nodes, node.multiOutputsNodes[idx])
	}
	return node
}

// IsMultiOutputs returns whether this node yields multiple outputs.
func (n *Node) IsMultiOutputs() bool {
	return len(n.multiOutputsShapes) > 0
}

// checkOps validates that the ops are from SimpleGo and from this builder.
// It also checks whether the Builder is not yet compiled.
func (b *Builder) checkOps(opType string, ops ...backends.Op) ([]*Node, error) {
	if b == nil {
		return nil, errors.Errorf("%s: Builder is nil (!?), cannot build a graph", opType)
	}
	if b.compiled {
		return nil, errors.Errorf("cannot add new op (%s) to Builder %q, it has already been compiled", opType, b.name)
	}
	nodes := make([]*Node, len(ops))
	var ok bool
	for idx, op := range ops {
		if op == nil {
			return nil, errors.Errorf("%s: input op #%d is nil!?", opType, idx)
		}
		nodes[idx], ok = op.(*Node)
		if !ok {
			return nil, errors.Errorf(
				"cannot use input op #%d in backend %q that was created on a different backend for %s",
				idx,
				b.backend.Name(),
				opType,
			)
		}
		if nodes[idx].builder != b {
			return nil, errors.Errorf(
				"%s: input op #%d was created with a different builder (%q), cannot use it with builder %q",
				opType,
				idx,
				nodes[idx].builder.name,
				b.name,
			)
		}
	}
	return nodes, nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	inputs, err := b.checkOps("OpShape", op)
	if err != nil {
		return shapes.Invalid(), err
	}
	return inputs[0].shape, nil
}

// checkFlat returns an error if flat is not a slice of one of the dtypes supported.
// It returns the supported dtype and the length of the flat slice.
func checkFlat(flat any) (dtype dtypes.DType, flatLen int, err error) {
	flatType := reflect.TypeOf(flat)
	if flatType.Kind() != reflect.Slice {
		return dtype, 0, errors.Errorf("flat data should be a slice, not %s", flatType.Kind())
	}
	dtype = dtypes.FromGoType(flatType.Elem())
	if dtype == dtypes.InvalidDType {
		return dtype, 0, errors.Errorf("flat is a slice of %T, not a valid GoMLX data type", flatType.Elem())
	}
	flatValue := reflect.ValueOf(flat)
	flatLen = flatValue.Len()
	return dtype, flatLen, nil
}

// addUnaryOp adds a generic binary op.
func (b *Builder) addUnaryOp(opType backends.OpType, operandOp backends.Op) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	shape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {

		return nil, err
	}
	return b.newNode(opType, shape, operand), nil
}

// addBinaryOp adds a generic binary op.
func (b *Builder) addBinaryOp(opType backends.OpType, lhsOp, rhsOp backends.Op) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.BinaryOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return b.newNode(opType, shape, lhs, rhs), nil
}

// addComparisonOp adds a generic comparison binary op.
func (b *Builder) addComparisonOp(opType backends.OpType, lhsOp, rhsOp backends.Op) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	shape, err := shapeinference.ComparisonOp(opType, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return b.newNode(opType, shape, lhs, rhs), nil
}
