package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"reflect"
	"slices"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	compiled bool

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
func (b *Builder) Compile(outputs ...backends.Op) backends.Executable {
	b.compiled = true
	b.outputs = b.checkOps(outputs)
	return &Executable{builder: b}
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

	// data for the specific node type.
	data any
}

// newNode adds a new node of the given opType and shape to the Builder graph.
// It's used by the other ops when creating new nodes.
func (b *Builder) newNode(opType backends.OpType, shape shapes.Shape, inputs ...*Node) *Node {
	n := &Node{
		opType:     opType,
		builderIdx: len(b.nodes),
		shape:      shape,
		inputs:     slices.Clone(inputs),
	}
	b.nodes = append(b.nodes, n)
	return n
}

// checkOps validates that the ops are from SimpleGo and from this builder.
// It also checks whether the Builder is not yet compiled.
func (b *Builder) checkOps(opType string, ops ...backends.Op) []*Node {
	if b == nil {
		exceptions.Panicf("%s: Builder is nil (!?), cannot build a graph", opType)
	}
	if b.compiled {
		exceptions.Panicf("cannot add new op (%s) to Builder %s, it has already been compiled", opType, b.name)
	}
	nodes := make([]*Node, len(ops))
	var ok bool
	for idx, op := range ops {
		if op == nil {
			exceptions.Panicf("%s: input op #%d is nil!?", opType, idx)
		}
		nodes[idx], ok = op.(*Node)
		if !ok {
			exceptions.Panicf("cannot use input op #%d in backend %q that was created on a different backend for %s", idx, b.backend, opType)
		}
		if nodes[idx].builder != b {
			exceptions.Panicf("%s: input op #%d was created with a different builder (%q), cannot use it with builder %q",
				opType, idx, nodes[idx].builder.name, b.name)
		}
	}
	return nodes
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) shapes.Shape {
	inputs := b.checkOps("OpShape", op)
	return inputs[0].shape
}

// checkFlat throws an exception if flat is not a slice of one of the dtypes supported.
// It returns the supported dtype and the length of the flat slice.
func checkFlat(flat any) (dtypes.DType, int) {
	flatType := reflect.TypeOf(flat)
	if flatType.Kind() != reflect.Slice {
		exceptions.Panicf("flat data should be a slice, not %s", flatType.Kind())
	}
	dtype := dtypes.FromGoType(flatType.Elem())
	if dtype == dtypes.InvalidDType {
		exceptions.Panicf("flat is a slice of %T, not a valid GoMLX data type", flatType.Elem())
	}
	flatValue := reflect.ValueOf(flat)
	return dtype, flatValue.Len()
}

// addUnaryOp adds a generic binary op.
func (b *Builder) addUnaryOp(opType backends.OpType, operandOp backends.Op) *Node {
	inputs := b.checkOps(opType.String(), operandOp)
	operand := inputs[0]
	shape := shapeinference.UnaryOp(opType, operand.shape)
	return b.newNode(opType, shape, operand)
}

// addBinaryOp adds a generic binary op.
func (b *Builder) addBinaryOp(opType backends.OpType, lhsOp, rhsOp backends.Op) *Node {
	inputs := b.checkOps(opType.String(), lhsOp, rhsOp)
	lhs, rhs := inputs[0], inputs[1]
	shape := shapeinference.BinaryOp(opType, lhs.shape, rhs.shape)
	return b.newNode(opType, shape, lhs, rhs)
}
