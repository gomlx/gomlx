package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"reflect"
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
	inputs     *[]Node

	// shape of the output.
	opType  backends.OpType
	shape   shapes.Shape
	builder *Builder

	// data for the specific node type.
	data any
}

// newNode adds a new node of the given opType and shape to the Builder graph.
// It's used by the other ops when creating new nodes.
func (b *Builder) newNode(opType backends.OpType, shape shapes.Shape) *Node {
	n := &Node{
		opType:     backends.OpTypeOpParameter,
		builderIdx: len(b.nodes),
		shape:      shape,
	}
	b.nodes = append(b.nodes, n)
	return n
}

// checkOps validates that the ops are from SimpleGo and from this builder.
// It also checks whether the Builder is not yet compiled.
func (b *Builder) checkOps(ops ...backends.Op) []*Node {
	if b == nil {
		exceptions.Panicf("Builder is nil (!?), cannot build a graph")
	}
	if b.compiled {
		exceptions.Panicf("cannot add new ops to Builder %s, it has already been compiled", b.name)
	}
	nodes := make([]*Node, len(ops))
	var ok bool
	for idx, op := range ops {
		if op == nil {
			exceptions.Panicf("input op #%d is nil!?", idx)
		}
		nodes[idx], ok = op.(*Node)
		if !ok {
			exceptions.Panicf("cannot use input op #%d in backend %q that was created on a different backend", idx, b.backend)
		}
		if nodes[idx].builder != b {
			exceptions.Panicf("op #%d was created with a different builder (%q), cannot use it with builder %q",
				idx, nodes[idx].builder.name, b.name)
		}
	}
	return nodes
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) shapes.Shape {
	inputs := b.checkOps(op)
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

// addBinaryOp adds a generic binary op
func (b *Builder) addBinaryOp(lhs, rhs backends.Op) *Node {

}
