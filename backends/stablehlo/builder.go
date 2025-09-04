package stablehlo

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/stablehlo"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	compiled bool

	builder *stablehlo.Builder
	fn      *stablehlo.Function

	parameterNames  []string
	parameterShapes []shapes.Shape
}

var _ backends.Builder = (*Builder)(nil)

// Builder creates a new builder used to define a new computation.
func (backend *Backend) Builder(name string) backends.Builder {
	if err := backend.CheckValid(); err != nil {
		klog.Error(err)
		return nil
	}
	b := &Builder{
		backend: backend,
		builder: stablehlo.New(name),
		name:    name,
	}
	b.fn = b.builder.Main()
	return b
}

// Node represents the output of an operation and implements a "backends.Op" interface.
type Node struct {
	value   *stablehlo.Value
	shape   shapes.Shape
	builder *Builder
}

// CheckValid returns an error if the backend or the builder are not ok.
//
// E.g.: they have been finalized or the builder has already been compiled.
func (b *Builder) CheckValid() error {
	if b == nil || b.builder == nil {
		return errors.Errorf("builder is nil or undefined for %q", BackendName)
	}
	return b.backend.CheckValid()
}

// verifyAndCastOp sanity checks that the op is valid and created with this builder.
func (b *Builder) verifyAndCastOp(op backends.Op, opName string) (*Node, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if op == nil {
		return nil, errors.Errorf("nil Op given as an input to %q", opName)
	}
	node, ok := op.(*Node)
	if !ok {
		return nil, errors.Errorf("nil or invalid Op (%T: %v) given as an input to %q, it must be an Op created by the same backend builder (%s:%s)",
			op, op, opName, b.backend.Name(), b.name)
	}
	if node.builder != b {
		return nil, errors.Errorf("op given to parameter %s was created with a different builder (%s) than the builder (%s) it is being used in -- Ops cannot cross to different builders",
			opName, node.builder.Name(), b.Name())
	}
	return node, nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	if err := b.CheckValid(); err != nil {
		return shapes.Invalid(), err
	}
	node, err := b.verifyAndCastOp(op, "OpShape")
	if err != nil {
		return shapes.Invalid(), err
	}
	return node.shape, nil
}

func (b *Builder) newNode(value *stablehlo.Value) *Node {
	return &Node{
		value:   value,
		shape:   ShapeFromStableHLO(value.Shape()),
		builder: b,
	}
}

// Parameter creates an input parameter for the computation.
//
// During the computation's execution this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	normalizedName := stablehlo.NormalizeIdentifier(name)
	if slices.Index(b.parameterNames, normalizedName) != -1 {
		if name == normalizedName {
			return nil, errors.Errorf("parameter named %q already exists", name)
		}
		return nil, errors.Errorf("parameter named %q (normalized to %q) already exists",
			name, normalizedName)
	}
	b.parameterNames = append(b.parameterNames, normalizedName)
	b.parameterShapes = append(b.parameterShapes, shape)
	value := b.fn.NewNamedInput(name, ShapeToStableHLO(shape))
	return b.newNode(value), nil
}

// Identity returns an Op whose output is the same as its input.
// It's a no-op that can serve as a place-holder.
func (b *Builder) Identity(x backends.Op) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	node, err := b.verifyAndCastOp(x, "OpShape")
	if err != nil {
		return nil, err
	}
	return node, nil
}

// Constant creates a constant in the graph with the given flat values and the shape defined by the dimensions.
//
// The flat value must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dimensions ...int) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if flat == nil {
		return nil, errors.Errorf("nil value given to Constant")
	}
	value, err := b.fn.ConstantFromFlatAndDimensions(flat, dimensions...)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Constant()")
	}
	return b.newNode(value), nil
}

// BroadcastInDim broadcasts x to an output with the given shape.
// broadcastAxes has an output axes value for each x axes (len(broadcastAxes) == x.Shape.Rank()).
// The i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
// broadcastAxes must be also increasing: this operation cannot be used to transpose axes, it will only
// broadcast and introduce new axes in-between.
// This also requires that the i-th input axis is either 1 or is the same as the
// output dimension it's broadcasting into.
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcastAxes will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcastAxes
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
func (b *Builder) BroadcastInDim(x backends.Op, outputShape shapes.Shape, broadcastAxes []int) (backends.Op, error) {
	node, err := b.verifyAndCastOp(x, "BroadcastInDim")
	if err != nil {
		return nil, err
	}
	value, err := b.fn.BroadcastInDim(node.value, ShapeToStableHLO(outputShape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// broadcastForBinaryOps returns the broadcasted versions of the two ops,
// converting them to Nodes in the process.
func (b *Builder) broadcastForBinaryOps(opType backends.OpType, lhs, rhs backends.Op) (lhsNode, rhsNode *Node, err error) {
	opName := opType.String()
	lhsNode, err = b.verifyAndCastOp(lhs, opName)
	if err != nil {
		return
	}
	rhsNode, err = b.verifyAndCastOp(rhs, opName)
	if err != nil {
		return
	}
	if lhsNode.shape.DType != rhsNode.shape.DType {
		return nil, nil, errors.Errorf("cannot broadcast %s and %s for %q: they have different dtypes",
			lhsNode.shape.DType, rhsNode.shape.DType, opType)
	}
	if rhsNode.shape.Equal(lhsNode.shape) {
		// No casting needed.
		return
	}

	// If any is a scalar, just broadcast it to the other one.
	if lhsNode.shape.IsScalar() {
		var value *stablehlo.Value
		value, err = b.fn.BroadcastInDim(lhsNode.value, rhsNode.value.Shape(), nil)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while building op %q", opType)
		}
		lhsNode = b.newNode(value)
		return
	} else if rhsNode.shape.IsScalar() {
		var value *stablehlo.Value
		value, err = b.fn.BroadcastInDim(rhsNode.value, lhsNode.value.Shape(), nil)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while building op %s", opName)
		}
		rhsNode = b.newNode(value)
		return
	}

	// Find the larger shape that fits both operands.
	newShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, nil, err
	}
	newShapeStableHLO := ShapeToStableHLO(newShape)
	broadcastAxes := xslices.Iota(0, newShape.Rank())
	if !newShape.Equal(lhsNode.shape) {
		value, err := b.fn.BroadcastInDim(lhsNode.value, newShapeStableHLO, broadcastAxes)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting lhs for op %q", opType)
		}
		lhsNode = b.newNode(value)
	}
	if !newShape.Equal(rhsNode.shape) {
		value, err := b.fn.BroadcastInDim(rhsNode.value, newShapeStableHLO, broadcastAxes)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting rhs for op %q", opType)
		}
		rhsNode = b.newNode(value)
	}
	return
}
