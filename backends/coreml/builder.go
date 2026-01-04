//go:build darwin

package coreml

import (
	"fmt"
	"reflect"

	"github.com/gomlx/go-coreml/model"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Builder keeps track of the CoreML computation graph being defined.
type Builder struct {
	notimplemented.Builder

	backend  *Backend
	name     string
	compiled bool

	// milBuilder is the go-coreml model builder
	milBuilder *model.Builder

	// nextConstID is used to generate unique names for constants
	nextConstID int

	// nodes are only created when their inputs have already been created.
	// This is a natural DAG (Directed Acyclic Graph) ordering of the graph.
	nodes []*Node

	// inputs will have parameter nodes
	inputs []*Node

	// outputs can be any type of node
	outputs []*Node

	// nodeMap maps GoMLX Op to MIL Value for tracking
	nodeMap map[backends.Op]*model.Value

	// Input/output metadata for execution
	inputNames   []string
	outputNames  []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape
}

// Compile-time check that Builder implements backends.Builder
var _ backends.Builder = (*Builder)(nil)

// Node represents a node in the CoreML computation graph.
type Node struct {
	builder    *Builder
	builderIdx int
	opType     backends.OpType
	shape      shapes.Shape
	milValue   *model.Value
	inputs     []*Node
}

// Name implements backends.Builder.
func (b *Builder) Name() string {
	return b.name
}

// Compile implements backends.Builder.
func (b *Builder) Compile(outputs []backends.Op, shardings []*backends.ShardingSpec) (backends.Executable, error) {
	if len(shardings) != 0 {
		return nil, errors.Errorf("sharding or distributed execution are not supported by CoreML backend")
	}

	var err error
	b.outputs, err = b.checkOps("Compile", outputs...)
	if err != nil {
		return nil, err
	}

	// Track output metadata and mark outputs in the MIL builder
	for i, node := range b.outputs {
		outputName := fmt.Sprintf("output_%d", i)
		b.milBuilder.Output(outputName, node.milValue)
		b.outputNames = append(b.outputNames, outputName)
		b.outputShapes = append(b.outputShapes, node.shape)
	}

	// Compile the MIL program using the runtime
	runtimeExec, err := b.backend.runtime.Compile(b.milBuilder)
	if err != nil {
		return nil, errors.Wrap(err, "failed to compile CoreML model")
	}

	b.compiled = true
	return newExecutable(b, runtimeExec), nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	nodes, err := b.checkOps("OpShape", op)
	if err != nil {
		return shapes.Invalid(), err
	}
	return nodes[0].shape, nil
}

// Parameter creates an input parameter for the computation.
func (b *Builder) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Op, error) {
	if b.compiled {
		return nil, errors.Errorf("cannot add parameter to compiled builder")
	}
	if sharding != nil {
		return nil, errors.Errorf("sharding not supported by CoreML backend")
	}

	// Convert GoMLX dtype to CoreML dtype
	milDType, err := gomlxDTypeToMIL(shape.DType)
	if err != nil {
		return nil, errors.Wrapf(err, "Parameter %q", name)
	}

	// Convert shape dimensions to int64
	dims := make([]int64, shape.Rank())
	for i := 0; i < shape.Rank(); i++ {
		dims[i] = int64(shape.Dimensions[i])
	}

	// Create input in MIL builder
	milValue := b.milBuilder.Input(name, milDType, dims...)

	// Create node
	node := b.newNode(backends.OpTypeParameter, shape, milValue)
	b.inputs = append(b.inputs, node)
	b.nodeMap[node] = milValue

	// Track input metadata
	b.inputNames = append(b.inputNames, name)
	b.inputShapes = append(b.inputShapes, shape)

	return node, nil
}

// Constant creates a constant in the graph.
func (b *Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	if b.compiled {
		return nil, errors.Errorf("cannot add constant to compiled builder")
	}

	// Validate and get dtype
	dtype, flatLen, err := checkFlat(flat)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	// Validate dimensions
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		return nil, errors.Errorf(
			"Constant: shape %s has size %d, but flat data has length %d",
			shape,
			shape.Size(),
			flatLen,
		)
	}

	// Convert to MIL dtype
	milDType, err := gomlxDTypeToMIL(dtype)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	// Convert dimensions to int64
	milShape := make([]int64, len(dims))
	for i, d := range dims {
		milShape[i] = int64(d)
	}

	// Generate unique name for constant
	constName := fmt.Sprintf("const_%d", b.nextConstID)
	b.nextConstID++

	// Create constant in MIL builder
	milValue := b.milBuilder.Const(constName, milDType, milShape, flat)

	// Create node
	node := b.newNode(backends.OpTypeConstant, shape, milValue)
	b.nodeMap[node] = milValue

	return node, nil
}

// checkOps validates that the ops are from CoreML and from this builder.
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

// newNode adds a new node of the given opType and shape to the Builder graph.
func (b *Builder) newNode(opType backends.OpType, shape shapes.Shape, milValue *model.Value, inputs ...*Node) *Node {
	n := &Node{
		builder:    b,
		opType:     opType,
		builderIdx: len(b.nodes),
		shape:      shape,
		milValue:   milValue,
		inputs:     inputs,
	}
	b.nodes = append(b.nodes, n)
	return n
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

// gomlxDTypeToMIL converts a GoMLX DType to a CoreML MIL DType.
func gomlxDTypeToMIL(dtype dtypes.DType) (model.DType, error) {
	switch dtype {
	case dtypes.Float16:
		return model.Float16, nil
	case dtypes.Float32:
		return model.Float32, nil
	case dtypes.Float64:
		return model.Float64, nil
	case dtypes.Int8:
		return model.Int8, nil
	case dtypes.Int16:
		return model.Int16, nil
	case dtypes.Int32:
		return model.Int32, nil
	case dtypes.Int64:
		return model.Int64, nil
	case dtypes.Bool:
		return model.Bool, nil
	default:
		return 0, errors.Errorf("unsupported dtype %s for CoreML backend", dtype)
	}
}
