//go:build darwin && cgo

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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

	// nodeMap maps GoMLX Value to MIL Value for tracking
	nodeMap map[backends.Value]*model.Value

	// mainFn is the main function, created lazily.
	mainFn *Function

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

// Main returns the main function of this builder.
func (b *Builder) Main() backends.Function {
	if b.mainFn == nil {
		b.mainFn = &Function{
			builder: b,
			name:    backends.MainName,
		}
	}
	return b.mainFn
}

// NewFunction creates a new named function within this builder.
func (b *Builder) NewFunction(name string) (backends.Function, error) {
	if name == backends.MainName {
		return nil, errors.Errorf("cannot create function with reserved name %q", backends.MainName)
	}
	fn := &Function{
		builder: b,
		name:    name,
	}
	return fn, nil
}

// Compile implements backends.Builder.
func (b *Builder) Compile() (backends.Executable, error) {
	// Ensure main function has been defined with outputs
	if b.mainFn == nil {
		return nil, errors.Errorf("no computation defined - call Main() and add operations before Compile()")
	}
	if !b.mainFn.returned {
		return nil, errors.Errorf("Main function has not called Return() - no outputs defined")
	}

	b.outputs = b.mainFn.outputs

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

// OpShape returns the shape of a computation Value.
func (b *Builder) OpShape(op backends.Value) (shapes.Shape, error) {
	nodes, err := b.checkOps("OpShape", op)
	if err != nil {
		return shapes.Invalid(), err
	}
	return nodes[0].shape, nil
}

// checkOps validates that the values are from CoreML and from this builder.
// It also checks whether the Builder is not yet compiled.
func (b *Builder) checkOps(opType string, ops ...backends.Value) ([]*Node, error) {
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
