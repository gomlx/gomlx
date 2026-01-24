// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/pkg/errors"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	compiled bool

	// mainFn is the main function of the computation.
	// Each function (including mainFn and closures) has its own nodes slice.
	mainFn *Function
}

// Compile-time check.
var _ backends.Builder = (*Builder)(nil)

// Name implements backends.Builder.
func (b *Builder) Name() string {
	return b.name
}

// Main returns the main function of this computation.
func (b *Builder) Main() backends.Function {
	return b.mainFn
}

// NewFunction creates a new named function within this builder.
// Named functions can be called with Call() and are independent of the main function.
func (b *Builder) NewFunction(name string) (backends.Function, error) {
	if b == nil {
		return nil, errors.Errorf("Builder is nil")
	}
	if b.compiled {
		return nil, errors.Errorf("cannot create new function, builder has already been compiled")
	}
	if name == "" {
		return nil, errors.Errorf("function name cannot be empty")
	}
	f := &Function{
		builder:   b,
		name:      name,
		parent:    nil, // Top-level functions have no parent
		nodeDedup: make(map[nodeDedupKey][]*Node),
	}
	return f, nil
}

// Compile implements backends.Builder.
func (b *Builder) Compile() (backends.Executable, error) {
	if !b.mainFn.returned {
		return nil, errors.Errorf("Main function must have Return() called before Compile()")
	}

	// Handle duplicate outputs by creating Identity nodes for duplicates.
	outputs := b.mainFn.outputs
	seenNodes := sets.Make[*Node]()
	for i, node := range outputs {
		if seenNodes.Has(node) {
			// Create an Identity node for this duplicate output.
			identityOp, err := b.mainFn.Identity(node)
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to create Identity node for duplicate output at index %d", i)
			}
			identityNode, ok := identityOp.(*Node)
			if !ok {
				return nil, errors.Errorf("Identity returned unexpected type for duplicate output at index %d", i)
			}
			outputs[i] = identityNode
		} else {
			seenNodes.Insert(node)
		}
	}
	for _, node := range outputs {
		if len(node.multiOutputsShapes) != 0 {
			return nil, errors.Errorf(
				"%s node %q is internal (with multiple-outputs) and cannot be used for output",
				b.Name(),
				node.opType,
			)
		}
	}

	// Update mainFn outputs (in case duplicates were handled) and compile
	b.mainFn.outputs = outputs
	mainFnExec, err := newFunctionExecutable(b.mainFn)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to compile main function")
	}
	b.mainFn.compiled = mainFnExec

	b.compiled = true
	return newExecutable(b, mainFnExec), nil
}

// Finalize immediately releases the resources associated with the Builder.
func (b *Builder) Finalize() {
	if b.mainFn != nil {
		b.mainFn.nodes = nil
		b.mainFn.nodeDedup = nil
		b.mainFn.parameters = nil
		b.mainFn.outputs = nil
	}
}

// Node in the SimpleGo computation graph.
type Node struct {
	// idx is the index of this node in its function's nodes slice.
	idx    int
	inputs []*Node

	// capturedInputs holds nodes from parent scopes that are used by closures
	// called by this node (for ops like If, While, Sort that use closures).
	// Each inner slice corresponds to one closure's captured values.
	// These are treated as additional inputs for dependency tracking and lifetime management.
	capturedInputs [][]*Node

	// shape of the output.
	opType  backends.OpType
	shape   shapes.Shape
	builder *Builder

	// function is the function in which this node was created.
	// This is used to detect cross-function node usage.
	function *Function

	// multiOutputsShapes are set for a few specialized nodes.
	// For most nodes this is set to nil.
	multiOutputsShapes []shapes.Shape
	multiOutputsNodes  []*Node
	isNodeSelectOutput bool
	selectOutputIdx    int

	// data for the specific node type.
	data any
}


// MultiOutputValues converts a multi-output node's outputs to []backends.Value.
func (node *Node) MultiOutputValues() []backends.Value {
	outputs := make([]backends.Value, len(node.multiOutputsNodes))
	for i, outNode := range node.multiOutputsNodes {
		outputs[i] = outNode
	}
	return outputs
}

// IsMultiOutputs returns whether this node yields multiple outputs.
func (n *Node) IsMultiOutputs() bool {
	return len(n.multiOutputsShapes) > 0
}

// checkValues validates that the values are from SimpleGo and from this builder.
// It also checks whether the Builder is not yet compiled.
func (b *Builder) checkValues(opType string, values ...backends.Value) ([]*Node, error) {
	if b == nil {
		return nil, errors.Errorf("%s: Builder is nil (!?), cannot build a graph", opType)
	}
	if b.compiled {
		return nil, errors.Errorf("cannot add new op (%s) to Builder %q, it has already been compiled", opType, b.name)
	}
	nodes := make([]*Node, len(values))
	var ok bool
	for idx, op := range values {
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
func (b *Builder) OpShape(op backends.Value) (shapes.Shape, error) {
	inputs, err := b.checkValues("OpShape", op)
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
