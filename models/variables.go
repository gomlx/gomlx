package models

import (
	"github.com/gomlx/gomlx/graph"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/gomlx/gopjrt/dtypes"
)

// Variable (or weights) of a model, typically learned during training, but can also be used as large constants.
//
// They have different "views":
//
//   - Outside a computation graph, they have a shape, and if already initialized, a concrete *tensor.Tensor value.
//   - Within a graph building function, they have a graph node that represents them. One per computation graph,
//     that's why the Variable.ValueGraph() method takes a graph parameter as input.
//
// You can create them during the graph building function, or outside of it.
//
// Always use it by reference (pointer), never by value, to keep all the various views in sync.
type Variable struct {
	name string

	// Trainable indicates whether the variable is trainable.
	// If set to false, it won't be touched by optimizers of a model.
	Trainable bool

	shape       shapes.Shape
	initializer VariableInitializer // Used if the variable is not yet initialized.
	value       *tensors.Tensor     // Value of the variable.

	// graphToNodes maps graph ids in which this variable was used to its parameter Node and
	// its last value Node.
	graphToNodes xsync.SyncMap[graph.GraphId, *variableNodes]
}

// VariableInitializer builds a valueNode that returns a value to initialize a variable of the given
// shape. It is defined in the Context.
type VariableInitializer = func(g *graph.Graph, shape shapes.Shape) *graph.Node

// variableNodes is used to store the variable parameter node (fed to the graph) and current value Node for a given graph.
// They can be different if the variable value is changed during the graph building with Variable.SetValueGraph.
type variableNodes struct {
	paramNode, valueNode *graph.Node
}

// Name of the variable within the scope.
func (v *Variable) Name() string {
	if v == nil {
		return "<nil>"
	}
	return v.name
}

// String implements stringer.
func (v *Variable) String() string {
	if v == nil || !v.Shape().Ok() {
		return "INVALID (NIL) VARIABLE"
	}
	return v.Name()
}

// IsValid returns whether the variable is holding a valid value.
func (v *Variable) IsValid() bool {
	if v == nil {
		return false
	}
	return v.shape.Ok()
}

// AssertValid panics if the variable is in an invalid state: if it's nil, or if it's shape is not yet set.
func (v *Variable) AssertValid() {
	if v == nil {
		Panicf("context.Variable is nil")
	}
	if !v.Shape().Ok() {
		Panicf("context.Variable has no shape")
	}
}

// Shape returns the variable shape.
func (v *Variable) Shape() shapes.Shape {
	if v == nil {
		return shapes.Shape{}
	}
	return v.shape
}

// DType returns the variable DType.
func (v *Variable) DType() dtypes.DType {
	if v == nil {
		return dtypes.InvalidDType
	}
	return v.shape.DType
}
