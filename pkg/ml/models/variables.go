package models

import (
	"fmt"

	"github.com/gomlx/gomlx/graph"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
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
	name, scope string

	// trainable indicates whether the variable is trainable.
	// If set to false, it won't be touched by optimizers of a model.
	trainable bool

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
//
// Notice that valueNode is never nil -- it is initialized with paramNode if the variable is read.
// The paramNode can be nil if the variable was written to only (Variable.SetValueGraph) before any use.
type variableNodes struct {
	paramNode, valueNode *graph.Node
}

// anyValueToTensor converts a value to a tensor, it it's not yet a tensor.
//
// See tensors.FromAnyValue for conversion.
//
// A graph *Node does not work here, this is assumed to be a concrete tensor value.
// See VariableWithValueGraph instead, to create a variable with a graph *Node.
func anyValueToTensor(value any) *tensors.Tensor {
	if tensorValue, ok := value.(*tensors.Tensor); ok {
		return tensorValue
	}
	if node, ok := value.(*graph.Node); ok {
		Panicf(
			"trying to feed a computation graph node (`*computation.Node`) as a concrete value will not work, "+
				"you have to provide a Go value or a tensor here -- *Node provided: %s", node)
	}
	return tensors.FromAnyValue(value)
}

// VariableWithValue creates or variable initialized with the given value.
//
// The value given must be concrete: either a tensor or a normal Go value that can be converted to a tensor. See tensors.FromAnyValue.
//
// If used in a graph building function, the variable value will be the initialization value of the
// variable; future executions of the graph will not set again the value -- e.g.: a value created in this way can be
// trained, the initialization only happens once.
//
// If used outside a graph building function, the variable value will be initialized with the given value.
//
// By default, variables are marked as trainable.
//
// See also VariableWithShape.
func VariableWithValue(name string, value any) (*Variable, error) {
	var valueT *tensors.Tensor
	err := TryCatch[error](func() { valueT = anyValueToTensor(value) })
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to parse value %v for variable %q", value, name)
	}

	// New variable: check, create and register it in Context and return.
	return &Variable{
		name:      name,
		shape:     valueT.Shape(),
		value:     valueT,
		trainable: true, // By default variables are trainable.
	}, nil
}

// VariableWithShape creates a variable with the given shape in the current scope.
// It is initialized with the current variable initializer set for the context.
// By default, variables are marked as trainable.
//
// If a Loader is configured (see SetLoader), and the value is available to load, it will override
// the value given here -- e.g.: the value could be actually loaded from the last checkpoint.
//
// Notice that variables information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// If Context is set with Context.Checked(true), this may panic if:
//
// - Context.Unique() and variable already exists (or was loaded);
// - Context.Reuse() and variable didn't exist (or was not loaded);
func VariableWithShape(name string, shape shapes.Shape, initializer VariableInitializer) *Variable {
	//TODO: implement initializers.
	return nil
}

// Name of the variable within the scope.
func (v *Variable) Name() string {
	if v == nil {
		return "<nil>"
	}
	return v.name
}

// Scope is an optional prefix for the variable name that gives context to where it is being used.
//
// For example, if you have a FNN model, you may have one "weights" variable per layer. The scope will
// hold the layer name.
//
// Scopes are not obligatory, they are left empty during the variable creation.
// But they can be set with SetScope or automatically set for all variables in a struct with SetVariablesScope.
func (v *Variable) Scope() string {
	return v.scope
}

// SetScope of a variable.
//
// See also SetVariablesScope.
func (v *Variable) SetScope(scope string) {
	v.scope = scope
}

// String returns the name, and if defined, prefixed with the scope.
func (v *Variable) String() string {
	if v == nil || !v.Shape().Ok() {
		return "INVALID (NIL) VARIABLE"
	}
	if v.scope == "" {
		return v.name
	}
	return fmt.Sprintf("%s/%s", v.scope, v.name)
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

// SetTrainable sets the variable trainable status. Returns itself, so calls can be cascaded.
func (v *Variable) SetTrainable(trainable bool) *Variable {
	v.AssertValid()
	v.trainable = trainable
	return v
}

// IsTrainable returns whether the variable should be updated during training.
func (v *Variable) IsTrainable() bool {
	v.AssertValid()
	return v.trainable
}

// Value returns the tensor holding the variable value. Use this to manipulate the value in Go.
// If building a computation graph, use ValueGraph instead.
//
// WARNING: memory management here is tricky: a call to SetValue will
// trigger the current value to be deallocated, and what was returned
// by a previous call to Value to become invalid. The recommendation
// is not to use this while training a model that uses the variable.
func (v *Variable) Value() *tensors.Tensor {
	v.AssertValid()
	return v.value
}

// SetValue updates the tensor holding the variable value.
//
// This does not allow changes in shape -- you will need to create a new variable for that.
//
// NOTE: Because often variables are large, the previous value is immediately freed (as opposed to
// waiting for a garbage collection). If the previous value is used somewhere else, use SetValuePreservingPrevious.
func (v *Variable) SetValue(value *tensors.Tensor) {
	if !value.Shape().Equal(v.shape) {
		Panicf("variable %q cannot have its value (%s) changed to a new shape (%s)", v, v.shape, value.Shape())
	}
	if v.value != nil {
		v.value.FinalizeAll()
	}
	v.SetValuePreservingPrevious(value)
}

// SetValuePreservingPrevious updates the tensor holding the variable's value while not freeing the previous value.
//
// If the previous value is not used, use SetValue instead that will free it immediately.
func (v *Variable) SetValuePreservingPrevious(value *tensors.Tensor) {
	v.value = value
	v.shape = value.Shape()
}

// InUseByGraph returns whether the variable is currently in use by the given graph.
func (v *Variable) InUseByGraph(g *graph.Graph) bool {
	v.AssertValid()
	_, found := v.graphToNodes.Load(g.GraphId())
	return found
}

// ChangedInGraph returns whether the variable is in use and was changed in the computation graph g.
func (v *Variable) ChangedInGraph(g *graph.Graph) bool {
	v.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		return false
	}
	return nodes.paramNode != nodes.valueNode
}

// ValueGraph returns the Node of the Graph that holds the current value of the variable.
//
// It can be changed for the graph (for instance, when applying a gradient descent) by SetValueGraph.
func (v *Variable) ValueGraph(g *graph.Graph) *graph.Node {
	v.AssertValid()
	gID := g.GraphId()
	nodes, found := v.graphToNodes.Load(gID)
	if found {
		return nodes.valueNode
	}

	// Find Exec this graph is part of, to create a side input parameter that will hold the variable value.
	e := getExecByGraphId(gID)
	if e == nil {
		Panicf("cannot access variable %q in a graph not managed by a models.Exec -- models.Variable can only be used "+
			"for computation graphs created using the models.Exec executor", v.name)
		panic(nil)
	}
	return e.createVariableParamNode(v, g)
}

// createVariableParamNode returns the given Graph g's Node that corresponds to the parameter that will be fed with
// the current variable value when the graph is executed. It's the initial value of the variable
// in the computation Graph.
//
// If the parameter node hasn't been created for the Graph g yet, one is created.
//
// Since the value of a variable can change in the middle of the graph (e.g: something that uses the
// variable after a gradient descent is applied) consider using ValueGraph to read the current associated
// value of a variable in a graph.
func (e *Exec) createVariableParamNode(v *Variable, g *graph.Graph) *graph.Node {
	v.AssertValid()
	g.AssertValid()
	gID := g.GraphId()
	nodes, found := v.graphToNodes.Load(gID)
	if found {
		return nodes.paramNode
	}

	// Store variable as side input to this graph.
	e.mu.Lock()
	variableIndexInGraph := len(e.sideInputs[gID])
	e.sideInputs[gID] = append(e.sideInputs[gID], v)
	e.mu.Unlock()

	// Creates a new graph parameter.
	paramName := fmt.Sprintf("v%05d_%s", variableIndexInGraph, v.String())
	paramNode := graph.Parameter(g, paramName, v.shape)
	nodes = &variableNodes{valueNode: paramNode, paramNode: paramNode}
	v.graphToNodes.Store(gID, nodes)
	return nodes.paramNode
}

// SetValueGraph sets the value (a *graph.Node) of the variable for the current graph.
//
// This is used to "communicate" among different parts of the graph building that this value node should
// be used as the new variable value.
//
// The Exec will also use the last value set with SetValueGraph and include it as a side output of the graph
// execution and then update the variable concrete value (with SetValue) accordingly after each graph execution.
//
// So a graph building function can update variable values, for instance, to update weights during gradient
// descent.
func (v *Variable) SetValueGraph(value *graph.Node) {
	v.AssertValid()
	g := value.Graph()
	g.AssertValid()
	gID := g.GraphId()

	if !value.Shape().Equal(v.shape) {
		Panicf("variable %q cannot have its value (%s) changed to a new shape (%s)", v, v.shape, value.Shape())
	}
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found || nodes == nil {
		nodes = &variableNodes{}
		v.graphToNodes.Store(gID, nodes)
	} else if nodes.paramNode != nodes.valueNode && nodes.valueNode != nil {
		// The variable has already been set for this graph, we just need to update its value.
		nodes.valueNode = value
		v.graphToNodes.Store(gID, nodes)
		return
	}

	// Find Exec this graph is part of, to create a side input parameter that will hold the variable value.
	e := getExecByGraphId(gID)
	if e == nil {
		Panicf("cannot access variable %q in a graph not managed by a models.Exec -- models.Variable can only be used "+
			"for computation graphs created using the models.Exec executor", v.name)
		panic(nil)
	}

	// Store variable as side output to this graph.
	e.mu.Lock()
	e.sideOutputs[gID] = append(e.sideOutputs[gID], v)
	e.mu.Unlock()

	// Notice that nodes.paramNode may be nil, in case the value of the variable was never used as input.
	nodes.valueNode = value
}

// Finalize the variable immediately freeing associated value.
//
// The variable is left in an unusable state, only do this if you are sure this variable is no longer in use.
func (v *Variable) Finalize() {
	if v.value != nil {
		v.value.FinalizeAll()
		v.value = nil
	}
	v.shape = shapes.Invalid()
	v.graphToNodes.Clear()
}
