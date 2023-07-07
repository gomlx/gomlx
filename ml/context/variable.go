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

package context

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context/initializers"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
)

// Variable is a value shared among computation graphs, or across multiple executions of the same graph.
// It's commonly used to store the weights (aka. parameters) of an ML model. It's defined in a scope in
// a Context.
//
// The materialized value can be accessed in between graph executions by Value and SetValue methods.
//
// During the computation graph building, for a particular graph, one can access the graph value (Node)
// of a variable with the methods
//
// They are only initialized when Context.InitializeVariables. That is, they are created and used in
// graph building possibly before they are actually initialized -- when building a graph they are
// passed as parameters (the corresponding graph node is called ParameterNode), and have their values
// passed only during execution.
type Variable struct {
	ctx         *Context
	name, scope string

	// Trainable indicates whether variable is trainable. If set to false it won't be
	// touched by trainers of the model.
	Trainable bool

	shape       shapes.Shape
	initializer VariableInitializer // Set if variable is not yet initialized.
	value       tensor.Tensor       // Value of the variable.

	// graphToNodes maps graph ids in which this variable was used to its parameter Node and
	// its last value Node.
	graphToNodes map[graph.GraphId]*variableNodes
}

// VariableInitializer builds a valueNode that returns a value to initialize a variable of the given
// shape. It is defined in the Context.
type VariableInitializer = initializers.VariableInitializer

// variableNodes is used to store the variable parameter and current value Node for a given graph.
type variableNodes struct {
	paramNode, valueNode *Node
}

// Name of the variable within the scope.
func (v *Variable) Name() string {
	v.AssertValid()
	return v.name
}

// String implements stringer.
func (v *Variable) String() string {
	if v == nil || !v.Shape().Ok() {
		return "INVALID (NIL) VARIABLE"
	}
	return fmt.Sprintf("%s/%s", v.Scope(), v.Name())
}

// AssertValid panics if the variable is in an invalid state: if it's nil or it's shape is not yet set.
func (v *Variable) AssertValid() {
	if v == nil {
		Panicf("context.Variable is nil")
	}
	if !v.Shape().Ok() {
		Panicf("context.Variable has no shape")
	}
}

// Scope where the variable was created.
func (v *Variable) Scope() string {
	v.AssertValid()
	return v.scope
}

// Shape returns the variable shape.
func (v *Variable) Shape() shapes.Shape {
	if v == nil {
		return shapes.Shape{}
	}
	return v.shape
}

// ParameterPrefix is used to prefix Graph parameter names for variablesMap.
const ParameterPrefix = "var:"

// ParameterName used when creating a parameter node in a Graph to access
// the variable.
func (v *Variable) ParameterName() string {
	v.AssertValid()
	return fmt.Sprintf("%s%s%s%s", ParameterPrefix, v.Scope(), ScopeSeparator, v.Name())
}

// Value returns the tensor holding the variable value. Use this to
// manipulate the value in Go. If building a computation Graph use
// Node().
//
// WARNING: memory management here is tricky: a call to SetValue will
// trigger the current value to be deallocated, and what is returned
// by a previous call to Value to become invalid. The recommendation
// is not to use this is a concurrent set up -- or to create proper
// locking mechanisms.
func (v *Variable) Value() tensor.Tensor {
	v.AssertValid()
	return v.value
}

// SetValue updates the tensor holding the variable value.
// NOTE: Because often variables are large in size, the previous value is immediately freed (as opposed to
// wait for garbage collection). If the previous value is used somewhere else, use SetValuePreservingOld.
func (v *Variable) SetValue(value tensor.Tensor) {
	v.value.FinalizeAll()
	v.value = value
}

// SetValuePreservingOld updates the tensor holding the variable value, and dont' free old value. If previous
// value is not used, use SetValue instead that will free it immediately.
func (v *Variable) SetValuePreservingOld(value tensor.Tensor) {
	v.value = value
}

// InUseByGraph returns whether the variable is currently in use by the given graph.
func (v *Variable) InUseByGraph(g *Graph) bool {
	v.AssertValid()
	_, found := v.graphToNodes[g.GraphId()]
	return found
}

// ChangedInGraph returns whether the variable is in use and was changed in the computation graph g.
func (v *Variable) ChangedInGraph(g *Graph) bool {
	v.AssertValid()
	nodes, found := v.graphToNodes[g.GraphId()]
	if !found {
		return false
	}
	return nodes.paramNode != nodes.valueNode
}

// ValueGraph returns the Node of the Graph that holds the current value of the variable. It can be changed
// for the graph (for instance when applying a gradient descent) by SetGraph.
func (v *Variable) ValueGraph(g *Graph) *Node {
	v.AssertValid()
	nodes, found := v.graphToNodes[g.GraphId()]
	if !found {
		// Use a newly created parameter node as the initial graph value Node.
		return v.ParamNode(g)
	}
	return nodes.valueNode
}

// SetValueGraph sets the Node associated with the current value of the variable for the computation
// graph where value is defined.
//
// This is used to "communicate" among different parts of the graph building that this value Node should
// be used as the variable value.
//
// train.Trainer will use the last value set here during graph building and use it as output of the graph
// execution and then update the variables (with SetValue) accordingly after each graph execution, for
// example, after each Trainer.TrainStep call.
func (v *Variable) SetValueGraph(value *Node) {
	v.AssertValid()
	g := value.Graph()
	g.AssertValid()
	nodes, found := v.graphToNodes[g.GraphId()]
	if !found {
		// Creates a parameter node, as this includes the variable as in use for the graph.
		_ = v.ParamNode(g)
		nodes = v.graphToNodes[g.GraphId()]
	}
	nodes.valueNode = value
}

// ParamNode returns the given Graph g's Node that corresponds to the parameter that will be fed with
// the current variable value when the graph is executed. It's the initial value of the variable
// in the computation Graph.
//
// If parameter Node hasn't been created for the Graph g yet, one is created.
//
// Since the value of a variable can change in the middle of the graph (e.g: something that uses the
// variable after a gradient descent is applied) consider using ValueGraph to read the current associated
// value of a variable in a graph.
func (v *Variable) ParamNode(g *Graph) *Node {
	v.AssertValid()
	g.AssertValid()
	nodes, found := v.graphToNodes[g.GraphId()]
	if !found {
		paramName := v.ParameterName()
		paramNode := g.Parameter(paramName, v.shape)
		nodes = &variableNodes{valueNode: paramNode, paramNode: paramNode}
		v.graphToNodes[g.GraphId()] = nodes
	}
	return nodes.paramNode
}

// SetTrainable sets the variable trainable status. Returns itself, so calls can be cascated.
func (v *Variable) SetTrainable(trainable bool) *Variable {
	v.AssertValid()
	v.Trainable = trainable
	return v
}
