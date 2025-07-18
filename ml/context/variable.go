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
	"strings"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/gomlx/gopjrt/dtypes"
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
	value       *tensors.Tensor     // Value of the variable.

	// graphToNodes maps graph ids in which this variable was used to its parameter Node and
	// its last value Node.
	graphToNodes xsync.SyncMap[graph.GraphId, *variableNodes]
}

// CloneToContext Variable.
//
// Value, name, scope, trainable state and initializer are cloned.
// But the new clone starts with no graph node mapping -- so it's assumed it's not in use by any Graph.
//
// The variable is then inserted into the given context.
func (v *Variable) CloneToContext(toCtx *Context) *Variable {
	newV := &Variable{
		ctx:       toCtx,
		name:      v.name,
		scope:     v.scope,
		shape:     v.shape,
		Trainable: v.Trainable,
	}
	if v.value != nil {
		newV.value = v.value.Clone()
	}
	toCtx.InAbsPath(v.scope).setVariableInScope(v.name, newV)
	return newV
}

// variableNodes is used to store the variable parameter node (fed to the graph) and current value Node for a given graph.
// They can be different if the variable value is changed during the graph building with [Variable.SetValueGraph].
type variableNodes struct {
	paramNode, valueNode *Node
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
	return fmt.Sprintf("%s/%s", v.Scope(), v.Name())
}

// IsValid returns whether the variable is holding a valid value.
func (v *Variable) IsValid() bool {
	if v == nil {
		return false
	}
	return v.shape.Ok()
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

// Reset sets the variable value to nil while preserving the shape, and marks the context that it needs initialization.
//
// This will force to be variable to be reinitialized the next time a graph using the variable is executed.
func (v *Variable) Reset() {
	v.ctx.data.needsInitialization = true
	if v.value != nil {
		// Don't wait for the GC to free the memory from the accelerator.
		v.value.FinalizeAll()
	}
	v.value = nil
}

// Scope where the variable was created.
func (v *Variable) Scope() string {
	if v == nil {
		return "<nil>"
	}
	return v.scope
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

// VariableParameterPrefix is used to prefix Graph parameter names for variablesMap.
const VariableParameterPrefix = "var:"

// ScopeAndName is a quick pretty-print way to refer to a variable.
func (v *Variable) ScopeAndName() string {
	return JoinScope(v.Scope(), v.Name())
}

// ParameterName used when creating a parameter node in a Graph to access the variable, or as a key when saving.
// It is a unique name for the variable that includes the scope and the variable name, and is reversible.
func (v *Variable) ParameterName() string {
	v.AssertValid()
	return VariableParameterNameFromScopeAndName(v.Scope(), v.Name())
}

// VariableScopeAndNameFromParameterName extracts the scope and name from a variable's [GetParameterName].
// It will return empty strings for an invalid parameter name.
func VariableScopeAndNameFromParameterName(parameterName string) (scope, name string) {
	if !strings.HasPrefix(parameterName, VariableParameterPrefix) {
		return
	}
	parts := strings.Split(parameterName[len(VariableParameterPrefix):], ScopeSeparator)
	if len(parts) == 1 {
		// Scope was not properly set.
		return
	}
	name = parts[len(parts)-1]
	if len(parts) > 2 {
		scope = strings.Join(parts[:len(parts)-1], ScopeSeparator)
	} else {
		scope = RootScope
	}
	return
}

// VariableParameterNameFromScopeAndName creates the [Variable.ParameterName] from its scope and name.
func VariableParameterNameFromScopeAndName(scope, name string) string {
	return fmt.Sprintf("%s%s%s%s", VariableParameterPrefix, scope, ScopeSeparator, name)
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
func (v *Variable) Value() *tensors.Tensor {
	v.AssertValid()
	return v.value
}

// SetValue updates the tensor holding the variable value.
// NOTE: Because often variables are large, the previous value is immediately freed (as opposed to
// wait for garbage collection). If the previous value is used somewhere else, use SetValuePreservingOld.
func (v *Variable) SetValue(value *tensors.Tensor) {
	if v.value != nil {
		v.value.FinalizeAll()
	}
	v.SetValuePreservingOld(value)
}

// SetValuePreservingOld updates the tensor holding the variable value, and dont' free old value. If previous
// value is not used, use SetValue instead that will free it immediately.
func (v *Variable) SetValuePreservingOld(value *tensors.Tensor) {
	v.value = value
	v.shape = value.Shape()
}

// InUseByGraph returns whether the variable is currently in use by the given graph.
func (v *Variable) InUseByGraph(g *Graph) bool {
	v.AssertValid()
	_, found := v.graphToNodes.Load(g.GraphId())
	return found
}

// ChangedInGraph returns whether the variable is in use and was changed in the computation graph g.
func (v *Variable) ChangedInGraph(g *Graph) bool {
	v.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		return false
	}
	return nodes.paramNode != nodes.valueNode
}

// ValueGraph returns the Node of the Graph that holds the current value of the variable. It can be changed
// for the graph (for instance when applying a gradient descent) by [SetValueGraph].
func (v *Variable) ValueGraph(g *Graph) *Node {
	v.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		// Use a newly created parameter node as the initial graph value Node.
		return v.ParamNode(g)
	}
	return nodes.valueNode
}

// SetValueGraph sets the value (a graph [*Node]) of the variable for the current graph.
//
// This is used to "communicate" among different parts of the graph building that this value Node should
// be used as the new variable value.
//
// [context.Exec] will also use the last value set with [SetValueGraph] and include it as the output of the graph
// execution and then update the variables (with [SetValue]) accordingly after each graph execution.
//
// So a graph building function can update variable values, for instance to update weights during gradient
// descent.
func (v *Variable) SetValueGraph(value *Node) {
	v.AssertValid()
	g := value.Graph()
	g.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		// Creates a parameter node, as this includes the variable as in use for the graph.
		_ = v.ParamNode(g)
		nodes, _ = v.graphToNodes.Load(g.GraphId())
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
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		paramName := v.ParameterName()
		paramNode := graph.Parameter(g, paramName, v.shape)
		nodes = &variableNodes{valueNode: paramNode, paramNode: paramNode}
		v.graphToNodes.Store(g.GraphId(), nodes)
	}
	return nodes.paramNode
}

// SetTrainable sets the variable trainable status. Returns itself, so calls can be cascaded.
func (v *Variable) SetTrainable(trainable bool) *Variable {
	v.AssertValid()
	v.Trainable = trainable
	return v
}

// Finalize variable and associate value.
// Variable is left in an unsuable state, only do this if you are sure this variable is no longer in use.
//
// Usually, one calls Context.Finalize, which in turns finalizes all variables.
func (v *Variable) Finalize() {
	if v.value != nil {
		v.value.FinalizeAll()
		v.value = nil
	}
	v.shape = shapes.Invalid()
	v.graphToNodes.Clear()
}
