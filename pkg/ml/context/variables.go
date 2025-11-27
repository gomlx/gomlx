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

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// Variable is a value shared among computation graphs, or across multiple executions of the same graph.
// It's commonly used to store the weights (aka. parameters) of an ML model. It's defined in a scope in
// a Context.
//
// The materialized value can be accessed in between graph executions by Value and SetValue methods.
//
// During the computation graph building, for a particular graph, one can access the graph value (Node)
// of a variable with the methods.
//
// They are only initialized when Context.InitializeVariables.
// That is, they are created and used in graph building possibly before they are actually initialized.
// When building a graph they are passed as parameters (the corresponding graph node is called ParameterNode),
// and have their values passed only during the execution of the compiled computation graph.
type Variable struct {
	ctx         *Context
	name, scope string

	// Trainable indicates whether the variable is trainable.
	// If set to false, it won't be touched by trainers.
	Trainable bool

	// shape holds the "logical shape" of the variable.
	// On a distributed strategy, this may be different from the individual shards shape.
	shape shapes.Shape

	initializer VariableInitializer // Set if the variable is not yet initialized.

	// It works in two modes:
	//
	// 1. Single-device: using Variable.SetValue and Variable.Value.
	// 2. Multi-device: using Variable.SetDistributedValue and Variable.DistributedValue.

	// value is a local tensor.
	value *tensors.Tensor

	// distValue holds the distributed view: A tensor sharded across a mesh.
	distValue *distributed.Tensor

	// sharding defines how the variable is split in a distributed context.
	// If nil, the variable is considered replicated (or local).
	sharding *distributed.ShardingSpec

	// graphToNodes maps graph ids in which this variable was used to its parameter Node and
	// its last value Node.
	graphToNodes xsync.SyncMap[graph.GraphId, *variableNodes]
}

// CloneToContext Variable.
//
// Value, name, scope, trainable state, sharding spec, and initializer are cloned.
// But the new clone starts with no graph node mapping -- so it's assumed it's not in use by any Graph.
//
// The variable is then inserted into the given context.
func (v *Variable) CloneToContext(toCtx *Context) (*Variable, error) {
	newV := &Variable{
		ctx:       toCtx,
		name:      v.name,
		scope:     v.scope,
		shape:     v.shape,
		Trainable: v.Trainable,
		sharding:  v.sharding,
	}

	// Copy the value if it exists.
	// If both v.value and v.distValue is set, we only clone v.distValue -- one can be created from the other if
	// needed later.
	var err error
	if v.distValue != nil {
		newV.distValue, err = v.distValue.Clone()
	} else if v.value != nil {
		newV.value, err = v.value.Clone()
	}
	if err != nil {
		return nil, err
	}
	toCtx.InAbsPath(v.scope).setVariableInScope(v.name, newV)
	return newV, nil
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

// CheckValid returns an error if the variable is in an invalid state: if it's nil or it's shape is not yet set.
func (v *Variable) CheckValid() error {
	if v == nil {
		return errors.New("context.Variable is nil")
	}
	if !v.Shape().Ok() {
		return errors.New("context.Variable has no shape")
	}
	return nil
}

// AssertValid panics if the variable is in an invalid state: if it's nil or it's shape is not yet set.
func (v *Variable) AssertValid() {
	err := v.CheckValid()
	if err != nil {
		panic(err)
	}
}

// Reset sets the variable value to nil while preserving the shape.
//
// If you intend to reuse the variable, remember to mark the context as needing initialization.
//
// This will force to be variable to be reinitialized the next time a graph using the variable is executed.
func (v *Variable) Reset() error {
	if v.value != nil {
		// Don't wait for the GC to free the memory from the accelerator.
		err := v.value.FinalizeAll()
		if err != nil {
			return err
		}
		v.value = nil
	}

	if v.distValue != nil {
		err := v.distValue.FinalizeAll()
		if err != nil {
			return err
		}
		v.distValue = nil
	}
	return nil
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

// ShardShape returns the sharded shape of the variable.
// It is the same as Shape() in single-device mode, but may differ in distributed mode.
func (v *Variable) ShardShape() shapes.Shape {
	if v.distValue != nil {
		return v.distValue.ShardShape()
	}
	// TODO: fix ShardShape in case distValue is not set yet (nil), but v.sharding is configured.
	return v.Shape()
}

// DType returns the variable DType.
func (v *Variable) DType() dtypes.DType {
	if v == nil {
		return dtypes.InvalidDType
	}
	return v.shape.DType
}

// WithSharding configures the sharding specification for this variable, used for distributed execution.
// Don't use this is running on a single device.
//
// Once a variable is set with SetDistributedValue the sharding specification cannot be changed.
// Setting a variable with SetValue, if the sharding is set, will automatically shard the value, and
// the sharding can no longer be changed.
//
// WithSharding returns the variable itself to allow chaining.
//
// On error, it panics. It's assumed to be used within the model (graph) function, but it can also be
// used during setup -- pre-model building. See SetSharding for a version that returns an error.
func (v *Variable) WithSharding(spec *distributed.ShardingSpec) *Variable {
	err := v.SetSharding(spec)
	if err != nil {
		panic(err)
	}
	return v
}

// SetSharding configures the sharding specification for this variable, used for distributed execution.
// Don't use this is running on a single device.
//
// Once a variable is set with SetDistributedValue the sharding specification cannot be changed.
// Setting a variable with SetValue, if the sharding is set, will automatically shard the value, and
// the sharding can no longer be changed.
//
// See also WithSharding for a version that panics on error -- more commonly used inside a model building (graph)
// function.
func (v *Variable) SetSharding(spec *distributed.ShardingSpec) error {
	if v.distValue != nil {
		if v.distValue.ShardingSpec() != nil {
			return errors.Errorf("variable %q cannot change sharding to %s, it already has a sharding spec set to %q",
				v.ScopeAndName(), spec, v.distValue.ShardingSpec())
		}
		v.sharding = spec
		// This is not changing anything.
		return nil
	}
	v.sharding = spec
	return nil
}

// Sharding returns the current sharding specification.
func (v *Variable) Sharding() *distributed.ShardingSpec {
	return v.sharding
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

// MustValue returns the tensor holding the variable value. Use this to manipulate the value in Go.
// If building a computation graph, use Variable.ValueGraph().
//
// This version panics on error. See details in Variable.Value().
func (v *Variable) MustValue() *tensors.Tensor {
	value, err := v.Value()
	if err != nil {
		panic(err)
	}
	return value
}

// Value returns the tensor holding the variable value. Use this to manipulate the value in Go.
// If building a computation graph, use Variable.ValueGraph().
//
// On a distributed setup, the Variable will be sharded (see Variable.WithSharding() or Variable.SetSharding).
// In these cases you can get the sharded values with Variable.DistributedValue(), or set it with
// Variable.SetDistributedValue().
// If you use Variable.Value() on a distributed value, it will merge the shards and return a tensor with the full value:
// this is useful, for instance, to save the variable (or print it).
//
// WARNING: memory management here is tricky: a call to SetValue triggers the current value to be deallocated,
// and what is returned by a previous call to Value to become invalid.
// The recommendation is not to use this in a concurrent setup -- or to create proper locking mechanisms.
func (v *Variable) Value() (*tensors.Tensor, error) {
	if err := v.CheckValid(); err != nil {
		return nil, err
	}

	if v.value != nil {
		// Value is already there.
		return v.value, nil
	}
	if v.distValue == nil {
		return nil, errors.Errorf("variable %q has no value", v.ScopeAndName())
	}
	var err error
	v.value, err = v.distValue.Merge()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to merge distributed variable %q", v.ScopeAndName())
	}
	return v.value, nil
}

// SetValue updates the tensor holding the variable value, and marks it as no longer needing initialization.
//
// On a distributed setup, the Variable will be sharded (see Variable.WithSharding() or Variable.SetSharding).
// In these cases you can get the sharded values with Variable.DistributedValue(), or set it with
// Variable.SetDistributedValue().
// If you use SetValue() on a distributed variable, the current distributed value (if any) is immediately freed.
// And later, upon request of the distributed value, the value will be sharded.
// This is useful, for instance, when loading the variable: it is loaded and set with SetValue(), but used as
// a distributed.Tensor later. The sharding happens automatically.
//
// NOTE: Because often variables are large, the previous value is immediately freed (as opposed to
// waiting for garbage collection). If the previous value is used somewhere else, use SetValuePreservingOld.
//
// This also invalidates any distributed or on-device copies of the variable.
//
// The shape of the variable is set to the new value set. If value is nil, the shape is left unchanged.
func (v *Variable) SetValue(value *tensors.Tensor) error {
	err := v.Reset()
	if err != nil {
		return err
	}
	return v.SetValuePreservingOld(value)
}

// MustSetValue updates the tensor holding the variable's value.
// See detail in Variable.SetValue().
//
// This version panics on error.
//
// Deprecated: use Variable.SetValue() instead.
func (v *Variable) MustSetValue(value *tensors.Tensor) {
	err := v.SetValue(value)
	if err != nil {
		panic(err)
	}
}

// SetValuePreservingOld updates the tensor holding the variable value and doesn't free the old value.
// If the previous value is not used, use Variable.SetValue() instead that will free it immediately.
//
// This affects also the distributed value: it is reset upon SetValuePreservingOld, but not immediately
// finalized.
//
// If the variable is set to nil, the context is automatically marked as needing initialization, just in case.
//
// The shape of the variable is set to the new value set. If value is nil, the shape is left unchanged.
func (v *Variable) SetValuePreservingOld(value *tensors.Tensor) error {
	if err := v.CheckValid(); err != nil {
		return err
	}
	v.value = value
	v.distValue = nil
	if value != nil {
		v.shape = value.Shape()
	} else {
		v.ctx.data.needsInitialization = true
	}
	return nil
}

// SetDistributedValue is like Variable.SetValue(), but uses a distributed.Tensor instead.
//
// If the variable is set to nil, the context is automatically marked as needing initialization, just in case.
//
// WARNING: memory management here is tricky: a call to SetValue triggers the current value to be deallocated,
// and what is returned by a previous call to Value to become invalid.
// The recommendation is not to use this in a concurrent setup -- or to create proper locking mechanisms.
func (v *Variable) SetDistributedValue(distValue *distributed.Tensor) error {
	err := v.Reset()
	if err != nil {
		return errors.WithMessagef(err, "while finalizing previous value of variable %q", v.ScopeAndName())
	}
	v.distValue = distValue
	if distValue != nil {
		v.shape = distValue.Shape()
		v.sharding = distValue.ShardingSpec()
	} else {
		// If setting to nil, marks the context as needing initialization.
		v.ctx.data.needsInitialization = true
	}
	return nil
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
// for the graph (for instance, when applying a gradient descent) by [SetValueGraph].
//
// It's a computation graph building function, and panics on errors.
func (v *Variable) ValueGraph(g *Graph) *Node {
	v.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if found {
		return nodes.valueNode
	}
	// Use a newly created parameter node as the initial graph value Node.
	node, err := v.paramNode(g)
	if err != nil {
		panic(err)
	}
	return node
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
//
// It's a computation graph building function, and panics on errors.
func (v *Variable) SetValueGraph(value *Node) {
	v.AssertValid()
	g := value.Graph()
	g.AssertValid()
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if !found {
		// Creates a parameter node, as this includes the variable as in use for the graph.
		_, err := v.paramNode(g)
		if err != nil {
			panic(err)
		}
		nodes, _ = v.graphToNodes.Load(g.GraphId())
	}
	nodes.valueNode = value
}

// paramNode creates given Graph g's Node that corresponds to the parameter that will be fed with
// the current variable value when the graph is executed. It's the initial value of the variable
// in the computation Graph.
//
// If parameter Node hasn't been created for the Graph g yet, one is created.
//
// Since the value of a variable can change in the middle of the graph (e.g: something that uses the
// variable after a gradient descent is applied) consider using ValueGraph to read the current associated
// value of a variable in a graph.
func (v *Variable) paramNode(g *Graph) (*Node, error) {
	if err := v.CheckValid(); err != nil {
		return nil, err
	}
	if err := g.CheckValid(); err != nil {
		return nil, err
	}
	nodes, found := v.graphToNodes.Load(g.GraphId())
	if found {
		return nodes.paramNode, nil
	}

	paramName := v.ParameterName()
	var paramNode *Node

	// Determine shape and sharding based on strategy.
	strategy := g.DistributedStrategy()
	switch strategy {
	case distributed.SPMD:
		// In SPMD, the graph runs on each device with local data.
		// If a sharding spec is present, we need to calculate the shard shape.
		// If no sharding spec is present, it is assumed replicated, so we use the full shape.
		if v.sharding == nil {
			v.sharding = v.ctx.defaultShardingSpec
		}
		// SPMD uses the shard shape.
		paramNode = graph.ShardedParameter(g, paramName, v.ShardShape(), v.sharding)

	case distributed.AutoSharding:
		// In AutoSharding, the graph sees the global logical shape.
		// We attach the sharding spec to the node if available.
		if v.sharding == nil {
			v.sharding = v.ctx.defaultShardingSpec
		}
		// AutoSharding uses the full logical shape as shape.
		paramNode = graph.ShardedParameter(g, paramName, v.shape, v.sharding)

	default:
		// Default (None) strategy: standard parameter.
		paramNode = graph.Parameter(g, paramName, v.shape)
	}
	nodes = &variableNodes{valueNode: paramNode, paramNode: paramNode}
	v.graphToNodes.Store(g.GraphId(), nodes)
	return paramNode, nil
}

// SetTrainable sets the variable trainable status. Returns itself, so calls can be cascaded.
func (v *Variable) SetTrainable(trainable bool) *Variable {
	v.AssertValid()
	v.Trainable = trainable
	return v
}

// Finalize variable and associate value(s).
// Variable is left in an unusable state, only do this if you are sure this variable is no longer in use.
//
// Usually, one calls Context.Finalize, which in turns finalizes all variables.
func (v *Variable) Finalize() error {
	if v.value != nil {
		v.value.MustFinalizeAll()
		v.value = nil
	}
	v.isValidLocal = false

	if v.distValue != nil {
		v.distValue.FinalizeAll()
		v.distValue = nil
	}
	v.isValidDistributed = false

	for _, t := range v.deviceValues {
		t.MustFinalizeAll()
	}
	v.deviceValues = nil
	v.validDevices = sets.Make[backends.DeviceNum]()

	v.shape = shapes.Invalid()
	v.graphToNodes.Clear()
}
