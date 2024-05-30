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

// Package context defines the Context and Variable types: Context organizes variablesMap
// and variablesMap manages the storage of values typically used as variablesMap.
package context

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context/initializers"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
)

// Context organizes information shared in a model (or anything else). A model can
// spawn multiple computation graphs, e.g: one computation for a training step, one for an evaluation
// step running on batch, and one for a prediction computation on exactly one example.
// All these computation graphs should share the same variable (weight) values (and other information). These
// variables and (hyper-) parameters are organized here.
//
// The Context organizes 3 types of information used by a model, and its graphs:
//
//  1. Variables: model variables or weights.
//  2. Parameters (normal): hyperparameters and also any arbitrary information that
//     needs sharing among the graph building functions using the Context.
//  3. Per-graph parameters: each graph will have it's own value. E.g: the parameter
//     "training" indicates if the model is being used for training or inference, and this value will be different
//     for a training or evaluation graph.
//
// All 3 types of information are organized in "scopes". The Context object is actually a thin wrapper that
// contains the current scope (similar to a current directory) and a link to the actual data. One can change
// scopes by using Context.In("new_scope"): it returns a new Context with the new scope set, but still pointing
// (sharing) all the data with the previous Context. E.g:
//
// One could create a new context with:
//
// ```
//
//		func main() {
//			ctx := context.NewContext(manager)
//			ctx.SetParam("dropout_rate") = 0.2  // Set default dropout to 0.2
//			...
//		}
//
//		func ModelGraph(ctx *context.Context, inputs []*Node) (logits *Node) {
//			...
//			{
//				ctx := ctx.In("output_layer")  // Enter "output_layer" scope in temporary new context (same data, different scope)
//	        	ctx.SetParam("dropout_rate", 0.6)  // Let's say we want the "output_layer" only to have dropout=0.6
//	         	logits = Dense(ctx, logits, output_dim)
//			}  // Exiting "output_later" scope, ctx is back to it's original scope.
//		}
//
// Finally, Context also allows one to checkpoint the variable values (save and load). See checkpoint package.
//
// TODO: Handling of devices with multiple instances (e.g.: multiple GPUs/TPUs).
type Context struct {
	// scope for currently created variables and registration.
	scope string

	// reuse of variables, if set to true.
	reuse bool

	// checked access to variables: whether to check for reuse if variable is new or not. If set
	// to false it makes reuse irrelevant.
	checked bool

	// deviceNumber where to store new variables.
	deviceNumber int

	// initializer is used to initialize variable values for a given shape.
	initializer VariableInitializer

	// contextData, where "data" component content is stored.
	data *contextData
}

// scopedVariableMap name to variable within a scope.
type scopedVariableMap map[string]*Variable

// contextData stores all context information and is shared among various Context, which
// serve only as scoped references.
type contextData struct {
	manager *Manager

	// params holds a model's building (hyper)parameters. Context
	// is agnostic about the semantics here, hence it's a scoped map of scope+key (both strings)
	// to any type of which Context has no knowledge. These values are interpreted by
	// the various model components independently. Some of them are used by trainers. E.g:
	//
	// * "learning_rate" -> float64: used by most optimizers to set a learning rate.
	//
	// Usually the root values are set just after context creation. But layers of a model
	// may set values within scopes for its sub-layers.
	params *ScopedParams

	// graphParams hold models parameters for a particular graph. It's scoped like params,
	// and these values are interpreted by the various model components independently. E.g:
	//
	// * "training" -> bool: used by Context.SetTraining() and Context.IsTraining() to
	//   indicate whether the graph is being used for training or for inference. This is
	//   used by layers that behave differently when training and inference (e.g: Dropout,
	//   BatchNorm, etc.).
	graphParams map[graph.GraphId]*ScopedParams

	// variablesMap for this context organized per scope.
	variablesMap map[string]scopedVariableMap

	// variables is a plain list of all variables, in creation order.
	variables []*Variable

	// loader, if set, is called to check whether there is a previous value of the variable to use.
	loader Loader

	// needsInitialization indicates whether there are uninitialized variables in
	// the context. It's set to false whenever one runs Context.InitializeVariables,
	// and it's set to true whenever a new variable is created without a value.
	needsInitialization bool
}

// Loader can be implemented by any library providing loading of variables for
// Context. Loader implementations need to provide values on demand -- as a variables are used,
// even if they load everything up-front.
//
// An example of a loader in gomlx/context/checkpoints. An example for testing can be found
// in context_test.go:ConstantLoader.
type Loader interface {
	// LoadVariable tries to load the variable v, usually specified by its scope and name.
	// If it's not found, returns false, and initialization continues as usual.
	// Errors can be reported with Context.Panic.
	LoadVariable(ctx *Context, v *Variable) (value tensor.Tensor, found bool)
}

// NewContext constructs a new and empty context.
func NewContext(manager *Manager) *Context {
	ctx := &Context{
		scope:        RootScope,
		deviceNumber: manager.DefaultDeviceNum(),
		checked:      true,
		data: &contextData{
			manager:      manager,
			params:       NewScopedParams(),
			graphParams:  make(map[graph.GraphId]*ScopedParams),
			variablesMap: make(map[string]scopedVariableMap),
		},
	}
	ctx.initializer = initializers.RandomUniformFn(initializers.NoSeed, -0.1, 0.1)
	return ctx
}

const (
	// ScopeSeparator is used between levels of scope. Scope names cannot use this character.
	ScopeSeparator = "/"

	// RootScope is the scope at the very root.
	// Some internal variables (e.g.: default random number generator state) are stored there.
	RootScope = ScopeSeparator
)

// copy creates a copy of the Context, but sharing the same "data" component.
func (ctx *Context) copy() *Context {
	ctx2 := &Context{}
	*ctx2 = *ctx
	return ctx2
}

// Scope returns the full scope path.
//
// Notice that Scope is part of the "reference" component of a Context.
func (ctx *Context) Scope() string {
	return ctx.scope
}

// EscapeScopeName replaces ScopeSeparator in the string and replaces them by "_".
func EscapeScopeName(scopeName string) string {
	return strings.Replace(scopeName, ScopeSeparator, "_", -1)
}

// In returns a new reference to the Context with the extra given scope. No ScopeSeparator ("/") is
// allowed in scope.
//
// Notice that Scope is part of the "reference" component of a Context.
func (ctx *Context) In(scope string) *Context {
	if scope == "" {
		Panicf("cannot use empty scope for Context.In()")
	}
	if strings.Contains(scope, ScopeSeparator) {
		Panicf("cannot use separator %q in scope element %q", ScopeSeparator, scope)
	}

	var newScope string
	if ctx.scope == ScopeSeparator {
		newScope = fmt.Sprintf("%s%s", ScopeSeparator, scope)
	} else {
		newScope = fmt.Sprintf("%s%s%s", ctx.scope, ScopeSeparator, scope)
	}
	return ctx.InAbsPath(newScope)
}

// InAbsPath returns a new reference to the Context with the extra given scope. It should start and have each element
// separated by ScopeSeparator. Use RootScope for the root scope.
//
// Notice that Scope is part of the "reference" component of a Context.
func (ctx *Context) InAbsPath(scopePath string) *Context {
	if !strings.HasPrefix(scopePath, ScopeSeparator) {
		Panicf("absolute scope path must start with separator %q, instead got %q", ScopeSeparator, scopePath)
	}
	_, found := ctx.data.variablesMap[scopePath]
	if !found {
		// Create variable mappings to the new scope.
		ctx.data.variablesMap[scopePath] = make(map[string]*Variable)
	}
	ctx2 := ctx.copy()
	ctx2.scope = scopePath
	return ctx2
}

// Reuse returns a new reference to the Context set to reuse of variables, if it is not already in reuse mode.
// Otherwise, returns itself.
// If checked is false, this setting is irrelevant.
//
// Notice that re-usability is part of the "reference" component of a Context.
func (ctx *Context) Reuse() *Context {
	ctx2 := ctx.copy()
	ctx2.reuse = true
	return ctx2
}

// Unique returns a new reference to the Context, set to only allow new variables, if it is not already in unique mode.
// If checked is false, this setting is irrelevant.
//
// Notice that re-usability is part of the "reference" component of a Context.
func (ctx *Context) Unique() *Context {
	if !ctx.reuse {
		return ctx
	}
	ctx2 := ctx.copy()
	ctx2.reuse = false
	return ctx2
}

// IsReuse returns whether Context is marked for reuse. This is irrelevant if IsChecked is false.
//
// Notice that re-usability is part of the "reference" component of a Context.
func (ctx *Context) IsReuse() bool { return ctx.reuse }

// Checked returns a new context with the checked flag set accordingly.
// If checked is true checks for reuse/uniqueness are checked according to IsReuse().
// If checked is false Variables are dynamically reused or created when needed, without any checks.
// Usually it is set to true when building models -- to prevent layers to overstepping on each other --
// and set to false for supporting variables (like optimizers, metrics and preprocessing).
//
// Notice that re-usability is part of the "reference" component of a Context.
func (ctx *Context) Checked(checked bool) *Context {
	if ctx.checked == checked {
		return ctx
	}
	ctx2 := ctx.copy()
	ctx2.checked = checked
	return ctx2
}

// IsChecked returns whether context is checking reuse rules.
//
// Notice that re-usability is part of the "reference" component of a Context.
func (ctx *Context) IsChecked() bool { return ctx.checked }

// WithInitializer returns a new reference to the Context, with the initializer set.
//
// Notice that default variable initialization is part of the "reference" component of a Context, so this change
// won't affect other context references.
func (ctx *Context) WithInitializer(initializer VariableInitializer) *Context {
	if initializer == nil {
		Panicf("Context.WithInitializer passed a nil initializer")
	}
	ctx2 := ctx.copy()
	ctx2.initializer = initializer
	return ctx2
}

// GetParam returns the value for the given param key, searching successively from
// the current scope back to the root scope ("/"), in case the key is not found.
//
// E.g: if current scope is "/a/b", it will search for the key in "/a/b" scope, then
// in "/a" and finally in "/", and return the first result found.
//
// See also GetGraphParam for parameters that are graph specific.
func (ctx *Context) GetParam(key string) (value any, found bool) {
	return ctx.data.params.Get(ctx.scope, key)
}

// GetParamOr either returns the value for the given param key in the context `ctx`,
// searching successively from the current scope back to the root scope ("/"), or if the
// key is not found, returns the given default value.
//
// It casts the value to the given type, and it will panic is that fails.
//
// It's a convenience method around `ctx.GetParam`.
func GetParamOr[T any](ctx *Context, key string, defaultValue T) T {
	valueAny, found := ctx.GetParam(key)
	if !found {
		return defaultValue
	}
	v, ok := valueAny.(T)
	if !ok {
		Panicf("GetParamOr[%T](ctx, %q, %v): ctx(scope=%q)[%q]=(%T) %#v, and cannot be converted to %T -- "+
			"Notice that when reloading a context from a checkpoint involves decoding them from Json, and "+
			"the original type of the param may have been decoded incorrectly causing this error. "+
			"Many types are automatically corrected, if one is missing please report, or fix it in package "+
			"`checkpoints`, in function `serializedParam.jsonDecodeTypeConvert`. "+
			"Unfortunately, custom parameter types won't work with `checkpoints` (saving/loading), but generic "+
			"`map[string]any` are handled correctly by Json and "+
			"are usually enough for these hyperparameters.",
			v, key, defaultValue, ctx.Scope(), key, valueAny, valueAny, v)
	}
	return v
}

// SetParam sets the given param in the current scope. It will be visible (by GetParam)
// within this scope and descendant scopes (but not by other scopes).
//
// Note: the scoped parameters of the context are saved in `checkpoints` package using
// Json encoding. This works well for `string`, `float64` and `int` and slices of those values,
// but other types may not be recovered correctly later.
// See `checkpoints` package to add support for some other specific type, if you get a different
// type when loading the json.
//
// See also SetGraphParam for parameters that are graph-specific.
func (ctx *Context) SetParam(key string, value any) {
	ctx.data.params.Set(ctx.scope, key, value)
}

// EnumerateParams enumerates all parameters for all scopes calls fn with their values.
func (ctx *Context) EnumerateParams(fn func(scope, key string, value any)) {
	ctx.data.params.Enumerate(fn)
}

// GetGraphParam returns the value for the given param key for the given graph,
// searching successively from the current scope back to the root scope ("/"), in
// case the key is not found.
//
// E.g: if current scope is "/a/b", it will search for the key in "/a/b" scope, then
// in "/a" and finally in "/", and return the first result found.
//
// This is very similar to GetParam, but used for parameters that are graph specific.
// For example Context.IsTraining and Context.SetTraining uses a Graph parameter to
// set this state, as the same Context is used for evaluation/inference graphs and
// training graphs, and they will have different values.
func (ctx *Context) GetGraphParam(g *Graph, key string) (value any, found bool) {
	var graphParams *ScopedParams
	graphParams, found = ctx.data.graphParams[g.GraphId()]
	if !found {
		return
	}
	return graphParams.Get(ctx.scope, key)
}

// GetGraphParamOr either returns the value for the given param key for the given graph,
// searching successively from the current scope back to the root scope ("/"), or if the
// key is not found, returns the given default value.
//
// It casts the value to the given type, and it will panic is that fails.
//
// It's a convenience method around `ctx.GetGraphParam`.
//
// This is very similar to GetParamOr, but used for parameters that are graph specific.
// For example Context.IsTraining and Context.SetTraining uses a Graph parameter to
// set this state, as the same Context is used for evaluation/inference graphs and
// training graphs, and they will have different values.
func GetGraphParamOr[T any](ctx *Context, g *Graph, key string, defaultValue T) T {
	valueAny, found := ctx.GetGraphParam(g, key)
	if !found {
		return defaultValue
	}
	v, ok := valueAny.(T)
	if !ok {
		Panicf("GetGraphParamOr[%T](ctx, g, %q, %v): ctx(scope=%q)[%q]=(%T) %#v, and cannot be converted to %T",
			v, key, defaultValue, ctx.Scope(), key, valueAny, valueAny, defaultValue)
	}
	return valueAny.(T)
}

// SetGraphParam sets the given Graph param in the current scope. It will be visible (by
// GetGraphParam) for this Graph within this scope and descendant scopes (but not by other
// scopes).
//
// Notice each time a new graph is created, the associated "graph parameters" in the context
// will be empty.
//
// This is very similar to SetParam, but used for parameters that are graph specific.
// For example Context.IsTraining and Context.SetTraining uses a Graph parameter to
// set this state, as the same Context is used for evaluation/inference graphs and
// training graphs, and they should have different values.
func (ctx *Context) SetGraphParam(g *Graph, key string, value any) {
	graphParams, found := ctx.data.graphParams[g.GraphId()]
	if !found {
		graphParams = NewScopedParams()
		ctx.data.graphParams[g.GraphId()] = graphParams
	}
	graphParams.Set(ctx.scope, key, value)
}

// EnumerateGraphParams enumerates all parameters for the graph, for all scopes calls fn with their values.
func (ctx *Context) EnumerateGraphParams(g *Graph, fn func(scope, key string, value any)) {
	graphParams, found := ctx.data.graphParams[g.GraphId()]
	if !found {
		// Nothing to enumerate.
		return
	}
	graphParams.Enumerate(fn)
}

// NeedsInitialization returns whether there are variables that needs initialization.
//
// Notice that variables information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) NeedsInitialization() bool {
	return ctx.data.needsInitialization
}

// InitializeVariables initializes all variables in the Context that don't yet have a value.
// Variables create with VariableWithValue or for which values were preloaded are not initialized.
// Errors are returned in Context.Error().
//
// Notice that variables information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) InitializeVariables() {
	var variablesToInitialize []*Variable
	ctx.EnumerateVariables(func(v *Variable) {
		if v.value == nil {
			variablesToInitialize = append(variablesToInitialize, v)
		}
	})
	if len(variablesToInitialize) == 0 {
		// Nothing to do.
		return
	}
	g := ctx.data.manager.NewGraph("InitializeVariables")
	valuesNodes := make([]*Node, 0, len(variablesToInitialize))
	for _, variable := range variablesToInitialize {
		valuesNodes = append(valuesNodes, variable.initializer(g, variable.shape))
	}
	var tuple *tensor.Device
	err := TryCatch[error](func() {
		g.Compile(graph.Tuple(valuesNodes...))
		tuple = g.Run(nil)
	})
	if err != nil {
		panic(errors.WithMessagef(err, "failed to compile/run variable initialization graph"))
	}
	values := tuple.SplitTuple()
	for ii, variable := range variablesToInitialize {
		variable.value = values[ii]
	}
	ctx.data.needsInitialization = false
}

// ExecSetVariablesInParams adds all variables (all scopes) used by the graph to the ParamsMap objects.
//
// `Exec*` methods are used by those implementing an executor (context.Exec) or related tests, not normally
// needed by end users.
func (ctx *Context) ExecSetVariablesInParams(params graph.ParamsMap, g *Graph) {
	g.AssertValid()
	ctx.EnumerateVariables(func(v *Variable) {
		if v.InUseByGraph(g) {
			if v.value == nil {
				Panicf("variable %q not initialized", v.ParameterName())
			}
			params[v.ParamNode(g)] = v.Value().Device(g, g.DeviceNum())
		}
	})
}

// findVariableInScope or nil if not found.
func (ctx *Context) findVariableInScope(name string) *Variable {
	return ctx.InspectVariable(ctx.scope, name)
}

// InspectVariable returns the variable with the given name for inspection. This shouldn't be used during
// building of models, since this bypasses the Reuse checks. It returns nil if a variable with the given
// name hasn't been created.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// The root scope is "/" (RootScope).
func (ctx *Context) InspectVariable(scope, name string) *Variable {
	scopeVars, ok := ctx.data.variablesMap[scope]
	if !ok {
		return nil
	}
	v := scopeVars[name]
	return v
}

// setVariableInScope.
func (ctx *Context) setVariableInScope(name string, v *Variable) {
	vSet, found := ctx.data.variablesMap[ctx.scope]
	if !found {
		vSet = make(scopedVariableMap)
		ctx.data.variablesMap[ctx.scope] = vSet
	}
	vSet[name] = v
	ctx.data.variables = append(ctx.data.variables, v)
}

// VariableWithShape creates or returns an existing variable with the given shape in the current scope.
// It is initialized with the current variable initializer set for the context.
// By default, variables are marked as trainable.
//
// If a Loader is configured (see SetLoader), and the value is available to load, it will override
// the value given here -- e.g.: the value could be actually loaded from the last checkpoint.
//
// Notice that variables information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) VariableWithShape(name string, shape shapes.Shape) *Variable {
	v := ctx.findVariableInScope(name)
	if v == nil && ctx.checked && ctx.reuse {
		Panicf("requested variable %q in scope %q with Context.Reuse set, but variable does not exist", name, ctx.scope)
	}
	if v != nil && ctx.checked && !ctx.reuse {
		Panicf("variable %q for scope %q already exists -- if this was deliberate, use Context.Reuse() or Context.Check(false)", name, ctx.scope)
	}

	if v != nil {
		if !shape.Eq(v.shape) {
			Panicf("requested to reuse variable %q in scope %q, but with different shape from original: previous shape=%s, requested shape=%s",
				name, ctx.scope, v.shape, shape)
		}
		return v
	}

	// New variable: check, create and register it in Context and return.
	v = &Variable{
		ctx:          ctx,
		name:         name,
		scope:        ctx.Scope(),
		shape:        shape,
		Trainable:    true,
		graphToNodes: make(map[graph.GraphId]*variableNodes),
	}
	ctx.setVariableInScope(name, v)

	// Try to load the variable. Report if something failed.
	if ctx.tryToLoad(v) {
		return v
	}

	// Set up variable for initialization.
	v.initializer = ctx.initializer
	ctx.data.needsInitialization = true
	return v
}

// tryToLoad tries to load the variable from the loader. It returns true if it succeeded.
func (ctx *Context) tryToLoad(v *Variable) bool {
	loader := ctx.data.loader
	if loader == nil {
		return false
	}
	value, found := loader.LoadVariable(ctx, v)
	if found {
		if value.Shape().Eq(v.shape) {
			v.value = value
		} else {
			Panicf("loading of variable %q returned shape %s, but variable was created "+
				"with shape %s -- did some hyperparameter change since variable was saved that changed "+
				"the variable shape?", v.ParameterName(), value.Shape(), v.shape)
		}
	}
	return found
}

func valueToTensor(value any) tensor.Tensor {
	if tensorValue, ok := value.(tensor.Tensor); ok {
		return tensorValue
	}
	if node, ok := value.(*Node); ok {
		Panicf(
			"trying to feed a computation graph node (`*computation.Node`) as a concrete value will not work, "+
				"you have to provide a Go value or a tensor here -- *Node provided: %s", node)
	}
	return tensor.FromAnyValue(value)
}

// VariableWithValue creates a variable that is initialized with the given value in the current scope.
// By default, variables are marked as trainable. The value given must be concrete, that is a tensor
// or a normal Go value, that can be converted to a tensor -- a graph *Node does not work here, this
// is assumed to be a constant.
//
// If a Loader is configured (see SetLoader), and the value is available to load, it will override
// the value given here -- e.g.: the value could be actually loaded from the last checkpoint.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) VariableWithValue(name string, value any) *Variable {
	v := ctx.findVariableInScope(name)

	// Check against reuse of variables.
	if ctx.checked && ctx.reuse && v == nil {
		Panicf("requested variable %q in scope %q with Context.Reuse set, but variable does not exist", name, ctx.scope)
	}
	if ctx.checked && !ctx.reuse && v != nil {
		Panicf("variable %q for scope %q already exists", name, ctx.scope)
	}

	var valueT tensor.Tensor
	err := TryCatch[error](func() { valueT = valueToTensor(value) })
	if err != nil {
		panic(errors.WithMessagef(err, "failed to parse value %v for variable %q in scope %q", value, name, ctx.scope))
	}

	if v != nil {
		// Pre-existing variable to reuse: check that the requested and previous shapes are the same.
		if !valueT.Shape().Eq(v.shape) {
			Panicf("requested to reuse variable %q in scope %q, but with value with different shape from original: previous shape=%s, requested value shape=%s",
				name, ctx.scope, v.shape, valueT.Shape())
		}
		return v
	}

	// New variable: check, create and register it in Context and return.
	v = &Variable{
		ctx:          ctx,
		name:         name,
		scope:        ctx.Scope(),
		shape:        valueT.Shape(),
		value:        valueT,
		Trainable:    true, // By default variables are trainable.
		graphToNodes: make(map[graph.GraphId]*variableNodes),
	}
	ctx.setVariableInScope(name, v)

	// Try to load the variable. Report if something failed.
	if ctx.tryToLoad(v) {
		return v
	}
	return v
}

// EnumerateVariables will call fn for each variable in the context. Notice
// the order of visitation is deterministic.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// Example:
//
//	fmt.Println("\nVariables:")
//	ctx.EnumerateVariables(func(v *context.Variable) {
//		fmt.Printf("\t%s::%s: shape=%s\n", v.Scope(), v.Name(), v.Shape())
//	})
func (ctx *Context) EnumerateVariables(fn func(v *Variable)) {
	for _, v := range ctx.data.variables {
		fn(v)
	}
}

// EnumerateVariablesInScope is similar to EnumerateVariables, but enumerate only those under the current
// context scope.
func (ctx *Context) EnumerateVariablesInScope(fn func(v *Variable)) {
	baseScope := ctx.Scope()
	baseScopeWithSeparator := baseScope + ScopeSeparator
	for _, v := range ctx.data.variables {
		if v.Scope() == baseScope || strings.HasPrefix(v.Scope(), baseScopeWithSeparator) {
			fn(v)
		}
	}
}

// NumVariables return the number of variables in this Context.
func (ctx *Context) NumVariables() int {
	return len(ctx.data.variables)
}

// NumParameters returns the summed-up number of all variables.
// It ignores the `DType`, so a `float64` will count as much as a `uint8`.
func (ctx *Context) NumParameters() int {
	total := 0
	ctx.EnumerateVariables(func(v *Variable) {
		total += v.Shape().Size()
	})
	return total
}

// Memory returns the total number of bytes summed across all variables.
// It does not include associated pointers and structures, just the bytes used by the raw data.
//
// Example:
//
//	fmt.Printf("Model memory usage: %s", data.ByteCountIEC(ctx.Memory()))
func (ctx *Context) Memory() int64 {
	total := int64(0)
	ctx.EnumerateVariables(func(v *Variable) {
		total += v.Shape().Memory()
	})
	return total
}

// Loader returns the current configured Loader for this context. See SetLoader for details on how the
// Loader is used.
//
// Notice that loader configuration is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) Loader() Loader {
	return ctx.data.loader
}

// SetLoader configures given loader to be used as the default Loader for this Context.
//
// Loader is used just after any new variable is created, either with VariableWithValue or VariableWithShape.
// If the Loader has a value of the variable created, it will override the value given in VariableWithValue, or
// skip the initializer for VariableWithShape.
//
// An example of a loader in gomlx/context/checkpoints.
//
// Notice that loader configuration is stored in the "data" component of Context objects, and is shared
// among all connected context references.
func (ctx *Context) SetLoader(loader Loader) {
	ctx.data.loader = loader
}

// ExecPopulateGraphParamsMap will enter the parameter values for every variable used in the given graph.
//
// `Exec*` methods are used by those implementing an executor (context.Exec) or related tests, not normally
// needed by end users.
func (ctx *Context) ExecPopulateGraphParamsMap(g *Graph, params graph.ParamsMap) {
	graphId := g.GraphId()
	ctx.EnumerateVariables(func(v *Variable) {
		nodes, found := v.graphToNodes[graphId]
		if !found {
			return
		}
		params[nodes.paramNode] = v.Value()
	})
}

// execPopulateGraphParamsSlice will fill the graph parameter values for every variable used in the given graph.
// It keeps a cache of the variables' mapping for faster access.
//
// `Exec*` methods are used by those implementing an executor (context.Exec) or related tests, not normally
// needed by end users.
func (ctx *Context) execPopulateGraphParamsSlice(g *Graph, params []*tensor.Device) {
	graphId := g.GraphId()
	ctx.EnumerateVariables(func(v *Variable) {
		nodes, found := v.graphToNodes[graphId]
		if !found {
			return
		}
		if nodes == nil || nodes.paramNode == nil || nodes.paramNode.ParameterHandle() == graph.InvalidParameterHandle {
			Panicf("invalid paramNode for variable %q", v.ParameterName())
		}
		var deviceT *tensor.Device
		err := TryCatch[error](func() { deviceT = v.Value().Device(g.Manager(), g.DeviceNum()) })
		if err != nil {
			panic(errors.WithMessagef(err, "failed to transfer variable \"%s::%s\" value to device",
				v.Scope(), v.Name()))
		}
		params[nodes.paramNode.ParameterHandle()] = deviceT
	})
}

// BuildTrainableVariablesGradientsGraph returns the gradient of the loss with respect to each trainable variable
// in the context that was used in the current graph.
// It returns a tuple Node.
// Non-trainable variables (Variable.Trainable == false) are not touched.
//
// Typically, this is used by an optimizer.
//
// Note, if during the computation graph the value of the variable is changed with Variable.SetValueGraph,
// this will calculate the gradient with respect to the new value (*Node) set.
func (ctx *Context) BuildTrainableVariablesGradientsGraph(loss *Node) []*Node {
	g := loss.Graph()
	var trainableVars []*Node
	ctx.EnumerateVariables(func(v *Variable) {
		if v.Trainable && v.InUseByGraph(g) {
			trainableVars = append(trainableVars, v.ValueGraph(g))
		}
	})
	return graph.Gradient(loss, trainableVars...)
}

// GetParam from Context object and cast to the give type. If
// parameter name is not defined, or if it cannot be cast to the given type,
// return defaultValue instead.
//
// It's a typed wrapper to Context.GetParam()
func GetParam[T any](ctx *Context, name string, defaultValue T) T {
	valueI, found := ctx.GetParam(name)
	if !found {
		return defaultValue
	}
	value, ok := valueI.(T)
	if ok {
		return value
	}

	// Try converting, for instance, a float32 could be converted to float64.
	v := reflect.ValueOf(valueI)
	var t T
	typeOfT := reflect.TypeOf(t)
	if !v.CanConvert(typeOfT) {
		log.Printf("Tried to read hyperparameter %q as %t, but failed because it was type %s.",
			name, any(t), v.Type())
		return defaultValue
	}
	return v.Convert(typeOfT).Interface().(T)
}

// GetGraphParam from Context object and cast to the give type. If
// parameter name is not defined, or if it cannot be cast to the given type,
// return defaultValue instead.
//
// It's a typed wrapper to Context.GetGraphParam()
func GetGraphParam[T any](ctx *Context, g *Graph, name string, defaultValue T) T {
	valueI, found := ctx.GetGraphParam(g, name)
	if !found {
		return defaultValue
	}
	value, ok := valueI.(T)
	if ok {
		return value
	}

	// Try converting, for instance, a float32 could be converted to float64.
	v := reflect.ValueOf(valueI)
	var t T
	typeOfT := reflect.TypeOf(t)
	if !v.CanConvert(typeOfT) {
		log.Printf("Tried to read hyperparameter %q as %t, but failed because it was type %s.",
			name, any(t), v.Type())
		return defaultValue
	}
	return v.Convert(typeOfT).Interface().(T)
}

const TrainingGraphParamKey = "training"

// IsTraining returns whether context is being used for training. This is only a convention and is defined
// by having Globals["training"] == true. See SetTraining to change this value.
//
// Notice that global parameters is part of the "reference" component of a Context, so this change
// won't affect other connected context references.
func (ctx *Context) IsTraining(g *Graph) bool {
	isTraining, found := ctx.GetGraphParam(g, TrainingGraphParamKey)
	return found && isTraining.(bool)
}

// SetTraining marks the context for the given graph as training. This is a convention
// adopted by the library components, and it simply sets
// Context.Globals["training"] to the given value. See IsTraining to check for this value.
//
// Notice that global parameters is part of the "reference" component of a Context, so this change
// won't affect other connected context references.
func (ctx *Context) SetTraining(g *Graph, value bool) {
	ctx.SetGraphParam(g, TrainingGraphParamKey, value)
}
