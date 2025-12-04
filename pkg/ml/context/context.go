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
	"encoding"
	"fmt"
	"iter"
	"reflect"
	"strings"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/scoped"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"golang.org/x/exp/slices"
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
//  2. Parameters: hyperparameters and also any arbitrary information that
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
//	func main() {
//		ctx := context.New()
//		ctx.SetParam("dropout_rate") = 0.2  // Set default dropout to 0.2
//		...
//	}
//
//	func ModelGraph(ctx *context.Context, inputs []*Node) (logits *Node) {
//		...
//		{
//			ctx := ctx.In("output_layer")  // Enter "output_layer" scope in temporary new context (same data, different scope)
//			ctx.SetParam("dropout_rate", 0.6)  // Let's say we want the "output_layer" only to have dropout=0.6
//			logits = Dense(ctx, logits, output_dim)
//		}  // Exiting "output_later" scope, ctx is back to it's original scope.
//	}
//
// Finally, Context also allows one to checkpoint the variable values (save and load). See the checkpoints package.
//
// Variable duplicate creation checking:
// the context is by default configure to with Context.Checked(true), which checks at every variable creation whether
// the variable already exists. This is useful to prevent unintended reuse of variables. When checked, variable creation
// (with Context.VariableWithShape and Context.VariableWithValue) will panic if:
//
// - Context.Unique() (the default) and variable already exists (or was loaded);
// - Context.Reuse() and variable didn't exist (or was not loaded);
//
// Remember to set Context.Reuse if you expect to load the variables, or disable Context.Checked(false) if only some
// variables are going to be loaded.
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
	// params holds a model's building (hyper)parameters. Context
	// is agnostic about the semantics here, hence it's a scoped map of scope+key (both strings)
	// to any type of which Context has no knowledge. These values are interpreted by
	// the various model components independently. Some of them are used by trainers. E.g:
	//
	// * "learning_rate" -> float64: used by most optimizers to set a learning rate.
	//
	// Usually the root values are set just after context creation. But layers of a model
	// may set values within scopes for its sub-layers.
	params *scoped.Params

	// graphParams hold models parameters for a particular graph. It's scoped like params,
	// and these values are interpreted by the various model components independently. E.g:
	//
	// * "training" -> bool: used by Context.SetTraining() and Context.IsTraining() to
	//   indicate whether the graph is being used for training or for inference. This is
	//   used by layers that behave differently when training and inference (e.g: Dropout,
	//   BatchNorm, etc.).
	graphParams map[graph.GraphId]*scoped.Params

	// variablesMap for this context organized per scope.
	variablesMap map[string]scopedVariableMap

	// variables is a plain list of all variables, in creation order.
	variables []*Variable

	// loader, if set, is called to check whether there is a previous value of the variable to use.
	loader Loader

	// needsInitialization indicates whether there are uninitialized variables in
	// the context. It's set to false whenever one runs Context.InitializeVariables,
	// and it's set to true whenever a new variable is created without a value.
	//
	// If it is set to false, it means all variables are set, and there is no need to initialize.
	// But a true is not 100% certain: it may be that the variables are already set, it requires checking.
	needsInitialization bool

	// defaultShardingSpec used for new variables, if execution is distributed.
	defaultShardingSpec *distributed.ShardingSpec
}

// Loader can be implemented by any library providing loading of variables for
// Context. Loader implementations need to provide values on demand -- as a variables are used,
// even if they load everything up-front.
//
// An example of a loader in gomlx/context/checkpoints. An example for testing can be found
// in context_test.go:ConstantLoader.
type Loader interface {
	// LoadVariable tries to load the variable v pointed by its scope and name.
	// If it's not found, returns false, and initialization continues as usual.
	// Errors can be reported with Context.Panic.
	//
	// It is called at most once for each variable: if a values is loaded owner is transferred and the Loader
	// can "forget" about that variable, it's assumed to be transferred to the context.
	LoadVariable(ctx *Context, scope, name string) (value *tensors.Tensor, found bool)

	// DeleteVariable is called whenever Context.DeleteVariable is called. The deletion should cascade to the
	// loader, otherwise the variable will reappear after deletion.
	//
	// If the variable doesn't exist in the loader, it should be a no-op.
	DeleteVariable(ctx *Context, scope, name string) error
}

// New returns an empty context, associated with freshly created data.
//
// Something to be mindful: the default variable initializer is a random uniform noise from [-0.05, 0.05].
// You can change the default by importing ml/context/default, or you can
// set your own with Context.WithInitializer. See available initializers in ml/context/initializers.
func New() *Context {
	ctx := &Context{
		scope:   RootScope,
		checked: true,
		data: &contextData{
			params:       scoped.New(ScopeSeparator),
			graphParams:  make(map[graph.GraphId]*scoped.Params),
			variablesMap: make(map[string]scopedVariableMap),
		},
	}
	ctx.initializer = DefaultInitializer(ctx)
	return ctx
}

// Clone does a (mostly) deep copy of the context, with new variable values, and a clone of the parameters.
// Both the context "pointer" (current scope) and the underlying data are cloned, with the following
// exceptions:
//
//   - The default initializer is simply copied.
//   - Graph params and variables are not cloned: the new context is assumed not to be associated to any
//     graph.
//   - Loader (typically a checkpoint saver/loader) is not cloned, but it can be manually
//     copied over by with newCtx.SetLoader(ctx.Loader()).
//   - The context state is copied over: needing initialization of variables, if to be checked
//     for new/reuse of variables, etc.
//   - The default sharding spec is simply copied (not cloned).
func (ctx *Context) Clone() (*Context, error) {
	newCtx := New()
	newCtx.scope = ctx.scope
	newCtx.reuse = ctx.reuse
	newCtx.checked = ctx.checked
	newCtx.initializer = ctx.initializer
	newCtx.data = &contextData{
		graphParams:  make(map[graph.GraphId]*scoped.Params),
		variablesMap: make(map[string]scopedVariableMap),
	}
	newCtx.data.needsInitialization = ctx.data.needsInitialization
	newCtx.data.params = ctx.data.params.Clone()
	newCtx.data.defaultShardingSpec = ctx.data.defaultShardingSpec
	var err error
	for v := range ctx.IterVariables() {
		_, err = v.CloneToContext(newCtx)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to clone variable %q while cloning the Context",
				v.Name())
		}
	}
	return newCtx, nil
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

// JoinScope and name into a single string.
// If scope is empty, name is returned.
// See also SplitScope.
func JoinScope(scope, name string) string {
	if strings.HasSuffix(scope, ScopeSeparator) {
		return scope + name
	}
	if scope == "" {
		return name
	}
	return fmt.Sprintf("%s%s%s", scope, ScopeSeparator, name)
}

// SplitScope splits the scope from the name for a combined string, typically created by JoinScope.
// If there is no scope configured, scope is set to "".
func SplitScope(scopeAndName string) (scope, name string) {
	if !strings.HasPrefix(scopeAndName, ScopeSeparator) {
		// No scope in scopeAndName.
		scope = ""
		name = scopeAndName
		return
	}
	separationIdx := strings.LastIndex(scopeAndName, ScopeSeparator)
	name = scopeAndName[separationIdx+1:]
	if separationIdx == 0 {
		scope = RootScope
	} else {
		scope = scopeAndName[:separationIdx]
	}
	return
}

// Scope returns the full scope path.
//
// Notice that Scope is part of the "reference" component of a Context.
func (ctx *Context) Scope() string {
	return ctx.scope
}

// EscapeScopeName replaces ScopeSeparator in the string and replaces them by "_".
func EscapeScopeName(scopeName string) string {
	return strings.ReplaceAll(scopeName, ScopeSeparator, "_")
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

// Inf returns a new reference to the Context with the extra given scope. No ScopeSeparator ("/") is
// allowed in scope.
// The name of the new scope is given as a format + args, which are passed to fmt.Sprintf.
//
// It is a shortcut to Context.In combined with fmt.Sprintf.
//
// Notice that Scope is part of the "reference" component of a Context.
func (ctx *Context) Inf(format string, args ...any) *Context {
	scope := fmt.Sprintf(format, args...)
	return ctx.In(scope)
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
// See also:
//
// * GetParamOr to get a parameter with a default, if one doesn't exist.
// * GetGraphParam for parameters that are graph specific.
func (ctx *Context) GetParam(key string) (value any, found bool) {
	return ctx.data.params.Get(ctx.scope, key)
}

// MustGetParam is like GetParam, but panics if the parameter is not found, or if it is not of type T.
//
// This is helpful when reading a hyperparameter that must be defined in a context during model building.
//
// It tries to cast the value to the given type. If it fails, it tries to convert the
// value to the given type (so an `int` will be converted to a `float64` transparently).
// If that also fails, an explaining exception is thrown.
func MustGetParam[T any](ctx *Context, key string) T {
	var t T
	valueAny, found := ctx.GetParam(key)
	if !found {
		Panicf("parameter %q (of type %T) not found in scope %q (and its parents)", key, t, ctx.Scope())
	}

	v := reflect.ValueOf(valueAny)
	typeOfT := reflect.TypeOf(t)
	valueT := reflect.New(typeOfT)
	if valueT.Type().Implements(textUnmarshalerType) && v.Kind() == reflect.String {
		if err := valueT.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(v.String())); err != nil {
			Panicf("can't UnmarshalText %s to %s", v.String(), typeOfT.String())
		}
		return valueT.Elem().Interface().(T)
		// Try converting; for instance, a float32 could be converted to float64.
	} else if !v.CanConvert(typeOfT) {
		Panicf("MustGetParam/GetParamOr[%T](ctx, %q): ctx(scope=%q)[%q]=(%T) %#v, and cannot be converted to %T -- "+
			"Notice that when reloading a context from a checkpoint involves decoding them from Json, and "+
			"the original type of the param may have been decoded incorrectly causing this error. "+
			"Many types are automatically corrected, if one is missing please report, or fix it in package "+
			"`checkpoints`, in function `serializedParam.jsonDecodeTypeConvert`. "+
			"Unfortunately, custom parameter types won't work with `checkpoints` (saving/loading), but generic "+
			"`map[string]any` are handled correctly by Json and "+
			"are usually enough for these hyperparameters.",
			v, key, ctx.Scope(), key, valueAny, valueAny, v)
	}
	return v.Convert(typeOfT).Interface().(T)
}

// GetParamOr either returns the value for the given param key in the context `ctx`,
// searching successively from the current scope back to the root scope ("/"), or if the
// key is not found or the key is set to nil, it returns the given default value.
//
// It tries to cast the value to the given type. If it fails, it tries to convert the
// value to the given type (so an `int` will be converted to a `float64` transparently).
// If that also fails, an explaining exception is thrown.
//
// It's a convenience method around `ctx.GetParam`.
func GetParamOr[T any](ctx *Context, key string, defaultValue T) T {
	valueAny, found := ctx.GetParam(key)
	if !found || valueAny == nil {
		return defaultValue
	}
	value, ok := valueAny.(T)
	if ok {
		return value
	}
	return MustGetParam[T](ctx, key)
}

var textUnmarshalerType = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()

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

// SetParams sets a collection of parameters in the current scope. It will be visible (by GetParam)
// within this scope and descendant scopes (but not by other scopes).
//
// This is a shortcut to multiple calls to `Context.SetParam` and the same observations apply.
func (ctx *Context) SetParams(keyValues map[string]any) {
	for key, value := range keyValues {
		ctx.data.params.Set(ctx.scope, key, value)
	}
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
	var graphParams *scoped.Params
	graphParams, found = ctx.data.graphParams[g.GraphId()]
	if !found {
		return
	}
	return graphParams.Get(ctx.scope, key)
}

// MustGetGraphParam is like GetGraphParam, but panics if the parameter is not found, or if it is not of type T.
//
// This is helpful when reading a graph hyperparameter that must be defined in a context during model building.
//
// It tries to cast the value to the given type. If it fails, it tries to convert the
// value to the given type (so an `int` will be converted to a `float64` transparently).
// If that also fails, an explaining exception is thrown.
//
// This is similar to MustGetParam, but it's used for parameters that are graph-specific.
// For example, Context.IsTraining and Context.SetTraining use a Graph parameter to
// set the training state, as the same Context is used for evaluation/inference graphs and
// training graphs, and their values will be different.
func MustGetGraphParam[T any](ctx *Context, g *Graph, key string) T {
	var t T
	valueAny, found := ctx.GetGraphParam(g, key)
	if !found {
		Panicf("parameter %q (of type %T) not found in scope %q (and its parents)", key, t, ctx.Scope())
	}

	v := reflect.ValueOf(valueAny)
	typeOfT := reflect.TypeOf(t)
	valueT := reflect.New(typeOfT)
	if valueT.Type().Implements(textUnmarshalerType) && v.Kind() == reflect.String {
		if err := valueT.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(v.String())); err != nil {
			Panicf("can't UnmarshalText %s to %s", v.String(), typeOfT.String())
		}
		return valueT.Elem().Interface().(T)
		// Try converting; for instance, a float32 could be converted to float64.
	} else if !v.CanConvert(typeOfT) {
		Panicf("MustGetGraphParam/GetGraphParamOr[%T](ctx, %q): ctx(scope=%q)[%q]=(%T) %#v, and cannot be converted to %T -- "+
			"Notice that when reloading a context from a checkpoint involves decoding them from Json, and "+
			"the original type of the param may have been decoded incorrectly causing this error. "+
			"Many types are automatically corrected, if one is missing please report, or fix it in package "+
			"`checkpoints`, in function `serializedParam.jsonDecodeTypeConvert`. "+
			"Unfortunately, custom parameter types won't work with `checkpoints` (saving/loading), but generic "+
			"`map[string]any` are handled correctly by Json and "+
			"are usually enough for these hyperparameters.",
			v, key, ctx.Scope(), key, valueAny, valueAny, v)
	}
	return v.Convert(typeOfT).Interface().(T)
}

// GetGraphParamOr either returns the value for the given param key for the given graph,
// searching successively from the current scope back to the root scope ("/"), or if the
// key is not found, or the value is set to nil, it returns the given default value.
//
// It tries to cast the value to the given type. If it fails, it tries to convert the
// value to the given type (so an `int` will be converted to a `float64` transparently).
// If that also fails, an explaining exception is thrown.
//
// It's a convenience method around `ctx.GetGraphParam`.
//
// This is very similar to GetParamOr, but it's used for parameters that are graph-specific.
// For example, Context.IsTraining and Context.SetTraining use a Graph parameter to
// set this state, as the same Context is used for evaluation/inference graphs and
// training graphs, and their values will be different.
func GetGraphParamOr[T any](ctx *Context, g *Graph, key string, defaultValue T) T {
	// GetGraphParam from the Context object and cast to the give type.
	// If the parameter key is not defined, or if it cannot be cast to the given type,
	// return defaultValue instead.
	//
	// It's a typed wrapper to Context.GetGraphParam()
	valueAny, found := ctx.GetGraphParam(g, key)
	if !found || valueAny == nil {
		return defaultValue
	}
	value, ok := valueAny.(T)
	if ok {
		return value
	}

	return MustGetGraphParam[T](ctx, g, key)
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
		graphParams = scoped.New(ScopeSeparator)
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
//   - configExec: closure to configure the initialization of the variables executor.
//     This is a hook that allows one to set distributed execution (and having the variables already pre-sharded
//     and stored on the correct devices). If can be nil, then nothing is configured.
//
// Notice that variables information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// Initialization functions are executed on the given backend.
//
// InitializeVariables also resets the RNG state for the context, if is not yet set.
func (ctx *Context) InitializeVariables(
	backend backends.Backend, configExec func(initializerExec *Exec) error) error {
	// Collect variables that need initialization.
	var variablesToInitialize []*Variable
	for v := range ctx.IterVariables() {
		if !v.HasValue() {
			variablesToInitialize = append(variablesToInitialize, v)
		}
	}
	if len(variablesToInitialize) == 0 {
		// Nothing to do.
		return nil
	}

	// Execute initialization for collected variables.
	e, err := NewExec(backend, ctx, func(ctx *Context, g *Graph) []*Node {
		g = g.WithName("VariableInitialization")
		initialValues := make([]*Node, 0, len(variablesToInitialize))
		for _, variable := range variablesToInitialize {
			if variable.initializer == nil {
				Panicf("failed to initialize variable %q: initializer was not configured (maybe it was read from"+
					" disk and an initialzier was not set)", variable.ScopeAndName())
			}
			initialValues = append(initialValues, variable.initializer(g, variable.shape))
		}
		return initialValues
	})
	if err != nil {
		return errors.WithMessagef(err, "failed to create executor for variable initialization")
	}
	if configExec != nil {
		// Caller configuration of the executor.
		err := configExec(e)
		if err != nil {
			return errors.WithMessagef(err, "failed to configure executor for variable initialization")
		}
	}
	e.isInitializeVariablesExec = true // Disallow recursive creation of variables within variable initialization.
	values, err := e.Exec()
	if err != nil {
		return errors.WithMessagef(err, "failed to compile/run variable initialization graph")
	}
	numDevices := e.NumDevices()
	if len(values) != numDevices*len(variablesToInitialize) {
		return errors.Errorf("failed to initialize variables: expected numDevices(%d) * %d values, got %d",
			numDevices, len(variablesToInitialize), len(values))
	}
	for ii, variable := range variablesToInitialize {
		if !values[ii].Ok() {
			return errors.Errorf(
				"graph execution to initialize variables failed: variable %q (#%d) generated value was invalid -- "+
					"maybe other variables as well", variable.ScopeAndName(), ii)
		}
		variable.value = values[ii]
	}
	ctx.data.needsInitialization = false
	return nil
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
			params[v.ValueGraph(g)] = v.value
		}
	})
}

// InspectVariable is an alias to GetVariableByScopeAndName.
// Deprecated: use GetVariableByScopeAndName, this will be removed in future releases.
func (ctx *Context) InspectVariable(scope, name string) *Variable {
	return ctx.GetVariableByScopeAndName(scope, name)
}

// GetVariableByScopeAndName returns the variable with the given name for inspection. It returns nil if a variable with the given
// name hasn't been created.
//
// It is not affected by [Context.Reuse] checks.
//
// This will trigger the loading of the variable if a loader (like `checkpoint.Checkpoint`) is attached.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// The root scope is "/" (RootScope).
func (ctx *Context) GetVariableByScopeAndName(scope, name string) *Variable {
	scopeVars, ok := ctx.data.variablesMap[scope]
	if ok {
		v, found := scopeVars[name]
		if found {
			return v
		}
	}

	// Try to load it, if a loader (checkpoint handler) is configured.
	loader := ctx.data.loader
	if loader == nil {
		return nil
	}
	value, found := loader.LoadVariable(ctx, scope, name)
	if !found {
		return nil
	}
	v := &Variable{
		ctx:       ctx,
		name:      name,
		scope:     scope,
		shape:     value.Shape(),
		value:     value,
		Trainable: true,
	}
	ctx.InAbsPath(scope).setVariableInScope(name, v)
	return v
}

// GetVariable returns the variable in the current context scope.
//
// It is not affected by [Context.Reuse] checks.
//
// This will trigger the loading of the variable if a loader (like `checkpoint.Checkpoint`) is attached.
func (ctx *Context) GetVariable(name string) *Variable {
	return ctx.GetVariableByScopeAndName(ctx.Scope(), name)
}

// InspectVariableInScope is an alias to GetVariable.
// Deprecated: use GetVariable instead. This alias will be removed in future releases.
func (ctx *Context) InspectVariableInScope(name string) *Variable {
	return ctx.GetVariableByScopeAndName(ctx.Scope(), name)
}

// InspectVariableIfLoaded returns the variable if it exists already, but it won't attempt to load it.
//
// It is similar to [GetVariableByScopeAndName] but won't attempt to load the variable if it's not yet loaded.
//
// Not normally needed, but may be handy for testing. See also [checkpoints.Config.Immediate].
func (ctx *Context) InspectVariableIfLoaded(scope, name string) *Variable {
	scopeVars, ok := ctx.data.variablesMap[scope]
	if !ok {
		return nil
	}
	return scopeVars[name]
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

// DeleteVariable if it exists.
//
// This should not be called from a graph building function or from within EnumerateVariables: the results are undefined if you do.
func (ctx *Context) DeleteVariable(scope, name string) error {
	// Even if variable doesn't exist in context yet, we need to remove it from the loader,
	// since it may only exist there at first.
	loader := ctx.data.loader
	if loader != nil {
		err := loader.DeleteVariable(ctx, scope, name)
		if err != nil {
			return err
		}
	}
	scopeVars, ok := ctx.data.variablesMap[scope]
	if !ok {
		return nil
	}
	v := scopeVars[name]
	if v == nil {
		return nil
	}
	v.value = nil
	v.graphToNodes.Map.Clear()
	delete(scopeVars, name)
	if len(scopeVars) == 0 {
		delete(ctx.data.variablesMap, scope)
	}
	ctx.data.variables = slices.DeleteFunc(
		ctx.data.variables, func(vCandidate *Variable) bool {
			return vCandidate == v
		})
	return nil
}

// DeleteVariablesInScope deletes all variables under the current scope (ctx.Scope()).
// It also resets the variables, freeing its values.
//
// This should not be called from a graph building function or from within EnumerateVariables: the results are undefined if you do.
func (ctx *Context) DeleteVariablesInScope() error {
	variables := make([]*Variable, 0, len(ctx.data.variables))
	baseScope := ctx.Scope()
	baseScopeWithSeparator := baseScope + ScopeSeparator
	if baseScope == RootScope {
		baseScopeWithSeparator = baseScope
	}
	loader := ctx.data.loader
	for _, v := range ctx.data.variables {
		if v.Scope() != baseScope && !strings.HasPrefix(v.Scope(), baseScopeWithSeparator) {
			// Not in scope, preserve variable.
			variables = append(variables, v)
			continue
		}

		// Free variable space.
		err := v.Reset()
		if err != nil {
			return err
		}

		// Inform the loader about the variable being deleted.
		if loader != nil {
			loader.DeleteVariable(ctx, v.Scope(), v.Name())
		}

		// Remove reference to variable.
		scopeVars, ok := ctx.data.variablesMap[v.Scope()]
		if !ok {
			continue
		}
		delete(scopeVars, v.name)
		if len(scopeVars) == 0 {
			delete(ctx.data.variablesMap, v.Scope())
		}
	}
	ctx.data.variables = variables
	return nil
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
//
// If Context is set with Context.Checked(true), this may panic if:
//
// - Context.Unique() and variable already exists (or was loaded);
// - Context.Reuse() and variable didn't exist (or was not loaded);
func (ctx *Context) VariableWithShape(name string, shape shapes.Shape) *Variable {
	v := ctx.GetVariableByScopeAndName(ctx.scope, name)
	if v == nil && ctx.checked && ctx.reuse {
		Panicf("requested variable %q in scope %q with Context.Reuse set, but variable does not exist", name, ctx.scope)
	}
	if v != nil && ctx.checked && !ctx.reuse {
		Panicf(
			"variable %q for scope %q already exists -- if this was deliberate, use Context.Reuse() or Context.Check(false)",
			name,
			ctx.scope,
		)
	}

	if v != nil {
		if !shape.Equal(v.shape) {
			Panicf(
				"requested to reuse variable %q in scope %q, but with different shape from original: previous shape=%s, requested shape=%s",
				name,
				ctx.scope,
				v.shape,
				shape,
			)
		}
		// We want to update/register the initializer, even if the value is already set (maybe read from a checkpoint).
		v.initializer = ctx.initializer
		return v
	}

	// New variable: check, create and register it in Context and return.
	v = &Variable{
		ctx:          ctx,
		name:         name,
		scope:        ctx.Scope(),
		shape:        shape,
		Trainable:    true,
		shardingSpec: ctx.data.defaultShardingSpec,
	}
	ctx.setVariableInScope(name, v)

	// Set up variable for initialization.
	v.initializer = ctx.initializer
	ctx.data.needsInitialization = true
	return v
}

func valueToTensor(value any) *tensors.Tensor {
	if tensorValue, ok := value.(*tensors.Tensor); ok {
		return tensorValue
	}
	if node, ok := value.(*Node); ok {
		Panicf(
			"trying to feed a computation graph node (`*computation.Node`) as a concrete value will not work, "+
				"you have to provide a Go value or a tensor here -- *Node provided: %s", node)
	}
	return tensors.FromAnyValue(value)
}

// VariableWithValue creates or returns a variable initialized with the given value in the current scope.
// If the variable already exists, its value is not overwritten.
//
// By default, variables are marked as trainable.
//
// The value given must be concrete, that is a tensor
// or a normal Go value, that can be converted to a tensor.
//
// A graph *Node does not work here, this is assumed to be a concrete tensor value.
// See VariableWithValueGraph instead, to create a variable with a graph *Node.
//
// If a Loader is configured (see SetLoader), and the value is available to load, it will override
// the value given here -- e.g.: the value could be actually loaded from the last checkpoint.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// If Context is set with Context.Checked(true), this may panic if:
//
// - Context.Unique() and variable already exists (or was loaded);
// - Context.Reuse() and variable didn't exist (or was not loaded);
//
// See Variable.SetValue if you want to overwrite the value of an existing variable.
//
// This is a graph building function and so it may panic if the variable cannot be created.
func (ctx *Context) VariableWithValue(name string, defaultValue any) *Variable {
	v := ctx.GetVariableByScopeAndName(ctx.scope, name)

	// Check against reuse of variables.
	if ctx.checked && ctx.reuse && v == nil {
		Panicf("requested variable %q in scope %q with Context.Reuse set, but variable does not exist",
			name, ctx.scope)
	}
	if ctx.checked && !ctx.reuse && v != nil {
		Panicf("variable %q for scope %q already exists", name, ctx.scope)
	}

	var valueT *tensors.Tensor
	err := TryCatch[error](func() { valueT = valueToTensor(defaultValue) })
	if err != nil {
		panic(
			errors.WithMessagef(
				err,
				"failed to parse defaultValue %v for variable %q in scope %q",
				defaultValue,
				name,
				ctx.scope,
			),
		)
	}

	if v != nil {
		// Pre-existing variable to reuse: check that the requested and previous shapes are the same.
		if !valueT.Shape().Equal(v.shape) {
			Panicf(
				"requested to reuse variable %q in scope %q, but with defaultValue with different shape from original: previous shape=%s, requested defaultValue shape=%s",
				name,
				ctx.scope,
				v.shape,
				valueT.Shape(),
			)
		}
		return v
	}

	// New variable: check, create and register it in Context and return.
	v = &Variable{
		ctx:       ctx,
		name:      name,
		scope:     ctx.Scope(),
		shape:     valueT.Shape(),
		value:     valueT,
		Trainable: true, // By default variables are trainable.
	}
	ctx.setVariableInScope(name, v)
	return v
}

// VariableWithValueGraph creates a variable in the current scope and sets it with graph computed *Node.
//
// By default, variables are marked as trainable.
//
// The value given must be a graph *Node, meaning it is computed in the graph.
// If you want to create a value with a default concrete (tensors.Tensor) value, use VariableWithValue instead.
//
// This is equivalent to calling VariableWithShape followed by Variable.SetValueGraph.
// If the variable already exists, its value will be overwritten.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// If Context is set with Context.Checked(true), this may panic if:
//
// - Context.Unique() and variable already exists (or was loaded);
// - Context.Reuse() and variable didn't exist (or was not loaded);
//
// This is a graph building function and so it may panic if the variable cannot be created.
func (ctx *Context) VariableWithValueGraph(name string, value *Node) *Variable {
	// Create a zero-initialized context.
	zeroCtx := ctx.WithInitializer(func(g *Graph, shape shapes.Shape) *Node {
		return graph.Zeros(g, shape)
	})
	v := zeroCtx.VariableWithShape(name, value.Shape())
	v.SetValueGraph(value)
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
//
// Deprecated: use IterVariables instead.
func (ctx *Context) EnumerateVariables(fn func(v *Variable)) {
	for _, v := range ctx.data.variables {
		fn(v)
	}
}

// IterVariables returns an iterator that yields each variable in the context.
// The order of iteration is deterministic.
//
// Notice that variables' information is stored in the "data" component of Context objects, and is shared
// among all connected context references.
//
// Variables not yet materialized, for instance with checkpoints with lazy loading, are not listed here.
//
// Example:
//
//	fmt.Println("\nVariables:")
//	for v := range ctx.IterateVariables() {
//		fmt.Printf("\t%s::%s: shape=%s\n", v.Scope(), v.Name(), v.Shape())
//	}
func (ctx *Context) IterVariables() iter.Seq[*Variable] {
	return func(yield func(*Variable) bool) {
		for _, v := range ctx.data.variables {
			if !yield(v) {
				return
			}
		}
	}
}

// EnumerateVariablesInScope is similar to EnumerateVariables, but enumerate only those under the current
// context scope.
func (ctx *Context) EnumerateVariablesInScope(fn func(v *Variable)) {
	baseScope := ctx.Scope()
	baseScopeWithSeparator := baseScope + ScopeSeparator
	if baseScope == RootScope {
		baseScopeWithSeparator = baseScope
	}
	for _, v := range ctx.data.variables {
		if v.Scope() == baseScope || strings.HasPrefix(v.Scope(), baseScopeWithSeparator) {
			fn(v)
		}
	}
}

// IterVariablesInScope is similar to IterVariables, but enumerate only those under the current
// context scope.
func (ctx *Context) IterVariablesInScope() iter.Seq[*Variable] {
	baseScope := ctx.Scope()
	return func(yield func(*Variable) bool) {
		baseScopeWithSeparator := baseScope + ScopeSeparator
		if baseScope == RootScope {
			baseScopeWithSeparator = baseScope
		}
		for _, v := range ctx.data.variables {
			if v.Scope() == baseScope || strings.HasPrefix(v.Scope(), baseScopeWithSeparator) {
				if !yield(v) {
					return
				}
			}
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
func (ctx *Context) Memory() uintptr {
	total := uintptr(0)
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

// SetLoader configures loader to be used as the default Loader for this Context.
//
// Loader is used just after any new variable is created, either with VariableWithValue or VariableWithShape.
// If the Loader has a value of the variable created, it will override the value given in VariableWithValue,
// or  skip the initializer for VariableWithShape.
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
		nodes, found := v.graphToNodes.Load(graphId)
		if !found {
			return
		}
		params[nodes.paramNode] = v.MustValue()
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

const GraphParamIsTraining = "training"

// IsTraining returns whether context is being used for training.
// This is only a convention adopted by the library components, and it is read
// with [Context.GetGraphParam] and [GraphParamIsTraining] for the current scope.
// See [SetTraining] to change this value.
//
// Notice that graph parameters are part of the "reference" component of a Context, so this change
// won't affect other connected context references.
func (ctx *Context) IsTraining(g *Graph) bool {
	return GetGraphParamOr(ctx, g, GraphParamIsTraining, false)
}

// SetTraining marks the context for the given graph as training.
// This is a convention adopted by the library components, and it simply sets it with
// [Context.SetGraphParam] and [GraphParamIsTraining] to the given value. See IsTraining to check for this value.
//
// Notice that the graph parameters is part of the "reference" component of a Context, so this change
// won't affect other connected context references.
func (ctx *Context) SetTraining(g *Graph, value bool) {
	ctx.SetGraphParam(g, GraphParamIsTraining, value)
}

// Finalize releases all variables and finalizes its values.
// Make sure to only call this is you are no longer using the context in any executor.
//
// After calling this, the context is left in an unusable state.
func (ctx *Context) Finalize() {
	for v := range ctx.IterVariables() {
		v.Finalize()
	}
	ctx.data.variables = nil
	ctx.data.variablesMap = nil
	ctx.data.needsInitialization = true
	ctx.data.loader = nil
}
