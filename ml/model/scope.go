// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding"
	"fmt"
	"iter"
	"path"
	"reflect"
	"strings"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/scoped"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/sets"
)

// JoinPath joins scopes (always with "/").
// It is platform-independent and always uses forward slashes, even on Windows.
// It also normalizes the path (handles "." and "..").
func JoinPath(elements ...string) string {
	return path.Join(elements...)
}

// SplitPath splits the path into its scope component and the last element name.
//
// Examples:
//
//	SplitPath("/a/b/c") -> ("/a/b", "c")
//	SplitPath("/a") -> ("/", "a")
//	SplitPath("") -> ("/", "")
//	SplitPath("/a/") -> ("/a", "")
//
// It is platform-independent and always uses forward slashes, even on Windows.
func SplitPath(fullPath string) (scope, name string) {
	scope, name = path.Split(fullPath)
	scope = strings.TrimSuffix(scope, ScopeSeparator)
	if scope == "" {
		scope = RootScopePath
	}
	return
}

// BasePath returns the last element of path.
// It is platform-independent and always uses forward slashes, even on Windows.
func BasePath(p string) string {
	return path.Base(p)
}

// Scope represents a "scoped" (a "current directory") view into a model's Store, which holds the scoped
// (in "directory" tree) variables and hyperparameters for the model.
//
// All types of information are organized in "scopes". The Scope object is actually a thin wrapper that
// contains the current scope (similar to a current directory) and a link to the actual data. One can change
// scopes by using Scope.In("new_scope"): it returns a new Scope with the new scope set, but still pointing
// (sharing) all the data with the previous Scope.
//
// Finally, Scope also allows one to checkpoint the variable values (save and load). See the checkpoint package.
type Scope struct {
	// scope for currently created variables and registration.
	// It is always an absolute path starting with "/".
	scope string

	// visitedPaths lists the sub-scopes already visited from the current scope. This
	// triggers an error (panic) if a re-entry happens on the same scope using In().
	// It is cleared when a new Scope is created via In() or Shared().
	visitedPaths sets.Set[string]

	// initializer is the default variable initializer for the current scope.
	// It's inherited when a new scope is created.
	initializer VariableInitializer

	store *Store
}

// Scope returns the full scope path.
func (s *Scope) Scope() string {
	return s.scope
}

// Store returns the Store associated with this Scope.
func (s *Scope) Store() *Store {
	return s.store
}

// copy creates a copy of the Scope, but sharing the same "data" component.
func (s *Scope) copy() *Scope {
	s2 := &Scope{}
	*s2 = *s
	return s2
}

// In returns a new "sub"-Scope of the current one with the given name.
//
// The nameFormat and its args are fed to fmt.Sprintf(), as a shortcut to allow formatted sub-scope names.
//
// This version panics if the sub-scope has not yet been visited.
// See also Scope.In and Scope.At.
func (s *Scope) In(nameFormat string, args ...any) *Scope {
	name := fmt.Sprintf(nameFormat, args...)
	if name == "" {
		Panicf("cannot use empty scope for Scope.In()")
	}
	s.checkName(name)

	if s.visitedPaths.Has(name) {
		Panicf("sub-scope %q already visited from %q, use Shared() if this is intended", name, s.scope)
	}
	s.visitedPaths.Insert(name)

	newScopePath := JoinPath(s.scope, name)
	s2 := s.copy()
	s2.scope = newScopePath
	s2.visitedPaths = sets.Make[string]()
	return s2
}

// Shared returns a new "sub"-Scope of the current one with the given name.
//
// The nameFormat and its args are fed to fmt.Sprintf(), as a shortcut to allow formatted sub-scope names.
//
// This version panics if the sub-scope has not yet been visited.
// See also Scope.In and Scope.At.
func (s *Scope) Shared(nameFormat string, args ...any) *Scope {
	name := fmt.Sprintf(nameFormat, args...)
	if name == "" {
		Panicf("cannot use empty scope for Scope.Shared()")
	}
	s.checkName(name)

	if !s.visitedPaths.Has(name) {
		Panicf("sub-scope %q not yet visited from %q, use In() first", name, s.scope)
	}

	newScopePath := JoinPath(s.scope, name)
	s2 := s.copy()
	s2.scope = newScopePath
	s2.visitedPaths = sets.Make[string]()
	return s2
}

// At returns a new "sub"-Scope of the current one with the given name.
//
// The nameFormat and its args are fed to fmt.Sprintf(), as a shortcut to allow formatted sub-scope names.
//
// This version works irrespective of whether the scope has been visited before or not.
// See also Scope.In and Scope.Shared.
func (s *Scope) At(nameFormat string, args ...any) *Scope {
	name := fmt.Sprintf(nameFormat, args...)
	if name == "" {
		Panicf("cannot use empty scope for Scope.At()")
	}
	s.checkName(name)

	newScopePath := JoinPath(s.scope, name)
	s2 := s.copy()
	s2.scope = newScopePath
	s2.visitedPaths = sets.Make[string]()
	return s2
}

// WithInitializer returns a new reference to the Scope, with the initializer set.
func (s *Scope) WithInitializer(initializer VariableInitializer) *Scope {
	if initializer == nil {
		Panicf("Scope.WithInitializer passed a nil initializer")
	}
	s2 := s.copy()
	s2.initializer = initializer
	return s2
}

// Initializer returns the initializer configured for the model.
func (s *Scope) Initializer() VariableInitializer {
	return s.initializer
}

// GetVariable returns the variable in the current scope.
// The name cannot contain the ScopeSeparator ("/").
func (s *Scope) GetVariable(name string) *Variable {
	s.checkName(name)
	return s.store.GetVariable(JoinPath(s.scope, name))
}

// DeleteVariable deletes the variable in the current scope.
// The name cannot contain the ScopeSeparator ("/").
func (s *Scope) DeleteVariable(name string) error {
	s.checkName(name)
	return s.store.DeleteVariable(JoinPath(s.scope, name))
}

// checkName panics if the name contains the ScopeSeparator ("/").
func (s *Scope) checkName(name string) {
	if strings.Contains(name, ScopeSeparator) {
		Panicf("name %q cannot contain ScopeSeparator %q, use Store methods for absolute paths", name, ScopeSeparator)
	}
}

// DeleteVariablesInScope deletes all variables under the current scope.
func (s *Scope) DeleteVariablesInScope() error {
	return s.store.DeleteVariablesInScope(s.scope)
}

// IterVariables returns an iterator that yields each variable in the model under the current scope (including sub-scopes).
func (s *Scope) IterVariables() iter.Seq[*Variable] {
	baseScope := s.scope
	baseScopeWithSeparator := baseScope
	if !strings.HasSuffix(baseScopeWithSeparator, "/") {
		baseScopeWithSeparator += "/"
	}
	return func(yield func(*Variable) bool) {
		for v := range s.store.IterVariables() {
			p := v.Path()
			if p == baseScope || strings.HasPrefix(p, baseScopeWithSeparator) {
				if !yield(v) {
					return
				}
			}
		}
	}
}

// IterParams returns an iterator that yields each parameter in the model under the current scope (including sub-scopes)
func (s *Scope) IterParams() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		baseScope := s.scope
		baseScopeWithSeparator := baseScope
		if !strings.HasSuffix(baseScopeWithSeparator, "/") {
			baseScopeWithSeparator += "/"
		}
		for paramValue := range s.store.params.Iter() {
			if paramValue.Scope == baseScope || strings.HasPrefix(paramValue.Scope, baseScopeWithSeparator) {
				fullPath := JoinPath(paramValue.Scope, paramValue.Key)
				if !yield(fullPath, paramValue.Value) {
					return
				}
			}
		}
	}
}

// IterGraphParams returns an iterator that yields each parameter in the model under the current scope (including sub-scopes) for the given graph.
func (s *Scope) IterGraphParams(g *Graph) iter.Seq[scoped.ParamValue] {
	baseScope := s.scope
	baseScopeWithSeparator := baseScope
	if !strings.HasSuffix(baseScopeWithSeparator, "/") {
		baseScopeWithSeparator += "/"
	}
	return func(yield func(scoped.ParamValue) bool) {
		for p := range s.store.IterGraphParams(g) {
			if p.Scope == baseScope || strings.HasPrefix(p.Scope, baseScopeWithSeparator) {
				if !yield(p) {
					return
				}
			}
		}
	}
}

// NumVariables return the number of variables in this Scope.
func (s *Scope) NumVariables() int {
	count := 0
	for range s.IterVariables() {
		count++
	}
	return count
}

// ByteSize returns the total number of bytes summed across all variables in this Scope.
func (s *Scope) ByteSize() int64 {
	total := int64(0)
	for v := range s.IterVariables() {
		total += v.Shape().ByteSize()
	}
	return total
}

// NumParameters returns the summed-up number of all variables in this Scope.
func (s *Scope) NumParameters() int {
	total := 0
	for v := range s.IterVariables() {
		total += v.Shape().Size()
	}
	return total
}

// GetParam returns the value for the given param key, searching successively from
// the current scope back to the root scope ("/").
func (s *Scope) GetParam(key string) (value any, found bool) {
	return s.store.params.Get(s.scope, key)
}

// MustGetParam is like GetParam, but panics if the parameter is not found, or if it is not of type T.
func MustGetParam[T any](s *Scope, key string) T {
	var t T
	valueAny, found := s.GetParam(key)
	if !found {
		Panicf("parameter %q (of type %T) not found in scope %q (and its parents)", key, t, s.scope)
	}

	v := reflect.ValueOf(valueAny)
	typeOfT := reflect.TypeOf(t)
	valueT := reflect.New(typeOfT)
	if valueT.Type().Implements(textUnmarshalerType) && v.Kind() == reflect.String {
		if err := valueT.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(v.String())); err != nil {
			Panicf("can't UnmarshalText %s to %s", v.String(), typeOfT.String())
		}
		return valueT.Elem().Interface().(T)
	} else if !v.CanConvert(typeOfT) {
		Panicf("MustGetParam[%T](s, %q): s(scope=%q)[%q]=(%T) %#v, and cannot be converted to %T",
			t, key, s.scope, key, valueAny, valueAny, t)
	}
	return v.Convert(typeOfT).Interface().(T)
}

// MustGetRootParam is like MustGetParam, but it uses the Store's root scope.
// It panics if the parameter is not found, or if it is not of type T.
func MustGetRootParam[T any](s *Store, key string) T {
	return MustGetParam[T](s.RootScope(), key)
}

// GetParamOr either returns the value for the given param key in the scope `s`,
// searching successively from the current scope back to the root scope ("/"), or if the
// key is not found or the key is set to nil, it returns the given default value.
func GetParamOr[T any](s *Scope, key string, defaultValue T) T {
	valueAny, found := s.GetParam(key)
	if !found || valueAny == nil {
		return defaultValue
	}
	value, ok := valueAny.(T)
	if ok {
		return value
	}
	return MustGetParam[T](s, key)
}

// GetRootParamOr works like GetParam but it assumes the root scope of the given Store.
func GetRootParamOr[T any](s *Store, key string, defaultValue T) T {
	return GetParamOr(s.RootScope(), key, defaultValue)
}

var textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()

// SetParam sets the given param in the current scope.
func (s *Scope) SetParam(key string, value any) {
	s.store.params.Set(s.scope, key, value)
}

// SetParams sets a collection of parameters in the current scope.
func (s *Scope) SetParams(keyValues map[string]any) {
	for key, value := range keyValues {
		s.store.params.Set(s.scope, key, value)
	}
}

// GetGraphParam returns the value for the given param key for the given graph,
// searching successively from the current scope back to the root scope ("/").
func (s *Scope) GetGraphParam(g *Graph, key string) (value any, found bool) {
	return GetGraphParam(g, JoinPath(s.scope, key))
}

// mustConvert any value into T. Panics if the value cannot be unmarshalled.
func mustConvert[T any](valueAny any, fullPath string) (T, error) {
	var t T
	v := reflect.ValueOf(valueAny)
	typeOfT := reflect.TypeOf(t)
	valueT := reflect.New(typeOfT)
	if valueT.Type().Implements(textUnmarshalerType) && v.Kind() == reflect.String {
		if err := valueT.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(v.String())); err != nil {
			return t, fmt.Errorf("can't UnmarshalText %s to %s for parameter %q: %w", v.String(), typeOfT.String(), fullPath, err)
		}
		return valueT.Elem().Interface().(T), nil
	} else if !v.CanConvert(typeOfT) {
		return t, fmt.Errorf("value of param %q (%#v) cannot be converted to %T", fullPath, valueAny, t)
	}
	return v.Convert(typeOfT).Interface().(T), nil
}

// GetGraphParamOr either returns the value for the given param key for the given graph,
// or if the key is not found, or the value is set to nil, it returns the given default value.
//
// It panics if the value is not of type T (as it tries to cast to T)
func GetGraphParamOr[T any](s *Scope, g *Graph, key string, defaultValue T) T {
	fullPath := JoinPath(s.scope, key)
	valueAny, found := GetGraphParam(g, fullPath)
	if !found || valueAny == nil {
		return defaultValue
	}
	value, ok := valueAny.(T)
	if !ok {
		return value
	}
	value, err := mustConvert[T](valueAny, fullPath)
	if err != nil {
		panic(err)
	}
	return value
}

// SetGraphParam sets the given Graph param in the current scope.
func (s *Scope) SetGraphParam(g *Graph, key string, value any) {
	SetGraphParam(g, JoinPath(s.scope, key), value)
}

// VariableWithShape creates or returns an existing variable with the given shape in the current scope.
func (s *Scope) VariableWithShape(name string, shape shapes.Shape) *Variable {
	s.checkName(name)
	fullPath := JoinPath(s.scope, name)
	return s.store.VariableWithShape(fullPath, shape, s.initializer)
}

// VariableWithValue creates or returns a variable initialized with the given value in the current scope.
func (s *Scope) VariableWithValue(name string, defaultValue any) *Variable {
	s.checkName(name)
	fullPath := JoinPath(s.scope, name)
	return s.store.VariableWithValue(fullPath, defaultValue)
}

// VariableWithNodeValue creates a variable in the current scope and sets it with graph computed *Node.
//
// This should only be used during graph/model building function.
func (s *Scope) VariableWithNodeValue(name string, value *Node) *Variable {
	s.checkName(name)
	fullPath := JoinPath(s.scope, name)
	return s.store.VariableWithNodeValue(fullPath, value)
}

// IsTraining returns whether current Store (and thus this Scope) is being used for training.
func (s *Scope) IsTraining(g *Graph) bool {
	return s.Store().IsTraining(g)
}

// BuildTrainableVariablesGradientsGraph returns the gradient of the loss with respect to each trainable variable
// in the scope (and sub-scopes) that was used in the current graph.
//
// Variables not marked as trainable are skipped.
func (s *Scope) BuildTrainableVariablesGradientsGraph(loss *Node) []*Node {
	g := loss.Graph()
	var trainableVars []*Node
	for v := range s.IterVariables() {
		if v.Trainable && v.InUseByGraph(g) {
			trainableVars = append(trainableVars, v.NodeValue(g))
		}
	}
	return graph.Gradient(loss, trainableVars...)
}
