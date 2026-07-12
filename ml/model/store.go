// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"iter"
	"slices"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/scoped"
	"github.com/gomlx/gomlx/support/exceptions"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
)

// Store stores variables and hyperparameters for the model.
// Both (variables and hyperparameters) are "scoped", organized in a tree, like directories
// separated by "/".
//
// Generally, a user only interacts with a Scope objects (works as a "current directory"),
// which translates its operations (like creating a new variable) into the Store.
//
// But if operating directly on the full Store (accessing a variable by its absolute path, iterating over
// all variables and parameters), one may use the Store directly. One can always access the Store from
// a Scope object using Scope.Store()
type Store struct {
	// params holds a model's building (hyper)parameters.
	params *scoped.Params

	// variablesMap for this store organized per absolute path.
	// The key is the full path of the variable.
	variablesMap map[string]*Variable

	// variables is a plain list of all variables, in creation order.
	variables []*Variable

	// loader, if set, is called to check whether there is a previous value of the variable to use.
	loader Loader

	// needsInitialization indicates whether there are uninitialized variables in
	// the model.
	needsInitialization bool

	// defaultShardingSpec used for new variables, if execution is distributed.
	defaultShardingSpec *distributed.ShardingSpec
}

// Loader can be implemented by any library providing loading of variables for
// Store. Loader implementations need to provide values on demand.
type Loader interface {
	// LoadVariable tries to load the variable indicated by its full path.
	// If it's not found, returns false, and initialization continues as usual.
	LoadVariable(store *Store, fullPath string) (value *tensors.Tensor, found bool)

	// DeleteVariable is called whenever Store.DeleteVariable is called.
	DeleteVariable(store *Store, fullPath string) error
}

// NewStore returns a new empty Store.
func NewStore() *Store {
	return &Store{
		params:       scoped.New(ScopeSeparator),
		variablesMap: make(map[string]*Variable),
	}
}

// StoreProvider is an interface for types that provide access to a Store.
// It works as an meta-type that accepts either a Scope or a Store object.
type StoreProvider interface {
	Store() *Store
}

// Store implements the StoreProvider interface.
func (s *Store) Store() *Store {
	return s
}

// Scope returns a new Scope for the given absolute path.
func (s *Store) Scope(fullPath string) *Scope {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	return &Scope{
		scope:        fullPath,
		store:        s,
		initializer:  DefaultInitializer(s),
		visitedPaths: sets.Make[string](),
	}
}

// RootScope returns a new Scope for the given Store.
func (s *Store) RootScope() *Scope {
	return s.Scope("/")
}

// Clone does a (mostly) deep copy of the Store, with new variable values, and a clone of the parameters.
func (s *Store) Clone() (*Store, error) {
	newStore := NewStore()
	newStore.needsInitialization = s.needsInitialization
	newStore.params = s.params.Clone()
	newStore.defaultShardingSpec = s.defaultShardingSpec
	newStore.loader = s.loader

	for _, v := range s.variables {
		_, err := v.CloneToStore(newStore)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to clone variable %q while cloning the Store", v.Path())
		}
	}
	return newStore, nil
}

const (
	// ScopeSeparator is used between levels of scope. Scope names cannot use this character.
	ScopeSeparator = "/"

	// RootScopePath is the scope at the very root.
	RootScopePath = "/"
)

// GetVariable returns the variable associated with its fullPath.
// It returns nil if a variable with the given fullPath hasn't been created.
//
// This will trigger the loading of the variable if a loader is attached.
func (s *Store) GetVariable(fullPath string) *Variable {
	if s == nil {
		exceptions.Panicf("Store is nil")
	}
	if s.variablesMap == nil {
		exceptions.Panicf("Store.variablesMap is nil")
	}
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	v, ok := s.variablesMap[fullPath]
	if ok {
		return v
	}

	// Try to load it, if a loader is configured.
	if s.loader == nil {
		return nil
	}
	value, found := s.loader.LoadVariable(s, fullPath)
	if !found {
		return nil
	}

	scopePath, name := SplitPath(fullPath)
	if scopePath == "" {
		scopePath = "/"
	} else if len(scopePath) > 1 && strings.HasSuffix(scopePath, "/") {
		scopePath = scopePath[:len(scopePath)-1]
	}

	v = &Variable{
		store:        s,
		name:         name,
		fullPath:     fullPath,
		shape:        value.Shape(),
		value:        value,
		Trainable:    true,
		shardingSpec: s.defaultShardingSpec,
	}
	s.setVariable(v)
	return v
}

// setVariable adds the variable to the store.
func (s *Store) setVariable(v *Variable) {
	s.variablesMap[v.fullPath] = v
	s.variables = append(s.variables, v)
}

// DeleteVariable if it exists.
func (s *Store) DeleteVariable(fullPath string) error {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	// Even if variable doesn't exist in store yet, we need to remove it from the loader.
	if s.loader != nil {
		err := s.loader.DeleteVariable(s, fullPath)
		if err != nil {
			return err
		}
	}
	v, ok := s.variablesMap[fullPath]
	if !ok {
		return nil
	}
	_ = v.Reset()
	delete(s.variablesMap, fullPath)
	s.variables = slices.DeleteFunc(s.variables, func(vCandidate *Variable) bool {
		return vCandidate == v
	})
	return nil
}

// MoveVariable moves a variable from one absolute path to another.
//
// This cannot be called from a graph-building function, as it will invalidate any existing executor that uses this
// variable. Any executor using the variable on the older path must be discarded and recreated.
func (s *Store) MoveVariable(fromPath, toPath string) error {
	if !strings.HasPrefix(fromPath, "/") {
		fromPath = "/" + fromPath
	}
	if !strings.HasPrefix(toPath, "/") {
		toPath = "/" + toPath
	}
	v, ok := s.variablesMap[fromPath]
	if !ok {
		return errors.Errorf("variable %q not found in store", fromPath)
	}
	if _, exists := s.variablesMap[toPath]; exists {
		return errors.Errorf("variable at target path %q already exists in store", toPath)
	}

	parts := strings.Split(toPath, "/")
	name := parts[len(parts)-1]

	delete(s.variablesMap, fromPath)
	s.variablesMap[toPath] = v
	v.name = name
	v.fullPath = toPath

	return nil
}

// DeleteVariablesInScope deletes all variables under the given absolute path.
func (s *Store) DeleteVariablesInScope(fullPath string) error {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	fullPathWithSeparator := fullPath
	if !strings.HasSuffix(fullPathWithSeparator, "/") {
		fullPathWithSeparator += "/"
	}

	var toDelete []string
	for p, v := range s.variablesMap {
		if p == fullPath || strings.HasPrefix(p, fullPathWithSeparator) {
			toDelete = append(toDelete, p)
			_ = v.Reset()
		}
	}

	for _, p := range toDelete {
		delete(s.variablesMap, p)
		if s.loader != nil {
			_ = s.loader.DeleteVariable(s, p)
		}
	}

	s.variables = slices.DeleteFunc(s.variables, func(v *Variable) bool {
		p := v.Path()
		return p == fullPath || strings.HasPrefix(p, fullPathWithSeparator)
	})

	return nil
}

// IterVariables returns an iterator that yields each variable in the store.
func (s *Store) IterVariables() iter.Seq[*Variable] {
	return func(yield func(*Variable) bool) {
		for _, v := range s.variables {
			if !yield(v) {
				return
			}
		}
	}
}

// IterParams returns an iterator that yields each parameter in the store.
func (s *Store) IterParams() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for paramValue := range s.params.Iter() {
			fullPath := JoinPath(paramValue.Scope, paramValue.Key)
			if !yield(fullPath, paramValue.Value) {
				return
			}
		}
	}
}

// IterGraphParams returns an iterator that yields each parameter in the store for the given graph.
func (s *Store) IterGraphParams(g *Graph) iter.Seq[scoped.ParamValue] {
	gs := getGraphStore(g)
	if gs == nil {
		return func(yield func(scoped.ParamValue) bool) {}
	}
	return gs.params.Iter()
}

// NumVariables return the number of variables in this Store.
func (s *Store) NumVariables() int {
	return len(s.variables)
}

// ByteSize returns the total number of bytes summed across all variables.
func (s *Store) ByteSize() int64 {
	total := int64(0)
	for _, v := range s.variables {
		total += v.Shape().ByteSize()
	}
	return total
}

// NumParameters returns the summed-up number of all variables.
func (s *Store) NumParameters() int {
	total := 0
	for _, v := range s.variables {
		total += v.Shape().Size()
	}
	return total
}

// InitializeVariables initializes all variables in the Store that don't yet have a value.
func (s *Store) InitializeVariables(backend compute.Backend, configExec func(initializerExec *Exec) error) error {
	// Collect variables that need initialization.
	var variablesToInitialize []*Variable
	for _, v := range s.variables {
		if !v.HasValue() {
			variablesToInitialize = append(variablesToInitialize, v)
		}
	}
	if len(variablesToInitialize) == 0 {
		s.needsInitialization = false
		return nil
	}

	// Execute initialization for collected variables.
	e, err := NewExec(backend, s, func(scope *Scope, g *graph.Graph) []*Node {
		g = g.WithName("VariableInitialization")
		initialValues := make([]*Node, 0, len(variablesToInitialize))
		for _, v := range variablesToInitialize {
			if v.initializer == nil {
				Panicf("failed to initialize variable %q: initializer was not configured", v.Path())
			}
			initialValues = append(initialValues, v.initializer(g, v.shape))
		}
		return initialValues
	})
	if err != nil {
		return errors.WithMessagef(err, "failed to create executor for variable initialization")
	}
	if configExec != nil {
		err := configExec(e)
		if err != nil {
			return errors.WithMessagef(err, "failed to configure executor for variable initialization")
		}
	}
	e.isInitializeVariablesExec = true
	values, err := e.Call()
	if err != nil {
		return errors.WithMessagef(err, "failed to compile/run variable initialization graph")
	}
	numDevices := e.NumDevices()
	if len(values) != numDevices*len(variablesToInitialize) {
		return errors.Errorf("failed to initialize variables: expected numDevices(%d) * %d values, got %d",
			numDevices, len(variablesToInitialize), len(values))
	}
	for ii, v := range variablesToInitialize {
		if !values[ii].Ok() {
			return errors.Errorf("graph execution to initialize variables failed: variable %q generated value was invalid", v.Path())
		}
		v.value = values[ii]
	}
	s.needsInitialization = false
	return nil
}

// Loader returns the current configured Loader for this Store.
func (s *Store) Loader() Loader {
	return s.loader
}

// SetLoader configures loader to be used as the default Loader for this Store.
func (s *Store) SetLoader(loader Loader) {
	s.loader = loader
}

// Finalize releases all variables and finalizes its values.
func (s *Store) Finalize() {
	for _, v := range s.variables {
		v.Finalize()
	}
	s.variables = nil
	s.variablesMap = nil
	s.needsInitialization = true
	s.loader = nil
}

// GetParam returns the value for the given param key, searching successively from
// the given absolute path back to the root scope ("/").
func (s *Store) GetParam(fullPath string) (value any, found bool) {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	scopePath, baseName := SplitPath(fullPath)
	return s.params.Get(scopePath, baseName)
}

// SetParam sets the given param in the given absolute path.
func (s *Store) SetParam(fullPath string, value any) {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	scopePath, baseName := SplitPath(fullPath)
	s.params.Set(scopePath, baseName, value)
}

// SetParams sets a collection of parameters in the store.
// The key is the full path, value is the value.
func (s *Store) SetParams(keyValues map[string]any) {
	for fullPath, value := range keyValues {
		s.SetParam(fullPath, value)
	}
}

// storeLink is a key used to store a pointer to a graphStore in a graph's state.
// This allows the graphStore to be retrieved from the graph.
type storeLink struct{}

// GetStore returns the current model.Store associated with the graph (if any, otherwise nil).
//
// The [model.Exec] executor will automatically set it for you.
// Alternatively, it can be set using [SetGraphStore].
func GetStore(g graph.GraphProvider) *Store {
	if g == nil {
		return nil
	}
	gr := g.Graph()
	if gr == nil {
		return nil
	}
	state := gr.State(storeLink{})
	if state == nil {
		return nil
	}
	return state.(*Store)
}

// SetStore associated with a graph.
// One doesn't need to do this usually, as it is taken care by the [model.Exec] executor.
// But if you are building the graph "manually", you can use this.
func SetStore(g graph.GraphProvider, store *Store) {
	g.Graph().AttachState(storeLink{}, store)
}

// graphState holds the state associate with this specific graph, in the form a `graphStore`.
type graphState struct{}

// graphStore is the [Store] information associated specifically to this graph (and that gets
// automatically destroyed when the graph is destroyed).
type graphStore struct {
	mu        sync.RWMutex
	params    *scoped.Params
	variables map[string]*variableNodes
}

func (gs *graphStore) getVariableNodes(v *Variable) (*variableNodes, bool) {
	if gs == nil {
		return nil, false
	}
	gs.mu.RLock()
	defer gs.mu.RUnlock()
	nodes, found := gs.variables[v.Path()]
	return nodes, found
}

func (gs *graphStore) setVariableNodes(v *Variable, nodes *variableNodes) {
	if gs == nil {
		return
	}
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.variables[v.Path()] = nodes
}

// IterVariables returns an iterator over the variables registered in the graphStore.
func (gs *graphStore) IterVariables() iter.Seq2[string, *variableNodes] {
	return func(yield func(string, *variableNodes) bool) {
		if gs == nil {
			return
		}
		gs.mu.RLock()
		defer gs.mu.RUnlock()
		for path, nodes := range gs.variables {
			if !yield(path, nodes) {
				return
			}
		}
	}
}

func newGraphStore() *graphStore {
	return &graphStore{
		params:    scoped.New(ScopeSeparator),
		variables: make(map[string]*variableNodes),
	}
}

// getGraphStore returns the graphStore attached to the graph.
// If none exists, it automatically creates and attaches one.
func getGraphStore(g *Graph) *graphStore {
	state := g.State(graphState{})
	if state == nil {
		gs := newGraphStore()
		g.AttachState(graphState{}, gs)
		return gs
	}
	return state.(*graphStore)
}

// setGraphStore attaches a graphStore to the graph.
func setGraphStore(g *Graph, gs *graphStore) {
	g.AttachState(graphState{}, gs)
}

// GetGraphParam returns the value for the given param key for the given graph,
// searching successively from the current scope back to the root scope ("/").
func GetGraphParam(g graph.GraphProvider, fullPath string) (value any, found bool) {
	gs := getGraphStore(g.Graph())
	if gs == nil {
		return
	}
	scope, key := SplitPath(fullPath)
	return gs.params.Get(scope, key)
}

// SetGraphParam sets the given Graph param in the current scope.
func SetGraphParam(g *Graph, fullPath string, value any) {
	gs := getGraphStore(g)
	if gs == nil {
		gs = newGraphStore()
		setGraphStore(g, gs)
	}
	scopePath, baseName := SplitPath(fullPath)
	gs.params.Set(scopePath, baseName, value)
}

// GraphParamIsTraining is the name of a graph param that indicates whether is current graph is being
// used to train models.
const GraphParamIsTraining = "training"

// IsTraining returns whether current Store (and thus this Scope) is being used for training.
func (s *Store) IsTraining(g *Graph) bool {
	return GetGraphParamOr(s.RootScope(), g, GraphParamIsTraining, false)
}

// SetTraining marks the current Store (and thus this Scope) for the given graph as training.
func (s *Store) SetTraining(g *Graph, value bool) {
	s.RootScope().SetGraphParam(g, GraphParamIsTraining, value)
}

// SetParams sets a collection of parameters in the current scope.
func (s *Store) SetParamsInScope(pathValues map[string]any) {
	for fullPath, value := range pathValues {
		s.SetParam(fullPath, value)
	}
}

// EscapeScopeName replaces ScopeSeparator in the string and replaces them by "_".
func EscapeScopeName(scopeName string) string {
	return strings.ReplaceAll(scopeName, ScopeSeparator, "_")
}

// VariableWithShape creates or returns an existing variable with the given shape.
func (s *Store) VariableWithShape(fullPath string, shape shapes.Shape, initializer VariableInitializer) *Variable {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	v := s.GetVariable(fullPath)
	if v != nil {
		if !shape.Equal(v.shape) {
			Panicf(
				"requested to reuse variable %q, but with different shape from original: previous shape=%s, requested shape=%s",
				fullPath,
				v.shape,
				shape,
			)
		}
		v.initializer = initializer
		return v
	}

	_, name := SplitPath(fullPath)
	v = &Variable{
		store:        s,
		name:         name,
		fullPath:     fullPath,
		shape:        shape,
		Trainable:    true,
		shardingSpec: s.defaultShardingSpec,
	}
	s.setVariable(v)
	v.initializer = initializer
	s.needsInitialization = true
	return v
}

// VariableWithNodeValue creates a variable with the path and [graph.Node] value given.
// The value is immediately available in the graph (see Variable.NodeValue()), but
// it will only have a concrete value once the model is executed (see Exec.Call).
//
// This should only be used during graph/model building function.
func (s *Store) VariableWithNodeValue(fullPath string, value *Node) *Variable {
	v := s.VariableWithShape(fullPath, value.Shape(), graph.Zeros)
	v.SetNodeValue(value)
	return v
}

// VariableWithValue creates or returns a variable initialized with the given value in the given variable path.
func (s *Store) VariableWithValue(fullPath string, defaultValue any) *Variable {
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	v := s.GetVariable(fullPath)
	valueT, err := valueToTensor(defaultValue)
	if err != nil {
		panic(errors.WithMessagef(err, "failed to parse value for variable %q", fullPath))
	}

	if v != nil {
		if !valueT.Shape().Equal(v.shape) {
			Panicf(
				"requested to reuse variable %q, but with defaultValue with different shape from original: previous shape=%s, requested defaultValue shape=%s",
				fullPath,
				v.shape,
				valueT.Shape(),
			)
		}
		return v
	}

	v = &Variable{
		store:     s,
		fullPath:  fullPath,
		shape:     valueT.Shape(),
		value:     valueT,
		Trainable: true,
	}
	_, v.name = SplitPath(fullPath)
	s.setVariable(v)
	return v
}
