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

package graph

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"log"
	"reflect"
	"runtime"
	"sync"
)

// ExecGraphFn is a type parameter for accepted function types for NewExec constructor.
type ExecGraphFn interface {
	func(*Graph) *Node |
		func(*Node) *Node |
		func(*Node, *Node) *Node |
		func(*Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node, *Node) *Node |
		func([]*Node) *Node |

		// With 2 outputs
		func(*Graph) (*Node, *Node) |
		func(*Node) (*Node, *Node) |
		func(*Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node, *Node) (*Node, *Node) |
		func([]*Node) (*Node, *Node) |

		// With 3 outputs
		func(*Graph) (*Node, *Node, *Node) |
		func(*Node) (*Node, *Node, *Node) |
		func(*Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func([]*Node) (*Node, *Node, *Node) |

		// With slice of nodes as output.
		func(*Graph) []*Node |
		func(*Node) []*Node |
		func(*Node, *Node) []*Node |
		func(*Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node, *Node, *Node) []*Node |
		func([]*Node) []*Node
}

// SideParamsFn is the functions that sets side parameters during execution
// for Graphs that defines those. Typically, this is used to set the variables.
type SideParamsFn func(graph *Graph, params []*tensor.Device)

// LoggerFn is the function used to log nodes marked for logging. It is called
// after the Call method, with the list of messages and corresponding values
// of the evaluated nodes.
type LoggerFn func(messages []string, values []tensor.Tensor)

// Exec creates and executes computation graphs as needed
// based on the inputs shapes.
//
// It simplifies the process of executing a graph building
// function with real values. For example, assume you wrote:
//
//	def LengthGraph(x *Node) *Node {
//	  return Sqrt(ReduceAllSum(Mul(x, x)))
//	}
//
// To actually use it with real values, one need to build
// the graph to a specific shape of x, and then execute it,
// which is not straight forward -- JIT compilation makes things
// faster, but it imposes some bureaucracy.
//
// With Exec one can do:
//
//	var Length = NewExec(LengthGraph)
//	x0 := []float32{4}
//	fmt.Printf("Length(%v) = %v\n", x0, Length.Call(x0)[0].Value())
//	x1 := []float64{1, 2, 3}
//	fmt.Printf("Length(%v) = %v\n", x1, Length.Call(x1)[0].Value())
//
// Notice that both calls to Length.Call will need to create different
// graphs (for different shapes of the input), but they will be cached,
// and if the same shapes are used in Call again, the cached compiled graph
// is reused.
//
// Also Call outputs a slice with all the outputs, even when there is only
// one output.
//
// If there are no inputs (for instance for some initialization function), then
// one needs to take a *Graph as the first parameter of graphFn. Example:
//
//	```
//	iotaMatrixExec := NewExec(func (g *Graph) *Node {
//		return IotaFull(g, types.Make(types.Float32, 3, 3))
//	})
//	fmt.Printf("IotaFull(3x3 matrix, float32)=%v\n", iotaMatrixExec.Call()[0].Value().([][]float32))
//	```
//
// The need to build different graphs for different shapes can be expensive
// when sizes of the inputs varies a lot. The usual solution is to use shapes
// with size in a power scale (for instance powers of 2) and masking of tensors
// for unused slices. For safety concerns there are a maximum number of different
// instantiations of the graph. It can be set or disabled with SetMaxCache.
//
// Errors are returned inside the returned tensors.
//
// There is concurrency safety with the cache, but XLA concurrency is
// not documented. TODO: figure it out.
type Exec struct {
	manager   *Manager
	deviceNum int

	graphFn                     any
	numInputs, numOutputs       int
	inputAsSlice, outputAsSlice bool
	inputIsGraph                bool
	name                        string

	// MaxCacheSize: if more than these different graph instantiations are
	// created, Exec starts returning errors in Call.
	maxCacheSize int

	// setSideParams for graphs that take them.
	setSideParams SideParamsFn
	loggerFn      LoggerFn

	// Protects cache structure.
	cacheMu sync.Mutex
	cache   []*execCacheEntry
}

// execCacheEntry: no hashing, just a simple list. This is faster
// for smaller tables. TODO: add a hashtable for cases with large caches.
type execCacheEntry struct {
	argsShapes     []shapes.Shape
	graph          *Graph
	numOutputs     int      // Number of flattened outputs for this graph, including logged nodes.
	loggedMessages []string // Messages for logged nodes.
}

const DefaultExecMaxCacheSize = 10

// NewExecAny constructs an Exec object that uses the given graphFn to build
// computation graphs. graphFn take only *Node parameters as input and
// return one or more *Node. Except if there are no inputs, in which case graphFn
// needs to take a *Graph as the first parameter.
//
// If any input or output parameter of graphFn is not a *Node (or *Graph is there are no inputs),
// or if there are no inputs or outputs, it returns an error.
func NewExecAny(manager *Manager, graphFn any) (*Exec, error) {
	graphFnT := reflect.TypeOf(graphFn)
	funcName := runtime.FuncForPC(reflect.ValueOf(graphFn).Pointer()).Name()
	exec := &Exec{
		manager:   manager,
		name:      fmt.Sprintf("Exec:%s", funcName),
		deviceNum: manager.DefaultDeviceNum(),

		graphFn:      graphFn,
		numInputs:    graphFnT.NumIn(),
		numOutputs:   graphFnT.NumOut(),
		maxCacheSize: DefaultExecMaxCacheSize,
		loggerFn:     defaultNodeLogger,
	}

	// Verify parameters.
	if graphFnT.Kind() != reflect.Func {
		return nil, errors.Errorf("graphFn must be a function")
	}

	var node *Node
	nodeType := reflect.TypeOf(node)
	var tmpGraph *Graph
	graphType := reflect.TypeOf(tmpGraph)

	if graphFnT.NumIn() < 1 || graphFnT.NumOut() < 1 {
		// It requires at least one input and one output.
		return nil, errors.Errorf("not enough input (%d)/output (%d) parameters, both need to be > 0",
			graphFnT.NumIn(), graphFnT.NumOut())
	}
	for ii := 0; ii < graphFnT.NumIn(); ii++ {
		if graphFnT.In(ii).Kind() == reflect.Slice && graphFnT.In(ii).Elem() == nodeType {
			if graphFnT.NumIn() != 1 {
				return nil, errors.Errorf("[]*Node parameters are only accepted as input if they are the only input, got function type %s instead", graphFnT)
			}
			exec.inputAsSlice = true
			break
		}
		if graphFnT.In(ii) == graphType {
			if graphFnT.NumIn() != 1 {
				return nil, errors.Errorf("*Graph parameter only accepted as input if they are the only input, got function type %s instead", graphFnT)
			}
			exec.inputIsGraph = true
			exec.numInputs = 0
			break
		}
		if graphFnT.In(ii) != nodeType {
			return nil, errors.Errorf("input parameter %d is not of type *Node or []*Node", ii)
		}
	}
	for ii := 0; ii < graphFnT.NumOut(); ii++ {
		if graphFnT.Out(ii).Kind() == reflect.Slice && graphFnT.Out(ii).Elem() == nodeType {
			if graphFnT.NumOut() != 1 {
				return nil, errors.Errorf("[]*Node parameters are only accepted as output if they are the only output, got function type %s instead", graphFnT)
			}
			exec.outputAsSlice = true
			break
		}
		if graphFnT.Out(ii) != nodeType {
			return nil, errors.Errorf("output parameter %d is not of type *Node", ii)
		}
	}
	return exec, nil
}

// NewExec constructs an Exec object that uses the given graphFn to build
// computation graphs. graphFn should take *Node as input and return a *Node.
// It's a wrapper for NewExecAny, but uses generics to type check that
// graphFn is valid.
func NewExec[F ExecGraphFn](manager *Manager, graphFn F) *Exec {
	e, err := NewExecAny(manager, graphFn)
	if err != nil {
		// This shouldn't happen for known types.
		log.Panicf("Invalid graphFn of type %T, resulted in error: %+v", graphFn, err)
	}
	return e
}

// InDevice sets the device num to be used by graphs constructed by Exec.
// This should be called before any invocations of Call().
// It returns a reference to itself so calls can be cascaded.
func (e *Exec) InDevice(deviceNum int) *Exec {
	e.deviceNum = deviceNum
	return e
}

// SetName sets the name of Exec, used to provide the name to graphs created.
// This should be called before any invocations of Call().
// It returns a reference to itself so calls can be cascaded.
func (e *Exec) SetName(name string) *Exec {
	e.name = name
	return e
}

// Name returns the Exec name, a string used as prefix for Graph construction.
func (e *Exec) Name() string {
	return e.name
}

// SetMaxCache sets the maximum size of the cache.
// Set it to -1 to have unlimited cache size.
// It returns a reference to itself so calls can be cascaded.
func (e *Exec) SetMaxCache(maxCacheSize int) *Exec {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()
	e.maxCacheSize = maxCacheSize
	return e
}

// SetSideParamsHook makes Exec call the given function everytime
// before executing a graph with the list of parameters.
//
// Side parameters are parameters created by the graphFn itself,
// and are not passed to it as input parameters. These could
// be variables in a model, or some global values. Exec
// has no knowledge of them, hence cannot set their values,
// and this serves as a hook to set them up just before
// the graph is executed.
//
// The function is called anyway, even if there are no
// side parameters to be set, so it can be used as a hook
// just before graph execution.
//
// SideParamsFn is a function that takes as input a slice
// of Device tensors that will be passed as input to graph
// execution. The first elements of the slice are the
// input parameters to graphFn function (given during
// the construction of Exec), and they will be filled
// already with the correct values.
func (e *Exec) SetSideParamsHook(fn SideParamsFn) *Exec {
	e.setSideParams = fn
	return e
}

// SetNodeLogger with the function to be called for the nodes
// marked for logging during execution. If set to nil
// nothing will be logged.
func (e *Exec) SetNodeLogger(loggerFn LoggerFn) {
	e.loggerFn = loggerFn
}

// GetNodeLogger returns the currently registered LoggerFn.
func (e *Exec) GetNodeLogger() LoggerFn {
	return e.loggerFn
}

// errorResult creates a device tensor with the given error and replicate it
// for every output.
func (e *Exec) errorResult(err error) []tensor.Tensor {
	t := tensor.DeviceWithError(err)
	res := make([]tensor.Tensor, e.numOutputs)
	for ii := range res {
		res[ii] = t
	}
	return res
}

// Call parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments. If a graph does
// not yet exist one is created, compiled and cached for the shapes.
// It returns the outputs in a slice, even if there is only one output.
func (e *Exec) Call(args ...any) []tensor.Tensor {
	results, _ := e.CallWithGraph(args...)
	return results
}

// CallWithGraph is similar to Call, but it also returns the computation graph used
// in the call. Since Exec creates different computation graphs for different set of
// parameters, this can help disambiguate in case the user needs to use the Graph for
// something else.
//
// Notice the returned *Graph may be nil, if it failed to parse the arguments and find
// the corresponding computation graph.
func (e *Exec) CallWithGraph(args ...any) ([]tensor.Tensor, *Graph) {
	if !e.inputAsSlice && len(args) != e.numInputs {
		return e.errorResult(
			errors.Errorf(
				"# of arguments to call (%d) don't match # arguments to graph function (%d) for %q",
				len(args), e.numInputs, e.Name())), nil
	}

	// Convert args to tensors.
	argsShapes := make([]shapes.Shape, 0, len(args))
	tensors := make([]*tensor.Device, 0, len(args)) // There may be more parameters, set with Exec.setSideParams later.
	for ii := range args {
		deviceT := anyToDeviceTensor(e.manager, e.deviceNum, args[ii])
		if !deviceT.Ok() {
			return e.errorResult(deviceT.Error()), nil
		}
		tensors = append(tensors, deviceT)
		argsShapes = append(argsShapes, deviceT.Shape())
	}

	// Get or build the graph.
	entry := e.findCacheEntry(argsShapes)
	if entry == nil {
		return e.errorResult(errors.Errorf(
			"maximum cache size reached for %q, cannot create another graph -- "+
				"a new computation graph needs to be created+compiled for each different shape of "+
				"the input, consider using padding, or if this is not a concern change "+
				"the cache size with exec.SetMaxCache()", e.Name())), nil
	}
	graph := entry.graph
	if !graph.Ok() {
		return e.errorResult(errors.WithMessagef(graph.Error(), "failed to build %q computation graph", e.Name())), graph
	}

	// Set extra input parameters created by the graph.
	if graph.NumParameters() > len(args) {
		tmp := make([]*tensor.Device, graph.NumParameters())
		copy(tmp, tensors)
		tensors = tmp
	}
	if e.setSideParams != nil {
		e.setSideParams(graph, tensors)
	}
	if graph.NumParameters() > len(args) {
		for ii, t := range tensors {
			if t == nil || !t.Ok() {
				return e.errorResult(errors.Errorf("parameter %d (%q) is nil or invalid, maybe a variable value not set as a parameter, cannot execute graph",
					ii, graph.ParameterByIndex(ii).ParameterName())), graph
			}
		}
	}

	// Execute graph.
	outputT := graph.RunWithTensors(tensors)
	if !outputT.Ok() {
		return e.errorResult(errors.WithMessagef(outputT.Error(), "failed to execute graph")), graph
	}
	if entry.numOutputs == 1 {
		return []tensor.Tensor{outputT}, graph
	}
	outputsDevice := outputT.SplitTuple()
	outputs := make([]tensor.Tensor, len(outputsDevice))
	for ii := range outputsDevice {
		outputs[ii] = outputsDevice[ii]
	}
	outputT.Finalize()

	// Call logger on logged nodes, even if no node is marked for logging (it serves as a hook).
	numGraphFnOutputs := entry.numOutputs - len(entry.loggedMessages)
	if e.loggerFn != nil {
		e.loggerFn(entry.loggedMessages, outputs[numGraphFnOutputs:])
	}
	return outputs[:numGraphFnOutputs], graph
}

// createAndCacheGraph creates and compiles the graph for the arguments with the given
// shapes. It creates and stores a cache entry for it and returns it.
// Returns nil if the cache size is >= MaxCacheSize.
// Should be called with cacheMu locked.
func (e *Exec) createAndCacheGraph(argsShapes []shapes.Shape) (entry *execCacheEntry) {
	if len(e.cache) >= e.maxCacheSize {
		return nil
	}
	entry = &execCacheEntry{graph: e.manager.NewGraph(fmt.Sprintf("%s#%d", e.name, len(e.cache)))}
	g := entry.graph
	var argsV []reflect.Value
	var args []*Node
	if e.inputAsSlice {
		args = make([]*Node, 0, len(argsShapes))
	} else if e.inputIsGraph {
		// Notice in this case len(argsShapes) == 0
		argsV = []reflect.Value{reflect.ValueOf(g)}
	} else {
		argsV = make([]reflect.Value, 0, len(argsShapes))
	}
	for ii, shape := range argsShapes {
		arg := g.Parameter(fmt.Sprintf("arg#%d", ii), shape)
		if e.inputAsSlice {
			args = append(args, arg)
		} else {
			argsV = append(argsV, reflect.ValueOf(arg))
		}
	}
	graphFnV := reflect.ValueOf(e.graphFn)
	if e.inputAsSlice {
		// If input is a slice of *Node, take argsV to be one parameter, the value of the slice.
		argsV = []reflect.Value{reflect.ValueOf(args)}
	}

	// Enumerate outputs from wrapped graphFn.
	outputsV := graphFnV.Call(argsV)
	var outputs []*Node
	if e.outputAsSlice {
		if len(outputsV) != 1 {
			g.SetErrorf("graphFn for %q returned %d results, as opposed to simply a slice of nodes -- []*Node",
				e.Name(), len(outputsV))
			return entry
		}
		outputs = outputsV[0].Interface().([]*Node)
	} else {
		outputs = make([]*Node, 0, len(outputsV))
		for _, outV := range outputsV {
			outputs = append(outputs, outV.Interface().(*Node))
		}
	}

	// Append logged nodes as outputs.
	for _, node := range g.LoggedNodes() {
		outputs = append(outputs, node)
		entry.loggedMessages = append(entry.loggedMessages, node.LogMessage())
	}

	// Compile graph.
	if g.Ok() {
		g.Compile(outputs...)
	}
	entry.argsShapes = make([]shapes.Shape, len(argsShapes))
	copy(entry.argsShapes, argsShapes)
	entry.numOutputs = len(outputs)
	e.cache = append(e.cache, entry)
	return entry
}

// findCacheEntry returns the graph for the given arguments shapes, or nil
// if no cache entry exists.
func (e *Exec) findCacheEntry(argsShapes []shapes.Shape) *execCacheEntry {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()

LoopCache:
	for _, entry := range e.cache {
		if len(argsShapes) != len(entry.argsShapes) {
			continue
		}
		for ii, shape := range argsShapes {
			if !shape.Eq(entry.argsShapes[ii]) {
				continue LoopCache
			}
		}
		return entry
	}

	// No graph in cache, create a new one.
	return e.createAndCacheGraph(argsShapes)
}

// Finalize clears the cache, finalizing the graphs. The Exec object shouldn't be
// used after that.
func (e *Exec) Finalize() {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()

	for _, entry := range e.cache {
		entry.graph.Finalize()
		entry.graph = nil
	}
	e.cache = e.cache[:0]
}

// defaultNodeLogger for nodes marked to be logged. It prints the message and
// the node value.
func defaultNodeLogger(messages []string, values []tensor.Tensor) {
	if len(messages) == 0 {
		return
	}
	fmt.Printf("defaultNodeLogger():\n")
	for ii, msg := range messages {
		fmt.Printf("\t%s: %s\n", msg, values[ii])
	}
}
