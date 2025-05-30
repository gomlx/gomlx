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
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

// Generated by `cmd/constraints_generator`:

// ExecGraphFn is a type parameter for accepted function types for NewExec constructor.
type ExecGraphFn interface {
	func(*Graph) *Node |
		func([]*Node) *Node |
		func(*Node) *Node |
		func(*Node, *Node) *Node |
		func(*Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node, *Node) *Node |
		func(*Graph) (*Node, *Node) |
		func([]*Node) (*Node, *Node) |
		func(*Node) (*Node, *Node) |
		func(*Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node) (*Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node, *Node) (*Node, *Node) |
		func(*Graph) (*Node, *Node, *Node) |
		func([]*Node) (*Node, *Node, *Node) |
		func(*Node) (*Node, *Node, *Node) |
		func(*Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Node, *Node, *Node, *Node, *Node, *Node) (*Node, *Node, *Node) |
		func(*Graph) []*Node |
		func([]*Node) []*Node |
		func(*Node) []*Node |
		func(*Node, *Node) []*Node |
		func(*Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node, *Node) []*Node |
		func(*Node, *Node, *Node, *Node, *Node, *Node) []*Node
}

// ExecGraphFnOneOutput are ExecGraphFn functions that return only one result.
// See ExecOnce.
type ExecGraphFnOneOutput interface {
	func(*Graph) *Node |
		func([]*Node) *Node |
		func(*Node) *Node |
		func(*Node, *Node) *Node |
		func(*Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node) *Node |
		func(*Node, *Node, *Node, *Node, *Node, *Node) *Node
}

// SideParamsFn is the functions that sets side parameters during execution
// for Graphs that defines those. Typically, this is used to set the variables of a model.
type SideParamsFn func(graph *Graph, inputBuffers []backends.Buffer, donate []bool)

// LoggerFn is the function used to log nodes marked for logging. It is called
// after the Call method, with the list of messages and corresponding values
// of the evaluated nodes.
type LoggerFn func(graph *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId)

// Exec creates and executes computation graphs as needed
// based on the inputs shapes.
//
// It simplifies the process of executing a graph building
// function with real values. For example, assume you wrote:
//
//	func L2Norm(x *Node) *Node {
//		return Sqrt(ReduceAllSum(Mul(x, x)))
//	}
//
// To use it with actual values (tensors.Tensor's), one needs to build
// the computation graph for the specific shape of x, and then execute it.
// While this is straight forward, it's lots of boilerplate code -- JIT compilation makes things
// faster, but it imposes some bureaucracy.
//
// With Exec one can do:
//
//	var l2NormExec = NewExec(backends.New(), L2Norm)
//	x0 := []float32{2}
//	fmt.Printf("L2Norm(%v) = %v\n", x0, l2NormExec.Call(x0)[0].Value())
//	x1 := []float64{4, 3}
//	fmt.Printf("L2Norm(%v) = %v\n", x1, l2NormExec.Call(x1)[0].Value())
//
// Notice that both calls to Length.Call will need to create different
// graphs (for different shapes of the input), but they will be cached,
// and if the same shapes are used in Call again, the cached compiled graph
// is reused.
//
// Also, Call outputs a slice with all the outputs, even when there is only
// one output.
//
// If there are no inputs (for instance for some initialization function), then
// one needs to take a *Graph as the first parameter of the graph function (graphFn).
// Example:
//
//	iotaMatrixExec := NewExec(backend, func (g *Graph) *Node {
//		return IotaFull(g, shapes.Make(dtype.Float32, 3, 3))
//	})
//	fmt.Printf("IotaFull(3x3 matrix, float32)=%v\n", iotaMatrixExec.Call()[0].Value())
//
// It also provides a short-form version, that will execute and free the compiled program:
//
//	iotaMatrix := ExecOnce(backend, func (g *Graph) *Node { return IotaFull(g, shapes.Make(dtype.Float32, 3, 3)) })
//
// The need to build different graphs for different shapes can be expensive
// when the shapes of the inputs varies a lot. The usual solution is to use shapes
// with dimensions in a power scale (for instance powers of 2) and masking of tensors
// for unused slices. For safety concerns there are a maximum number of different
// instantiations of the graph. It can be set or disabled with SetMaxCache.
//
// Errors are returned as panic. See Panicf.
type Exec struct {
	backend   backends.Backend
	deviceNum backends.DeviceNum

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
	cache   []*execGraphCacheEntry
}

// execGraphCacheEntry: no hashing, just a simple list. This is faster
// for smaller tables. TODO: add a hashtable for cases with large caches.
type execGraphCacheEntry struct {
	argsShapes     []shapes.Shape
	graph          *Graph
	numOutputs     int      // Number of flattened outputs for this graph, including logged nodes.
	loggedMessages []string // Messages for logged nodes.
	loggedNodeIds  []NodeId
}

// DefaultExecMaxCacheSize is the value used to initialize new Exec objects.
var DefaultExecMaxCacheSize = 32

// NewExecAny constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// `graphFn` can take only *Node parameters as input and returns one or more *Node.
// Except if there are no inputs, in which case graphFn needs to take a *Graph as the first parameter.
//
// It will panic if the inputs are invalid.
//
// See also the generics NewExec, which checks for valid graphFn in compile time.
func NewExecAny(backend backends.Backend, graphFn any) *Exec {
	graphFnT := reflect.TypeOf(graphFn)
	funcName := runtime.FuncForPC(reflect.ValueOf(graphFn).Pointer()).Name()
	exec := &Exec{
		backend:   backend,
		name:      fmt.Sprintf("Exec:%s", funcName),
		deviceNum: 0,

		graphFn:      graphFn,
		numInputs:    graphFnT.NumIn(),
		numOutputs:   graphFnT.NumOut(),
		maxCacheSize: DefaultExecMaxCacheSize,
		loggerFn:     DefaultNodeLogger,
	}

	// Verify parameters.
	if graphFnT.Kind() != reflect.Func {
		Panicf("graphFn must be a function")
	}

	var node *Node
	nodeType := reflect.TypeOf(node)
	var tmpGraph *Graph
	graphType := reflect.TypeOf(tmpGraph)

	if graphFnT.NumIn() < 1 || graphFnT.NumOut() < 1 {
		// It requires at least one input and one output.
		Panicf("not enough input (%d)/output (%d) parameters, both need to be > 0",
			graphFnT.NumIn(), graphFnT.NumOut())
	}
	for ii := 0; ii < graphFnT.NumIn(); ii++ {
		if graphFnT.In(ii).Kind() == reflect.Slice && graphFnT.In(ii).Elem() == nodeType {
			if graphFnT.NumIn() != 1 {
				Panicf("[]*Node parameters are only accepted as input if they are the only input, got function type %s instead", graphFnT)
			}
			exec.inputAsSlice = true
			break
		}
		if graphFnT.In(ii) == graphType {
			if graphFnT.NumIn() != 1 {
				Panicf("*Graph parameter only accepted as input if they are the only input, got function type %s instead", graphFnT)
			}
			exec.inputIsGraph = true
			exec.numInputs = 0
			break
		}
		if graphFnT.In(ii) != nodeType {
			Panicf("input parameter %d is not of type *Node or []*Node", ii)
		}
	}
	for ii := 0; ii < graphFnT.NumOut(); ii++ {
		if graphFnT.Out(ii).Kind() == reflect.Slice && graphFnT.Out(ii).Elem() == nodeType {
			if graphFnT.NumOut() != 1 {
				Panicf("[]*Node parameters are only accepted as output if they are the only output, got function type %s instead", graphFnT)
			}
			exec.outputAsSlice = true
			break
		}
		if graphFnT.Out(ii) != nodeType {
			Panicf("output parameter %d is not of type *Node", ii)
		}
	}
	return exec
}

// NewExec constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// graphFn should take *Node as input and return a *Node -- except if there are no (Node) inputs,
// in which case it should take a single *Graph input.
//
// It's a wrapper for NewExecAny, but uses generics to type check that
// graphFn is valid.
func NewExec[F ExecGraphFn](backend backends.Backend, graphFn F) *Exec {
	return NewExecAny(backend, graphFn)
}

// NewExecOrError creates an Exec object or returns an error if it fails.
//
// It is like NewExec, but it doesn't panic.
func NewExecOrError[F ExecGraphFn](backend backends.Backend, graphFn F) (*Exec, error) {
	var e *Exec
	err := TryCatch[error](func() {
		e = NewExecAny(backend, graphFn)
	})
	if err != nil {
		return nil, err
	}
	return e, nil
}

// ExecOnce builds the graph and executes it with the given arguments, and returns the one output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize for functions that return only one output.
//
// See ExecOnceN if you have multiple outputs.
func ExecOnce[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) *tensors.Tensor {
	return ExecOnceN(backend, graphFn, args...)[0]
}

// ExecOnceN builds the graph and executes it with the given arguments, and returns various output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize.
//
// See ExecOnce for a more convenient version if you have only one output.
func ExecOnceN[F ExecGraphFn](backend backends.Backend, graphFn F, args ...any) []*tensors.Tensor {
	e := NewExec(backend, graphFn)
	defer e.Finalize()
	return e.Call(args...)
}

// InDevice sets the device num to be used by graphs constructed by Exec.
// This should be called before any invocations of Call().
// It returns a reference to itself so calls can be cascaded.
func (e *Exec) InDevice(deviceNum backends.DeviceNum) *Exec {
	e.deviceNum = deviceNum
	return e
}

// DeviceNum returns the device being used by this Exec.
// It defaults to 0 and can be changed with Exec.InDevice.
func (e *Exec) DeviceNum() backends.DeviceNum {
	return e.deviceNum
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

// SetSideParamsHook configures a function to be called just before executing a graph, so it can set extra parameters.
//
// Mostly, this is for internal use and end-users will not likely need this. The context.Exec object uses this to pass
// the variable values as side inputs to the graph.
//
// Exec takes care of creating parameters (with graph.Parameter) for every value passed to Call before
// calling the graph building function (the graph building function is executed only the first time, after the
// graph is compiled it is re-used for future executions).
//
// But a graph building functions may want to create extra parameters itself (with graph.Parameter), which we call
// "side parameters".
//
// The values to feed these "side parameters" are not passed to Exec.Call, but instead set with a SideParamsFn, which
// is configured here.
//
// SideParamsFn is called after the graph is already built, just before the execution.
// It is passed with a slice of the backend.Buffer to be fed to the graph execution.
// The side parameters in this slice will be left nil, and it's expected that SideParamsFn will set
// them to the appropriate input.
//
// It also includes the boolean map of the inputs to donate, which SideParamsFn
// can set accordingly (for the side parameters).
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

// Call parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
// If a graph does not yet exist, one is created, compiled and cached for the shapes.
//
// It returns the outputs in a slice, even if there is only one output.
//
// Errors (with full stack-traces) are raised with `panic`.
func (e *Exec) Call(args ...any) []*tensors.Tensor {
	results, _ := e.CallWithGraph(args...)
	return results
}

// CallOrError parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
// If a graph does not yet exist, one is created, compiled and cached for the shapes.
//
// It returns the outputs in a slice, even if there is only one output.
//
// Errors (with full stack-traces) are returned.
func (e *Exec) CallOrError(args ...any) (results []*tensors.Tensor, err error) {
	err = TryCatch[error](func() {
		results, _ = e.CallWithGraph(args...)
	})
	if err != nil {
		return nil, err
	}
	return results, nil
}

// CallWithGraph is similar to Call, but it also returns the computation graph used
// in the call. Since Exec creates different computation graphs for different set of
// parameters, this can help disambiguate in case the user needs to use the Graph for
// something else.
//
// It returns the outputs in a slice, even if there is only one output, and the graph used
// to execute the computation.
//
// Errors (with full stack-traces) are raised with `panic`.
func (e *Exec) CallWithGraph(args ...any) (results []*tensors.Tensor, g *Graph) {
	return e.compileAndExecute(true, args...)
}

// unwrapListOfTensors will convert something like []any{[]*tensors.Tensor{t1, t2, ...}} to []any{t1, t2,...}
func unwrapListOfTensors(args []any) []any {
	if len(args) != 1 {
		return args
	}
	switch v := args[0].(type) {
	case []*tensors.Tensor:
		return xslices.Map(v, func(x *tensors.Tensor) any { return x })
	}
	// Otherwise, process as usual.
	return args
}

// PreCompile will build the computation graph and compile it, but not yet execute.
// Useful when one wants to measure the time separately, from graph compilation and its execution.
//
// Notice, this will include the time to convert args to tensors. If you want to isolate that time,
// pre-convert args to tensors first.
func (e *Exec) PreCompile(args ...any) {
	_, _ = e.compileAndExecute(false, args...)
	return
}

// compileAndExecute compiles graph for arguments and optionally executes it.
func (e *Exec) compileAndExecute(execute bool, args ...any) (results []*tensors.Tensor, g *Graph) {
	args = unwrapListOfTensors(args)
	if !e.inputAsSlice && len(args) != e.numInputs {
		Panicf(
			"# of arguments to call (#args=%d) don't match # arguments to the graph function (#args=%d) for %q",
			len(args), e.numInputs, e.Name())
	}

	// Convert args to tensors.
	argsAsBuffer := make([]backends.Buffer, len(args)) // There may be more parameters, set with Exec.setSideParams later.
	argsShapes := make([]shapes.Shape, len(args))      // There may be more parameters, set with Exec.setSideParams later.
	argsDonate := make([]bool, len(args))
	for ii, arg := range args {
		err := TryCatch[error](func() {
			argsAsBuffer[ii], argsShapes[ii], argsDonate[ii] = anyToBuffer(e.backend, e.deviceNum, arg)
		})
		if err != nil {
			panic(errors.WithMessagef(err, "Failed to convert argument #%d of %d to device(%d) -- type %T: %v",
				ii, len(args), e.deviceNum, args[ii], args[ii]))
		}
	}

	// Get or build the graph.
	entry := e.findOrCreateGraph(argsShapes)
	if entry == nil {
		Panicf(
			"maximum cache size of %d reached for %q, cannot create another g -- "+
				"a new computation g needs to be created+compiled for each different shape of "+
				"the input, consider using padding, or if this is not a concern change "+
				"the cache size with executable.SetMaxCache()", e.maxCacheSize, e.Name())
		panic(nil) // Disable lint error.
	}
	g = entry.graph

	// Now that the graph is created, we know the exact number of parameters: if the graph building function created
	// new graph.Parameter, we may need to include those in our argsAsBuffer and argsDonate accordingly.
	if g.NumParameters() > len(argsAsBuffer) {
		numNew := g.NumParameters() - len(argsAsBuffer)
		argsAsBuffer = slices.Grow(argsAsBuffer, numNew)
		argsAsBuffer = argsAsBuffer[:g.NumParameters()]
		argsDonate = slices.Grow(argsDonate, numNew)
		argsDonate = argsDonate[:g.NumParameters()]
	}

	// The new parameters (if any) created are still nil, and need to be set. This is done by a "SideParamsFn",
	// configured by Exec.SetSideParamsHooks.
	if e.setSideParams != nil {
		e.setSideParams(g, argsAsBuffer, argsDonate)
	}

	// Check all parameters were set.
	for ii, t := range argsAsBuffer {
		if t == nil {
			Panicf("parameter %d (%q) is nil or invalid, maybe a variable value not set as a "+
				"parameter, cannot execute g", ii, g.GetParameterByHandle(ParameterHandle(ii)).GetParameterName())
		}
	}

	// Execute graph.
	if !execute {
		return
	}
	results = g.RunWithBuffers(argsAsBuffer, argsDonate)

	// Call logger on logged nodes, even if no node is marked for logging (it serves as a hook).
	numGraphFnOutputs := entry.numOutputs - len(entry.loggedMessages)
	if e.loggerFn != nil {
		var loggerResults []*tensors.Tensor
		if len(entry.loggedMessages) > 0 {
			loggerResults = results[numGraphFnOutputs:]
		}
		e.loggerFn(g, entry.loggedMessages, loggerResults, entry.loggedNodeIds)
	}
	if len(results) != numGraphFnOutputs {
		results = results[:numGraphFnOutputs]
	}
	return
}

// createAndCacheGraph creates and compiles the graph for the arguments with the given
// shapes. It creates and stores a cache entry for it and returns it.
// Returns nil if the cache size is >= MaxCacheSize.
// Should be called with cacheMu locked.
func (e *Exec) createAndCacheGraph(argsShapes []shapes.Shape) (entry *execGraphCacheEntry) {
	if e.maxCacheSize > 0 && len(e.cache) >= e.maxCacheSize {
		return nil
	}
	entry = &execGraphCacheEntry{graph: NewGraph(e.backend, fmt.Sprintf("%s#%d", e.name, len(e.cache)))}
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
		arg := Parameter(g, fmt.Sprintf("arg#%d", ii), shape)
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
	start := time.Now()
	outputsV := graphFnV.Call(argsV)
	if klog.V(1).Enabled() {
		elapsed := time.Since(start)
		klog.Infof("Build graph time for %q: %s", e.name, elapsed)
	}

	var outputs []*Node
	if e.outputAsSlice {
		if len(outputsV) != 1 {
			Panicf("graphFn for %q returned %d results, as opposed to simply a slice of nodes -- []*Node",
				e.Name(), len(outputsV))
		}
		outputs = outputsV[0].Interface().([]*Node)
	} else {
		outputs = make([]*Node, 0, len(outputsV))
		for _, outV := range outputsV {
			outputs = append(outputs, outV.Interface().(*Node))
		}
	}

	// Append logged nodes as outputs.
	loggedNodes := g.LoggedNodes()
	entry.loggedMessages = make([]string, 0, len(loggedNodes))
	entry.loggedNodeIds = make([]NodeId, 0, len(loggedNodes))
	for _, node := range loggedNodes {
		outputs = append(outputs, node)
		entry.loggedMessages = append(entry.loggedMessages, node.LogMessage())
		entry.loggedNodeIds = append(entry.loggedNodeIds, node.Id())
	}

	// Compile graph.
	g.Compile(outputs...)
	entry.argsShapes = make([]shapes.Shape, len(argsShapes))
	copy(entry.argsShapes, argsShapes)
	entry.numOutputs = len(outputs)
	e.cache = append(e.cache, entry)
	return entry
}

// findOrCreateGraph returns the graph for the given arguments shapes: either from cache or by creating a new one.
// if no cache entry exists.
func (e *Exec) findOrCreateGraph(argsShapes []shapes.Shape) *execGraphCacheEntry {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()

LoopCache:
	for _, entry := range e.cache {
		if len(argsShapes) != len(entry.argsShapes) {
			continue
		}
		for ii, shape := range argsShapes {
			if !shape.Equal(entry.argsShapes[ii]) {
				continue LoopCache
			}
		}
		return entry
	}

	// No graph in cache, create a new one.
	return e.createAndCacheGraph(argsShapes)
}

// Finalize clears the cache, finalizing the compiled graphs. The Exec object shouldn't be
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

// DefaultNodeLogger for nodes marked to be logged. It prints the message and
// the node value for each logged node.
//
// It accepts special prefixes on message name that affects the printing:
//
//   - #full : prints full tensor value (as opposed to abbreviated).
func DefaultNodeLogger(g *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId) {
	if len(messages) == 0 {
		return
	}
	fmt.Printf("DefaultNodeLogger(Graph %q):\n", g.Name())
	for ii, msg := range messages {
		if strings.HasPrefix(msg, "#full ") {
			fmt.Printf("\t(Node #%d) %s: %s\n", nodes[ii], msg[6:], values[ii].GoStr())
			continue
		}
		fmt.Printf("\t(Node #%d) %s: %s\n", nodes[ii], msg, values[ii])
	}
}
