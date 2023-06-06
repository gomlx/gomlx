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

// Package graph is the core package for GoMLX. It is used to create and run computation graphs
// on XLA -- a just-in-time compiler that allows for very efficient numerical computations. It
// also includes an autograd system and many useful higher level machine learning tools.
//
// The main elements in the package are:
//
//   - Manager: holds information on the device where numerical computations will be run. Everything
//     runs within the scope of a Manager. Currently, it supports "Host" (using the CPU for numerical
//     computation, using the Eigen library) and "CUDA" for GPUs. TPU support is also planned.
//
//   - Graph: created by the Manager, this is used to construct a computation graph that can then
//     be "just-in-time" compiled and executed efficiently. To construct a Graph one put together
//     nodes or "ops" defining the operations.
//
//   - Node: represents the result of an operation ("op" for short). E.g: Add, Sub, Mul, Sigmoid,
//     ReshapeWithShape, etc. Each node has a fixed shape that is known in "graph building time" (see discussion
//     below).
//
//   - Context: created by the Manager, a higher level abstraction convenient when building gradient
//     descent based models (like Neural Networks). It organizes Variable objects into "scope", which
//     usually holds the learnable weights for ML. It also allows for loading/saving of these values.
//
// ## Deferred error Handling
//
// Graph (and its Nodes) and Context methods instead of returning error stores them -- or stores
// the first error that happened during the building of a Graph. This way the user doesn't
// need to check for errors at every op -- which severely impact readability. Instead, the user
// can check for the error only at the very end of building a Graph. Since the stack trace is
// preserved, it's easy to trace where and what caused the error.
//
// Notice that unfortunately there is no way to statically, in compile time, check for many
// of the errors that for human would be relatively easy to spot without running the program.
// There is no way in Go (and any other language I know) to run arbitrary logic in compile time.
//
// ## Delayed Execution
//
// When using ML frameworks, it usually requires the user to think about different "times" that
// things are happening. The same is true for GoMLX, and it is helpful to keep those in mind
// upfront, to have the right mental model:
//
//   - **Compile time**: this is during Go compilation. Some amount of type checking is done here, but
//     most of the tensor compatibility cannot be done statically here, unfortunately. Even if for
//     a human it would be obvious without running a program that some operation among different shaped
//     tensors shouldn't be allowed, there is no way in Go (and any other language I know) to run
//     arbitrary logic in compile time to validate tensor shape compatibility. So most of the checking is
//     left to "graph building time".
//
//   - **Graph building time**: this is when one is building a computation Graph, using the various
//     ops (Add, Sub, Mul, ReduceSum, etc.). No actual computation happens here, just the building
//     of the Graph (it's a kind of program) that will be executed later. This happens in "runtime",
//     meaning after the Go program is compiled. And only in graph building time that proper shapes
//     are checked, and good error (with stack traces) are reported back. This means that development
//     often involves coding, and then running the Graph building to see if shapes are correct and
//     what one wants -- Graph building is very fast, since not data is actually manipulated.
//     Creating tests that just build the graph is the recommended way to develop.
//
//   - **Computation/Training/Evaluation time**: this happens after the Graph is built and compiled,
//     and all one does is feed values in, and get the computation out -- using a very fast just-in-time
//     compiled code. error reports here are terser and harder to debug, but usually most of the
//     issues are caught in Graph building time.
//     TODO: improve handling and reporting of things like NaNs appearing in the computation.
//
// TODO: Allow some of the ops to accept constant values directly, so one can write something like
//
//	`Add(x, 1.0)` as opposed to `Add(x, Const(g, 1.0))`
package graph

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
	"strings"
)

// Graph with the operations and dependencies needed to run a computation.
//
// It uses a deferred error reporting model, where if any error happens during the building of a model
// the first error is stored, and all further operations become no-ops. At the very end one can check
// with Graph.Error() if any error occurred and report that: it includes a stack trace. See discussion on
// package documentation.
type Graph struct {
	error error

	manager   *Manager
	graphId   GraphId
	name      string
	deviceNum int
	comp      *xla.Computation

	nodes []*Node

	parameters            []*Node
	parametersNames       []string
	parameterNameToHandle map[string]ParameterHandle

	traced bool

	scalars scalarCache
}

// NodeXlaHandle is used by the underlying XLA implementation.
type NodeXlaHandle int

// GraphId is a unique Graph id within a manager.
type GraphId int

// InvalidNodeXlaHandle is returned when trying to add an invalid node, or a node that
// has no XLA counterpart (usually some form of No-op).
const InvalidNodeXlaHandle = NodeXlaHandle(-1)

// NodeId is a unique NodeId within a Graph
type NodeId int

// InvalidNodeId indicates a node that failed to be created.
const InvalidNodeId = NodeId(-1)

// ParameterHandle is a key to be used by Graph implementations to refer to its
// internal parameters.
type ParameterHandle int

// InvalidParameterHandle represents an invalid (or non-existent) parameter.
const InvalidParameterHandle = ParameterHandle(-1)

// ParamsMap is a shortcut for the map of parameters and their values passed to a graph
// execution. The values are anything that is accepted by tensor.FromAnyValue().
type ParamsMap map[*Node]any

func newGraph(m *Manager, name string, graphId GraphId, deviceNum int) *Graph {
	return &Graph{
		manager:               m,
		graphId:               graphId,
		name:                  name,
		comp:                  xla.NewComputation(m.client, name),
		deviceNum:             deviceNum,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
	}
}

// Name of the computation this Graph defines, set during its construction.
func (g *Graph) Name() string { return g.name }

// Manager this Graph is attached to.
func (g *Graph) Manager() *Manager { return g.manager }

// Client returns the client where this Graph is located, it's given by its manager and indicates
// the device for the computations.
func (g *Graph) Client() *xla.Client { return g.manager.Client() }

// DeviceNum returns the device number where this Graph is executed/built.
func (g *Graph) DeviceNum() int { return g.deviceNum }

var finalizedGraphError = errors.New("Graph has been freed (Graph.Finalize)")

// Finalize frees the associated data with the compiled graph (if it is compiled) and
// all the nodes. The graph is left in an unusable state.
func (g *Graph) Finalize() {
	g.SetError(finalizedGraphError)
	if g.comp != nil {
		g.comp.Finalize()
		g.comp = nil
	}
	g.nodes = nil
}

// GraphId is a unique id (within a Manager) of the graph. It's a counter that starts with 0.
func (g *Graph) GraphId() GraphId {
	return g.graphId
}

// Error returns the first error that happened during the building of the Graph. It's just a convenience
// method to report errors, so they can be handled at the end of Graph building (as opposed to
// at every step). See also `Ok`, which reports whether there were any errors.
// Node creation methods (all the math ops) become no-op if the graph has an error.
func (g *Graph) Error() error {
	if g == nil {
		return errors.Errorf("the Graph is nil")
	}
	return g.error
}

// Ok returns whether there were no errors during the computation Graph building so far.
func (g *Graph) Ok() bool { return g != nil && g.error == nil }

// MustOk panics if graph is not ok, printing stack of where the error happened. Otherwise,
// it's a no-op.
func (g *Graph) MustOk() {
	if !g.Ok() {
		panic(fmt.Sprintf("Graph failed: %+v", g.Error()))
	}
}

// SetError for the Graph. After an error is set, most operations become no-ops. Only the first error
// is kept.
func (g *Graph) SetError(err error) {
	if !g.Ok() {
		return
	}
	g.error = err
}

// ResetError clears the Graph error state. This will not fix any underlying causes of the
// error, and may leave the Graph in an unstable, undefined state. Used only for convenience
// for testing, when Graph errors are deliberately (for testing) being created, and we want
// to reset them (as opposed to creating a new Graph).
func (g *Graph) ResetError() {
	g.error = nil
}

// SetErrorf is similar to SetError, but allows formatting in place. It also automatically
// adds stack trace.
func (g *Graph) SetErrorf(format string, args ...any) {
	if !g.Ok() {
		return
	}
	g.SetError(errors.WithStack(fmt.Errorf(format, args...)))
}

// SetTraced defines whether each node creation is traced. If true, every node
// will save a stack-trace of where it was created, which is helpful for debugging.
// See Node.Track().
func (g *Graph) SetTraced(tracked bool) {
	g.traced = tracked
}

// registerNode in the graph, returning a new unique id within the Graph.
func (g *Graph) registerNode(node *Node) (id NodeId) {
	if !g.Ok() {
		return InvalidNodeId
	}
	id = NodeId(len(g.nodes))
	g.nodes = append(g.nodes, node)
	return
}

func (g *Graph) NodeById(id NodeId) *Node {
	if id == InvalidNodeId {
		return g.InvalidNode()
	}
	if int(id) >= len(g.nodes) {
		g.SetErrorf("invalid request Graph.NodeById(id=%d): there are only %d nodes", id, len(g.nodes))
		return g.InvalidNode()
	}
	return g.nodes[id]
}

// selectOutputNode either takes the last node (if no output was given), or takes the one given (if only one),
// or creates a tuple with all the outputs.
func (g *Graph) selectOutputNode(outputs ...*Node) *Node {
	if !g.Ok() {
		return nil
	}
	for ii, n := range outputs {
		if n == nil {
			g.SetErrorf("output node %d is nil when compiling graph %q", ii, g.name)
			return nil
		}
		if n.Graph() != g {
			g.SetErrorf("output node %d is part of a different graph (name=%q) than the one being compiled (name=%q)", ii, n.graph.name, g.name)
			return nil
		}
	}

	// Find output (root) node.
	var root *Node
	if len(outputs) == 0 {
		root = g.nodes[len(g.nodes)-1]
	} else if len(outputs) == 1 {
		root = outputs[0]
	} else {
		root = Tuple(outputs...)
	}
	return root
}

// Compile just-in-time (JIT) compiles the Graph into a Computation that can be executed.
// If the output node is not given, it assumes it's the last node created in the graph.
// If more than one output is provided, it creates a tuple of those elements, and when
// executed the graph will output a Tuple.
func (g *Graph) Compile(outputs ...*Node) {
	if g.comp.IsCompiled() {
		return
	}
	root := g.selectOutputNode(outputs...)
	if !g.Ok() {
		return
	}

	shapes := make([]shapes.Shape, 0, g.NumParameters())
	for _, node := range g.parameters {
		shapes = append(shapes, node.shape)
	}
	err := g.comp.Compile(shapes, int(root.xlaHandle))
	if err != nil {
		g.SetErrorf("failed to compile with XLA: %w", err)
	}
	return
}

// MustCompile calls Compile and panics if an error happened.
func (g *Graph) MustCompile(output ...*Node) {
	g.Compile(output...)
	if !g.Ok() {
		fmt.Printf("%s\n", g)
		fmt.Printf("First error: %+v\n", g.Error())
		panic(fmt.Sprintf("Failed to compile graph %q: %v", g.name, g.Error()))
	}
}

// ConvertToStableHLO returns the StableHLO C++ object for the compiled graph.
// The graph needs to be compiled.
func (g *Graph) ConvertToStableHLO() (*xla.StableHLO, error) {
	if !g.Ok() {
		return nil, g.Error()
	}
	if !g.comp.IsCompiled() {
		return nil, errors.Errorf("ReadableStableHLO requires a compiled graph")
	}
	if !g.Ok() {
		return nil, g.Error()
	}
	return g.comp.ToStableHLO()
}

// AOTCompile returns the Ahead-Of-Time compiled version of the graph, that can be used for
// execution later.
//
// The graph needs to be compiled. And it is AOT-compiled to the same platform it was already
// compiled -- TODO: cross-compile.
//
// It returns a binary serialized format that can be executed later, without linking the whole GoMLX machinery.
// See tutorial on instructions and an example of how to do this.
func (g *Graph) AOTCompile() ([]byte, error) {
	if !g.Ok() {
		return nil, g.Error()
	}
	if !g.comp.IsCompiled() {
		return nil, errors.Errorf("AOTCompile requires a compiled graph")
	}
	if !g.Ok() {
		return nil, g.Error()
	}
	shapes := make([]shapes.Shape, 0, g.NumParameters())
	for _, node := range g.parameters {
		shapes = append(shapes, node.shape)
	}
	return g.comp.AOTCompile(shapes)
}

// RunError runs the graph with the given parameters, and optionally the output node to calculate.
//
// The params can use Go values, Local tensors or Device tensors. Go values and Local tensors will be transferred to
// Device tensors (located in the Manager's accelerator memory) before the graph is executed.
func (g *Graph) RunError(params ParamsMap) (*tensor.Device, error) {
	numParams := g.NumParameters()
	if len(params) != numParams {
		return nil, errors.Errorf("graph %q takes %d parameters, %d given to RunError()", g.name, numParams, len(params))
	}
	deviceParams, err := g.deviceDataForParam(params)
	if err != nil {
		return nil, err
	}
	result, err := g.comp.Run(deviceParams)
	if err != nil {
		return nil, err
	}
	deviceT := tensor.InternalNewDevice(result)
	return deviceT, deviceT.Error()
}

// RunWithTensors is a slightly faster execution path for the graph, but inputs
// must be provided already in Device tensors and in order.
func (g *Graph) RunWithTensors(params []*tensor.Device) (*tensor.Device, error) {
	if !g.Ok() {
		return nil, errors.WithMessage(g.Error(), "graph in error, cannot be executed")
	}
	deviceParams := make([]*xla.OnDeviceBuffer, 0, len(params))
	for _, param := range params {
		deviceParams = append(deviceParams, param.ShapedBuffer())
	}
	result, err := g.comp.Run(deviceParams)
	if err != nil {
		return nil, errors.Wrap(err, "failed RunWithTensors()")
	}
	deviceT := tensor.InternalNewDevice(result)
	return deviceT, deviceT.Error()
}

// Run runs the graph with the given parameters, and optionally the output node to calculate.
//
// The params can use Go values, Local tensors or Device tensors. Go values and Local tensors will be transferred to
// Device tensors (located in the Manager's accelerator memory) before the graph is executed.
//
// Any errors are reported in the returned tensor.Device.
func (g *Graph) Run(params ParamsMap) *tensor.Device {
	deviceT, err := g.RunError(params)
	if err != nil {
		return tensor.MakeDeviceWithError(errors.Wrap(err, "failed Graph.Run()"))
	}
	return deviceT
}

// MustRun is an alias to RunError that panics in case of error.
func (g *Graph) MustRun(params ParamsMap) *tensor.Device {
	global, err := g.RunError(params)
	if err != nil {
		fmt.Printf("%s\n", g)
		fmt.Printf("Graph error: %+v\n", g.Error())
		panic(fmt.Sprintf("Failed to run graph %q: %v", g.name, err))
	}
	return global
}

// deviceDataForParam converts each parameter to a Device tensor, and then returns their OnDeviceBuffer reference.
func (g *Graph) deviceDataForParam(params ParamsMap) ([]*xla.OnDeviceBuffer, error) {
	buffers := make([]*xla.OnDeviceBuffer, len(params))
	for node, param := range params {
		handle := node.ParameterHandle()
		if handle == InvalidParameterHandle {
			return nil, errors.Errorf("parameter for node %q is invalid (InvalidParameterHandle)", node)
		}
		if !buffers[handle].IsNil() {
			return nil, errors.Errorf("parameter for node %q defined more than once", node)
		}
		if handle == InvalidParameterHandle {
			return nil, errors.Errorf("parameter for node %q doesnt have a valid xlaHandle", node)
		}
		deviceT := anyToDeviceTensor(g.manager, g.deviceNum, param)
		if deviceT.Error() != nil {
			return nil, errors.Wrapf(deviceT.Error(), "parameter for node %q failed to transfer to server", node)
		}
		buffers[handle] = deviceT.ShapedBuffer()
	}
	return buffers, nil
}

// anyToDeviceTensor converts generic values to a tensor.Device on the requested
// device number.
func anyToDeviceTensor(manager *Manager, deviceNum int, value any) *tensor.Device {
	anyT, ok := value.(tensor.Tensor)
	if !ok {
		// Convert Go value to a local tensor.
		localT := tensor.FromAnyValue(value)
		anyT = localT
	}
	return anyT.Device(manager, deviceNum)
}

// NumParameters returns the number of parameters created for this graph.
func (g *Graph) NumParameters() int {
	return len(g.parameters)
}

// ParameterByIndex returns the ii-th parameter, in order of creation, registered for this graph.
func (g *Graph) ParameterByIndex(ii int) *Node {
	return g.parameters[ii]
}

// ParameterByName returns the parameter registered with the given name. Returns nil if the parameter
// with the given name hasn't been registered (see Parameter method).
func (g *Graph) ParameterByName(name string) (node *Node) {
	if !g.Ok() {
		return
	}
	if name == "" {
		return
	}
	handle, ok := g.parameterNameToHandle[name]
	if !ok {
		return
	}
	return g.parameters[handle]
}

// Parameter registers an input parameter for a computation Graph (e.g: a feature used as input).
// It can be used in two different ways: as a Node when building the Graph, so when defining a
// function that uses the parameter, or as the key in the map of the inputs when executing
// the computation Graph (see Manager.RunError).
func (g *Graph) Parameter(name string, shape shapes.Shape) (node *Node) {
	if !g.Ok() {
		return g.InvalidNode()
	}

	parameterHandle := ParameterHandle(len(g.parameters))
	if name == "" {
		name = fmt.Sprintf("p#%d", parameterHandle)
	}

	// Check whether the parameter already exists, and return it instead if yes.
	if handle, ok := g.parameterNameToHandle[name]; ok {
		// If the parameter already exists, return it instead.
		node = g.parameters[handle]
		if !node.shape.Eq(shape) {
			// Shape requested and the one that already exists don't match,
			// report the error.
			g.SetErrorf("requested parameter %q already exists with a different shape:"+
				" requested shape %s, previous shape %s", name, shape, node.shape)
			node = g.InvalidNode()
		}
		return
	}

	// Create new node.
	serialized := &xla.SerializedNode{
		Type:  xla.ParameterNode,
		Str:   name,
		Shape: shape,
		Int:   int(parameterHandle),
	}
	node = newNode(g, serialized, nil)
	if node == nil {
		return g.InvalidNode()
	}
	g.parameters = append(g.parameters, node)
	g.parameterNameToHandle[name] = parameterHandle
	return
}

// String converts the Graph to a multi-Graph string.
func (g *Graph) String() string {
	if !g.Ok() {
		return fmt.Sprintf("Computation Graph: #ERROR: %v", g.error)
	}
	parts := []string{fmt.Sprintf("Graph: %d nodes, %d parameters", len(g.nodes), g.NumParameters())}
	for ii, node := range g.nodes {
		parts = append(parts, fmt.Sprintf("#%d\t%s", ii, node))
	}
	return strings.Join(parts, "\n")
}

// InvalidNode returns an empty node. This is usually what is returned by operations
// when the graph is in error.
func (g *Graph) InvalidNode() *Node {
	if g == nil {
		return nil
	}
	return &Node{graph: g, id: InvalidNodeId}
}

// LoggedNodes returns all nodes from the graph marked to be logged. Exec object makes use
// of this information and logs those values when executing the graph.
func (g *Graph) LoggedNodes() (nodes []*Node) {
	for _, node := range g.nodes {
		if node.IsLogged() {
			nodes = append(nodes, node)
		}
	}
	return
}

// scalarCache provides a cache of a scalar value -- the key always use a float64 -- to
// its pre-created *Node. It helps avoid creating duplicate nodes for common values.
//
// I keeps a cache for each dtype of the scalar.
type scalarCache map[shapes.DType]map[float64]*Node

// getScalarConst either creates a scalar constant or returns a previously created returned
// from the cache. It shouldn't be called directly by users, rather Scalar and Const use it.
func (g *Graph) getScalarConst(dtype shapes.DType, value float64) (output *Node) {
	dtypeMap, found := g.scalars[dtype]
	if !found {
		dtypeMap = make(map[float64]*Node)
		g.scalars[dtype] = dtypeMap
	}
	output, found = dtypeMap[value]
	if found {
		return
	}
	output = Const(g, shapes.CastAsDType(value, dtype))
	dtypeMap[value] = output
	return
}
