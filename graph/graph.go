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
// on XLA/PJRT (using github.com/gomlx/gopjrt) -- a just-in-time compiler that allows for very
// efficient numerical computations.
//
// It requires PJRT plugins corresponding to accelerators ("cpu, "cuda", "tpu", etc.) to be available
// (see github.com/gomlx/gopjrt) to compile and execute programs. Gopjrt is distributed with a "cpu"
// plugin, and it describes how to install a "cuda" plugin, for Nvidia graphics cards.
//
// It also includes an autograd system and many useful higher level machine learning tools.
//
// The main elements in the package (or related) are:
//
//   - Manager: manages an XLA/PJRT (through gopjrt) connection: a PJRT plugin and a client.
//     The whole computation building, compilation and execution runs within the scope of a Manager.
//
//   - Graph: created by the Manager, this is used to construct a computation graph that can then
//     be "just-in-time" compiled and executed efficiently.
//     To construct a `Graph` one puts together nodes or "ops" defining the desired sequence of operations.
//
//   - Node: represents the result of an operation ("op" for short). E.g: Add, Sub, Mul, Sigmoid,
//     Reshape, etc. Each node has a fixed shape that is known in "graph building time" (see discussion
//     below).
//
//   - context.Context: created by the Manager, a higher level abstraction convenient when building gradient
//     descent based machine learning (ML) models (like Neural Networks). It organizes Variable objects into
//     "scope", which usually holds the learnable weights for ML. It also allows for loading/saving of these values.
//
// ## Error Handling
//
// Graph (and its Nodes) and context.Context methods "throw" errors with `panic`. This prevents having to manage
// error returning for every function call.
// It always throws meaningful error messages, with the full stack, to ease tracking bugs and solve issues.
//
// Notice that unfortunately, there is no way to statically, in compile time, check for many
// of the errors that for a human would be relatively easy to spot without running the program.
// There is no way in Go to run arbitrary logic in compile time.
//
// ## Delayed Execution
//
// When using ML frameworks, it usually requires the user to think about different "times" that
// things are happening. The same is true for GoMLX, and it is helpful to keep those in mind
// upfront, to have the right mental model:
//
//   - **Compile time**: this is during Go compilation. Some amount of type checking is done here, but
//     most of the tensor shape compatibility cannot be done statically here, unfortunately. Even if for
//     a human it would be obvious without compiling and running a program that some operation among
//     different shaped tensors shouldn't be allowed, there is no way in Go to run
//     arbitrary logic in compile time to validate tensor shape compatibility. So most of the checking is
//     left to "graph building time".
//     Maybe one day one can write a gomlx_linter the runs before the compiler that could catch some of these.
//
//   - **Graph building time**: this is when one is building a computation Graph, using the various
//     ops (Add, Sub, Mul, ReduceSum, etc.). No actual computation happens here, just the building
//     of the Graph (it's a kind of program) that will be executed later. This happens in "runtime",
//     meaning after the Go program is compiled. And only in graph building time that proper shapes
//     are checked, and good error (with stack traces) are reported back. This means that development
//     often involves coding, and then running the Graph building to see if shapes are correct and
//     what one wants -- Graph building is very fast, since not data is actually manipulated.
//     Creating tests that just build the graph is the recommended way to develop.
//     Quick sketching of code can be done on a Jupyter Notebook — see
//     [github.com/janpfeifer/gonb](https://github.com/janpfeifer/gonb) for Jupyter notebook support
//     for the Go language.
//     Once the model is built, it is usually "just in time" (JIT) compiled, and can be run.
//
//   - **Computation/Training/Evaluation time**: this happens after the Graph is built and compiled,
//     and all one does is feed values in, and get the computation out -- using a very fast just-in-time
//     compiled code.
//     Error reports here are terser and harder to debug (they come from the underlying C++
//     library), but usually most of the issues are caught in Graph building time.
//     In particular, there is a `nanlogger` library that helps identify where `NaN` or `Inf` first appears
//     in the middle of computation — handy to debug the math.
package graph

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	xla "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"strings"
	"time"
)

// Graph with the operations and dependencies needed to run a computation.
type Graph struct {
	manager   *Manager
	id        GraphId
	name      string
	deviceNum int

	xlaBuilder *xla.XlaBuilder
	exec       *pjrt.LoadedExecutable

	// nodes includes all nodes known to Graph.
	// We also keep a map of xlabuilder.Op to the Graph nodes, but notice that some xlabuilder.Op may be created
	// that the graph doesn't know about, and will not be found in the opToNode mapping.
	nodes    []*Node
	opToNode map[*xla.Op]*Node

	// parameters keeps track of parameter nodes, its names and a mapping of name to index.
	parameters            []*Node
	parametersNames       []string
	parameterNameToHandle map[string]ParameterHandle

	traced bool

	// scalars maintains a cache of scalar values already created in the current Graph for re-use.
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

// newConfiguringGraph creates a Graph without the xlaBuilder set yet, which allows it to be further configured.
func newConfiguringGraph(m *Manager, name string, graphId GraphId, deviceNum int) *Graph {
	return &Graph{
		manager:               m,
		id:                    graphId,
		name:                  name,
		deviceNum:             deviceNum,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
	}
}

// WithName sets the name of the Graph.
//
// It can only be called just after creation of the graph with NewGraph.
// Once any operation is created with the graph, the graph parameters (WithName, WithDeviceNum, etc.) become immutable,
// and changing them will panic.
//
// It returns the graph passed, so configuring methods can be cascaded.
func (g *Graph) WithName(name string) *Graph {
	if g.manager == nil {
		exceptions.Panicf("Graph has been finalized already")
	}
	if g.xlaBuilder != nil {
		exceptions.Panicf("Graph can only be configured before the Graph is actually used, just after it's creation")
	}
	g.name = name
	return g
}

// WithDeviceNum sets the deviceNum to use for this Graph.
//
// It can only be called just after creation of the graph with NewGraph.
// Once any operation is created with the graph, the graph parameters (WithName, WithDeviceNum, etc.) become immutable,
// and changing them will panic.
//
// It returns the graph passed, so configuring methods can be cascaded.
func (g *Graph) WithDeviceNum(deviceNum int) *Graph {
	if g.manager == nil {
		exceptions.Panicf("Graph has been finalized already")
	}
	if g.xlaBuilder != nil {
		exceptions.Panicf("Graph can only be configured before the Graph is actually used, just after it's creation")
	}
	g.deviceNum = deviceNum
	return g
}

// build creates the XlaBuilder if not yet created -- which makes the Graph parameters immutable -- and checks
// the Graph hasn't yet been compiled -- in which case it can't be further changed.
func (g *Graph) build() *xla.XlaBuilder {
	if g.manager == nil {
		exceptions.Panicf("Graph has been finalized already")
	}
	if g.exec != nil {
		exceptions.Panicf("Graph already compiled and can't be used for building")
	}
	if g.xlaBuilder == nil {
		// Lazy construction of builder: this allows one to further configure the Graph object before using it.
		g.xlaBuilder = xla.New(g.name)
	}
	return g.xlaBuilder
}

// Manager this Graph is attached to.
func (g *Graph) Manager() *Manager { return g.manager }

// Name of the computation this Graph defines, set during its construction.
func (g *Graph) Name() string { return g.name }

// DeviceNum returns the device number where this Graph is executed/built.
func (g *Graph) DeviceNum() int { return g.deviceNum }

// Finalize frees the associated data with the compiled graph (if it is compiled) and
// all the nodes.
// The graph is left in an unusable state.
// It is safe to call it more than once — subsequent calls are no-ops.
func (g *Graph) Finalize() {
	if g == nil {
		panic(errors.New("the Graph is nil"))
	}
	if g.xlaBuilder != nil {
		g.xlaBuilder.Destroy()
		g.xlaBuilder = nil
	}
	if g.exec != nil {
		err := g.exec.Destroy()
		if err != nil {
			klog.Errorf("Failure while destroying Graph executable: %+v", err)
		}
		g.exec = nil
	}
	g.nodes = nil
	g.parameters = nil
	g.parametersNames = nil
	g.parameterNameToHandle = nil
	g.name = ""
	g.manager = nil
}

// GraphId is a globally unique id (even across Manager's) of the graph. It's a counter that starts with 0.
func (g *Graph) GraphId() GraphId {
	return g.id
}

// IsValid returns whether the Graph is in a valid state: it is valid if it is in a configuring, building or compiled state.
func (g *Graph) IsValid() bool {
	return !(g == nil || g.manager == nil)
}

// AssertValid panics if graph is nil, or if it has already been finalized.
func (g *Graph) AssertValid() {
	if g == nil {
		exceptions.Panicf("the Graph is nil")
	}
	if g.manager == nil {
		exceptions.Panicf("Graph has been finalized already")
	}
}

// AssertBuilding panics if graph is nil, has been finalized, or has already been compiled and therefore immutable.
// If Graph was in a configuring state (just after creation) this triggers it to enter into a "building" state.
func (g *Graph) AssertBuilding() {
	g.AssertValid()
	_ = g.build()
}

// AssertCompiled panics if graph is nil, if it has already been finalized or if it is not yet compiled.
func (g *Graph) AssertCompiled() {
	g.AssertValid()
	if g.exec == nil {
		exceptions.Panicf("Graph not compiled yet, it can't be used for execution")
	}
}

// SetTraced defines whether each node creation is traced.
// If true, every node will save a stack-trace of where it was created, which is helpful for debugging.
// See Node.Track().
//
// This is expensive, but can be handy for debugging.
func (g *Graph) SetTraced(traced bool) {
	g.AssertBuilding()
	g.traced = traced
}

// registerNode in the graph mapping its underlying op and returns a new unique id within the Graph.
func (g *Graph) registerNode(node *Node) (id NodeId) {
	g.AssertBuilding()
	id = NodeId(len(g.nodes))
	g.nodes = append(g.nodes, node)
	g.opToNode[node.op] = node
	node.id = id
	return
}

// NodeById returns the node for the given id.
func (g *Graph) NodeById(id NodeId) *Node {
	g.AssertBuilding()
	if id == InvalidNodeId || int(id) >= len(g.nodes) {
		exceptions.Panicf("invalid request Graph.NodeById(id=%d): there are only %d nodes", id, len(g.nodes))
	}
	return g.nodes[id]
}

// selectOutputNode either takes the last node (if no output was given), or takes the one given (if only one),
// or creates a tuple with all the outputs.
func (g *Graph) selectOutputNode(outputs ...*Node) *Node {
	g.AssertValid()
	for ii, n := range outputs {
		if n == nil {
			exceptions.Panicf("output node %d is nil when compiling graph %q", ii, g.name)
		}
		if n.Graph() != g {
			exceptions.Panicf("output node %d is part of a different graph (name=%q) than the one being compiled (name=%q)",
				ii, n.graph.name, g.name)
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
//
// If more than one output is provided, it creates a tuple of those elements and use that as output --
// the tuple is split when executed.
func (g *Graph) Compile(outputs ...*Node) {
	g.AssertValid()
	if g.exec.IsCompiled() {
		return
	}

	if klog.V(1).Enabled() {
		start := time.Now()
		defer func() {
			elapsed := time.Since(start)
			klog.Infof("Graph.Compile time for graph %q: %s", g.Name(), elapsed)
		}()
	}

	root := g.selectOutputNode(outputs...)
	outputShapes := make([]shapes.Shape, 0, g.NumParameters())
	for _, node := range g.parameters {
		outputShapes = append(outputShapes, node.shape)
	}
	err := g.exec.Compile(outputShapes, int(root.xlaHandle))
	if err != nil {
		panic(errors.WithMessage(err,
			"Something went wrong when compiling the Graph with XLA, "+
				"if it is graph building related (invalid values) consider reporting this "+
				"in `gihtub.com/gomlx/gomlx` -- try printing your graph `fmt.Printf(\"%s\", g)` "+
				" at the end of your graph building function to help debug"))
	}
	return
}

// Run runs the compiled graph with the given parameters.
//
// The params can use Go values, Local tensors or Device tensors. Go values and Local tensors will be transferred to
// Device tensors (located in the Manager's accelerator memory) before the graph is executed.
func (g *Graph) Run(params ParamsMap) *tensors.Device {
	g.AssertCompiled()
	numParams := g.NumParameters()
	if len(params) != numParams {
		exceptions.Panicf("graph %q takes %d parameters, but %d were given to Run()", g.name, numParams, len(params))
	}
	deviceParams, err := g.deviceDataForParam(params)
	if err != nil {
		panic(errors.WithMessagef(err, "Graph(%q).Run() failed to convert parameters to Device tensor", g.name))
	}
	result, err := g.exec.Run(deviceParams)
	if err != nil {
		panic(errors.WithMessagef(err, "Graph(%q).Run() failed to run JIT compiled", g.name))
	}
	return tensors.InternalNewDevice(result)
}

// RunWithTensors is a slightly faster execution path for the graph, but inputs
// must be provided already in Device tensors and in order.
func (g *Graph) RunWithTensors(params []*tensors.Device) *tensors.Device {
	g.AssertCompiled()
	deviceParams := make([]*xla.OnDeviceBuffer, 0, len(params))
	for _, param := range params {
		deviceParams = append(deviceParams, param.ShapedBuffer())
	}
	result, err := g.exec.Run(deviceParams)
	if err != nil {
		panic(errors.WithMessagef(err, "Graph(%q).RunWithTensors() failed to run JIT compiled", g.name))
	}
	return tensors.InternalNewDevice(result)
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
		device := anyToDeviceTensor(g.manager, g.deviceNum, param)
		buffers[handle] = device.ShapedBuffer()
	}
	return buffers, nil
}

// anyToDeviceTensor converts generic values to a tensor.Device on the requested device number.
func anyToDeviceTensor(manager *Manager, deviceNum int, value any) *tensors.Device {
	t, ok := value.(tensors.Tensor)
	if !ok {
		t = tensors.FromAnyValue(value)
	}
	return t.Device(manager, deviceNum)
}

// NumParameters returns the number of parameters created for this graph.
func (g *Graph) NumParameters() int {
	g.AssertValid()
	return len(g.parameters)
}

// ParameterByIndex returns the ii-th parameter, in order of creation, registered for this graph.
func (g *Graph) ParameterByIndex(ii int) *Node {
	g.AssertValid()
	return g.parameters[ii]
}

// ParameterByName returns the parameter registered with the given name. Returns nil if the parameter
// with the given name hasn't been registered (see Parameter method).
func (g *Graph) ParameterByName(name string) (node *Node) {
	g.AssertValid()
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
// the computation Graph (see Manager.Run).
func (g *Graph) Parameter(name string, shape shapes.Shape) (node *Node) {
	g.AssertValid()
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
			exceptions.Panicf("requested parameter %q already exists with a different shape:"+
				" requested shape %s, previous shape %s", name, shape, node.shape)
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
		return
	}
	g.parameters = append(g.parameters, node)
	g.parameterNameToHandle[name] = parameterHandle
	return
}

// String converts the Graph to a multi-Graph string.
func (g *Graph) String() string {
	if g == nil {
		return "Graph(nil)!?"
	}
	if g.exec.IsNil() {
		return "Invalid Graph (already finalized)"
	}
	parts := []string{fmt.Sprintf("Graph: %d nodes, %d parameters", len(g.nodes), g.NumParameters())}
	for ii, node := range g.nodes {
		parts = append(parts, fmt.Sprintf("#%d\t%s", ii, node))
	}
	return strings.Join(parts, "\n")
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
// It keeps a cache for each dtype of the scalar.
type scalarCache map[dtypes.DType]map[float64]*Node

// getScalarConst either creates a scalar constant or returns a previously created returned
// from the cache. It shouldn't be called directly by users, rather Scalar and Const use it.
func (g *Graph) getScalarConst(dtype dtypes.DType, value float64) (output *Node) {
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
