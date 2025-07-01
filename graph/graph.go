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
//   - Backend: manages an XLA/PJRT (through gopjrt) connection: a PJRT plugin and a client.
//     The whole computation building, compilation and execution runs within the scope of a Backend.
//
//   - Graph: created by the Backend, this is used to construct a computation graph that can then
//     be "just-in-time" compiled and executed efficiently.
//     To construct a `Graph` one puts together nodes or "ops" defining the desired sequence of operations.
//
//   - Node: represents the result of an operation ("outputOps" for short). E.g: Add, Sub, Mul, Sigmoid,
//     Reshape, etc. Each node has a fixed shape that is known in "graph building time" (see discussion
//     below).
//
//   - context.Context: created by the Backend, a higher level abstraction convenient when building gradient
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

//go:generate go run ../internal/cmd/graph_generator

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Graph with the operations and dependencies needed to run a computation.
type Graph struct {
	backend backends.Backend
	builder backends.Builder

	id   GraphId
	name string

	// nodes includes all nodes known to Graph.
	nodes []*Node

	// parameters keeps track of parameter nodes, its names and a mapping of name to index.
	parameters            []*Node
	parametersNames       []string
	parameterNameToHandle map[string]ParameterHandle

	traced bool

	// scalars maintains a cache of scalar values already created in the current Graph for re-use.
	scalars scalarCache

	// tensorConstants maintains a cache of tensors that have been converted to a constant node in the graph,
	// to avoid creating duplicate nodes.
	tensorConstants tensorConstCache

	// aliasToNode allows retrieval or nodes by their aliases.
	aliasToNode map[string]*Node

	// aliasScope is the current scope for aliases
	aliasScope []string

	// Compiled Graph
	executable backends.Executable
}

// GraphId is globally unique.
var (
	muGraphCount sync.Mutex
	graphCount   GraphId
)

// tensorConstCache provides a cache of tensors used (converted to constants) in Graph, so they can be reused if needed.
//
// Notice this has the disadvantage of holding a reference to the tensor, while the Graph is alive, so it won't be
// GC-ed until the graph is destroyed.
type tensorConstCache map[*tensors.Tensor]*Node

// NewGraph constructs an empty Graph.
//
// Empty Graph's can still be further configured (e.g. Graph.WithName) until one starts building a computation with them.
//
// After building a computation they can be compiled (see Graph.Compile) at which point they can only be executed.
//
// If they are finalized (see Graph.Finalize) resources are released immediately (instead of waiting for the GC) and
// the Graph can no longer be used.
func NewGraph(backend backends.Backend, name string) *Graph {
	muGraphCount.Lock()
	defer muGraphCount.Unlock()

	if name == "" {
		name = fmt.Sprintf("graph_#%d", graphCount)
	}
	g := &Graph{
		backend:               backend,
		id:                    graphCount,
		name:                  name,
		parameterNameToHandle: make(map[string]ParameterHandle),
		scalars:               make(scalarCache),
		tensorConstants:       make(tensorConstCache),
		aliasToNode:           make(map[string]*Node),
	}
	graphCount += 1
	return g
}

// NodeXlaHandle is used by the underlying XLA implementation.
type NodeXlaHandle int

// GraphId is a unique Graph id within a manager.
type GraphId int

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

// WithName sets the name of the Graph.
//
// It can only be called just after creation of the graph with NewGraph.
// Once any operation is created with the graph, the graph configuration (e.g.: Graph.WithName) become immutable,
// and changing them will panic.
//
// It returns the graph passed, so configuring methods can be cascaded.
func (g *Graph) WithName(name string) *Graph {
	g.AssertConfiguring()
	g.name = name
	return g
}

// build creates the XlaBuilder if not yet created -- which makes the Graph parameters immutable -- and checks
// the Graph hasn't yet been compiled -- in which case it can't be further changed.
func (g *Graph) build() backends.Builder {
	if g.backend == nil {
		exceptions.Panicf("Graph has been finalized already")
	}
	if g.executable != nil {
		exceptions.Panicf("Graph already compiled and can't be used for building")
	}
	if g.builder == nil {
		// Lazy construction of builder: this allows one to further configure the Graph object before using it.
		g.builder = g.backend.Builder(g.name)
	}
	return g.builder
}

// Backend this Graph is using.
func (g *Graph) Backend() backends.Backend { return g.backend }

// Name of the computation this Graph defines, set during its construction.
func (g *Graph) Name() string { return g.name }

// Finalize frees the associated data with the compiled graph (if it is compiled) and
// all the nodes.
// The graph is left in an unusable state.
// It is safe to call it more than once — subsequent calls are no-ops.
func (g *Graph) Finalize() {
	if g == nil {
		return
	}
	if g.builder != nil {
		g.builder = nil
	}
	if g.executable != nil {
		g.executable.Finalize()
		g.executable = nil
	}
	g.nodes = nil
	g.parameters = nil
	g.parametersNames = nil
	g.parameterNameToHandle = nil
	g.name = ""
	g.backend = nil
}

// GraphId is a globally unique id (even across Backend's) of the graph. It's a counter that starts with 0.
func (g *Graph) GraphId() GraphId {
	return g.id
}

// IsValid returns whether the Graph is in a valid state: it is valid if it is in a configuring, building or compiled state.
func (g *Graph) IsValid() bool {
	return !(g == nil || g.backend == nil)
}

// AssertValid panics if graph is nil, or if it has already been finalized.
func (g *Graph) AssertValid() {
	if g == nil {
		exceptions.Panicf("the Graph is nil")
	}
	if g.backend == nil {
		exceptions.Panicf("Graph %q has been finalized already", g.name)
	}
}

// AssertConfiguring panics if graph is not in "configuring" phase: that is, if one already started
// building a computation with it, or if it has already been compiled.
// It also panics if it is not valid (e.g.: if it has been finalized).
func (g *Graph) AssertConfiguring() {
	g.AssertValid()
	if g.builder != nil {
		exceptions.Panicf("Graph %q building already started, it can not be further configured", g.name)
	}
	if g.executable != nil {
		exceptions.Panicf("Graph %q is already compiled, it can not be further configured", g.name)
	}
}

// AssertBuilding panics if graph is nil, has been finalized, or has already been compiled and therefore immutable.
// If Graph was in a configuring state (just after creation) this triggers it to enter into a "building" state.
func (g *Graph) AssertBuilding() {
	g.AssertValid()
	if g.executable != nil {
		exceptions.Panicf("Graph %q has already been compiled, one cannot further build computations with it", g.name)
	}
	_ = g.build()
}

// AssertCompiled panics if graph is nil, if it has already been finalized or if it is not yet compiled.
func (g *Graph) AssertCompiled() {
	g.AssertValid()
	if g.executable == nil {
		exceptions.Panicf("Graph %q not compiled yet, it can't be used for execution", g.name)
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

// registerNode in the graph mapping its underlying outputOps and returns a new unique id within the Graph.
// If Graph.traced is set, it also sets Node.trace to an error with a stack-trace.
func (g *Graph) registerNode(node *Node) (id NodeId) {
	g.AssertBuilding()
	if node.NumOutputs() == 1 && node.DType() == dtypes.InvalidDType {
		exceptions.Panicf("trying to add non-multioutput node with invalid DType: %s", node)
	}
	id = NodeId(len(g.nodes))
	g.nodes = append(g.nodes, node)
	node.id = id
	if g.traced {
		node.trace = errors.New("Stack-trace")
	}
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

// LastNode returns the last node created.
// It returns nil if no node has been created for this graph yet.
func (g *Graph) LastNode() *Node {
	return xslices.Last(g.nodes)
}

// Nodes return a slice of all nodes.
// The slice is owned by Graph and shouldn't be changed.
func (g *Graph) Nodes() []*Node {
	return g.nodes
}

// Compile just-in-time (JIT) compiles the Graph into a Computation that can be executed.
//
// At least one output must be given.
func (g *Graph) Compile(outputs ...*Node) {
	g.AssertValid()
	g.AssertBuilding() // But by then end of this call it will be in "compiled" state.
	if len(outputs) == 0 {
		exceptions.Panicf("no outputs selected when Graph.Compile graph %q", g.name)
	}

	// Sanity check on the output nodes.
	for ii, node := range outputs {
		if node.NumOutputs() != 1 {
			exceptions.Panicf("Graph(%q).Compile cannot take multi-output nodes (output #%d: %s), this type of Node is internal only", g.name, ii, node)
		}
		if node == nil {
			exceptions.Panicf("output node %d is nil when compiling graph %q", ii, g.name)
			panic(nil) // Never executed, just to quiet warning below.
		}
		if node.Graph() != g {
			exceptions.Panicf("output node %d is part of a different graph (name=%q) than the one being compiled (name=%q)",
				ii, node.graph.name, g.name)
		}
	}

	// Create "identities" for duplicate outputs, and create a mapping if there are any:
	outputsSet := types.MakeSet[*Node]()
	for ii, node := range outputs {
		if outputsSet.Has(node) {
			outputs[ii] = Identity(node)
		} else {
			outputsSet.Insert(node)
		}
	}

	if klog.V(1).Enabled() {
		start := time.Now()
		defer func() {
			elapsed := time.Since(start)
			klog.Infof("Graph.Compile time for graph %q: %s", g.Name(), elapsed)
		}()
	}

	outputsOps := xslices.Map(outputs, func(node *Node) backends.Op { return node.outputOps[0] })
	var err error
	g.executable, err = g.builder.Compile(outputsOps...)
	if err != nil {
		panic(errors.WithMessagef(err, "Graph failed to compile for the backend"))
	}
	return
}

// donateBuffer holds a buffer to be donated to the execution of a graph.
// It is built using DonateTensor.
type donateBuffer struct {
	buffer backends.Buffer
	shape  shapes.Shape
}

// DonateTensorBuffer can be used by Graph.Run, Graph.RunWithMap or as input to Exec.Call, and it marks the Tensor to
// donate its on-device buffer to the execution.
//
// This allows the accelerator (GPU) to reuse the space of the donated buffer, which saves space if the original
// value is no longer used.
// Useful in particular is updating some state in a loop.
//
// This doesn't work if the tensor shares the buffer with
// the device (usually CPU plugins). You can check that with IsShared().
//
// Example:
//
//	myState := myExec.Call(DonateTensorBuffer(myState, backend))[0]
//
// It requires the backend and the deviceNum (defaults to 0) of the device buffer to donate.
//
// Notice that after this, t's value in the device becomes invalid.
func DonateTensorBuffer(t *tensors.Tensor, backend backends.Backend, deviceNum ...backends.DeviceNum) any {
	//if t.IsShared() {
	//	exceptions.Panicf("DonateTensorBuffer can only be used for non-shared tensors, for tensor shaped %s", t.Shape())
	//}
	d := &donateBuffer{shape: t.Shape()}
	d.buffer = t.DonateBuffer(backend, deviceNum...) // DonateBuffer may destroy the tensor, if there is no local storage.
	return d
}

// Run the compiled Graph with the inputs given in order -- same order as the parameters were created.
//
// The values for inputs can be:
//
// 1. A tensors.Tensor.
// 2. Any multi-dimensional slice (e.g.: [][]float32 for a 2D float32 value) that is dynamically converted to a temporary tensor.
// 3. The output of DonateTensorBuffer, which then donates the device buffer being used by a tensor -- if there are any.
//
// This is a very "bare bones" way to running the Graph. Typically, one would use the Exec object instead (which
// dynamically generates a new Graph for inputs of different shapes when needed).
//
// To donate the inputs buffers (if they are no longer used, e.g. when updating a state), consider using DonateTensorBuffer.
func (g *Graph) Run(inputs ...any) (outputs []*tensors.Tensor) {
	g.AssertCompiled()
	deviceNum := backends.DeviceNum(0) // Hard-coded for now.

	numParams := g.NumParameters()
	if len(inputs) != numParams {
		exceptions.Panicf("graph %q takes %d parameters, but %d were given to Run()", g.name, numParams, len(inputs))
	}
	buffers := make([]backends.Buffer, numParams)
	donate := make([]bool, numParams)
	for ii, input := range inputs {
		buffers[ii], _, donate[ii] = anyToBuffer(g.backend, deviceNum, input)
	}
	return g.RunWithBuffers(buffers, donate)
}

// RunWithMap runs the compiled graph with the inputs given as a map of the corresponding parameter node to tensor value to use.
//
// The params can use Go values, Local tensors or Device tensors. Go values and Local tensors will be transferred to
// Device tensors (located in the Backend's accelerator memory) before the graph is executed.
//
// This is a very "bare bones" way to running the Graph. Typically, one would use the Exec object instead (which
// dynamically generates a new Graph for inputs of different shapes when needed).
//
// To donate the inputs buffers (if they are no longer used, e.g. when updating a state), consider using DonateTensorBuffer.
func (g *Graph) RunWithMap(inputs ParamsMap) (outputs []*tensors.Tensor) {
	g.AssertCompiled()
	deviceNum := backends.DeviceNum(0) // Hard-coded for now.

	numParams := g.NumParameters()
	if len(inputs) != numParams {
		exceptions.Panicf("graph %q takes %d parameters, but %d were given to RunWithMap()", g.name, numParams, len(inputs))
	}
	for node := range inputs {
		if node.Type() != NodeTypeParameter {
			exceptions.Panicf("graph %q RunWithMap() received a non-parameter node as key to an input", g.name)
		}
	}
	buffers := make([]backends.Buffer, g.NumParameters())
	donate := make([]bool, g.NumParameters())
	for node, value := range inputs {
		handle := node.GetParameterHandle()
		if buffers[handle] != nil {
			exceptions.Panicf("Graph %q input for node %q defined more than once", g.name, node)
		}
		buffers[handle], _, donate[handle] = anyToBuffer(g.backend, deviceNum, value)
	}
	return g.RunWithBuffers(buffers, donate)
}

// RunWithBuffers executes the graph using as inputs the on-device buffers.
//
// For the normal user, consider using the Exec wrapper, or Graph.Run.
//
// The donate slice indicates which buffers can be donated to the execution -- they are immediately finalized after
// the execution is finished.
//
// Notice that for repeated output nodes in the graph (the same output node returned in more than one position), the
// returned tensors are shared.
func (g *Graph) RunWithBuffers(inputs []backends.Buffer, donate []bool) (outputs []*tensors.Tensor) {
	g.AssertCompiled()
	numParams := g.NumParameters()
	if len(inputs) != numParams {
		exceptions.Panicf("graph %q takes %d parameters, but %d were given to RunWithBuffers()", g.name, numParams, len(inputs))
	}
	if len(donate) != numParams {
		exceptions.Panicf("graph %q takes %d donate values for the input parameters, but %d were given to RunWithBuffers()", g.name, numParams, len(donate))
	}
	var start time.Time
	var results []backends.Buffer
	var err error
	if klog.V(1).Enabled() {
		start = time.Now()
		results, err = g.executable.Execute(inputs, donate)
		elapsed := time.Since(start)
		klog.V(1).Infof("Graph.RunWithBuffers: %s elapsed", elapsed)
	} else {
		results, err = g.executable.Execute(inputs, donate)
	}
	if err != nil {
		panic(errors.WithMessagef(err, "Graph failed to execute"))
	}
	outputs = xslices.Map(results, func(buf backends.Buffer) *tensors.Tensor { return tensors.FromBuffer(g.backend, buf) })
	return
}

// anyToBuffer converts generic values to a tensor.Device on the requested device number, and whether the buffer can
// be donated.
func anyToBuffer(backend backends.Backend, deviceNum backends.DeviceNum, value any) (backends.Buffer, shapes.Shape, bool) {
	t, ok := value.(*tensors.Tensor)
	if ok {
		// If a Tensor is given without Donate, it is assumed not for donation.
		return t.Buffer(backend, deviceNum), t.Shape(), false
	}
	b, ok := value.(*donateBuffer)
	if ok {
		return b.buffer, b.shape, true
	}
	// A Go value by default is converted to a buffer and can be donated.
	t = tensors.FromAnyValue(value)
	shape := t.Shape()
	return t.DonateBuffer(backend, deviceNum), shape, true

}

// NumParameters returns the number of parameters created for this graph.
func (g *Graph) NumParameters() int {
	g.AssertValid()
	return len(g.parameters)
}

// GetParameterByHandle returns the ii-th parameter, in order of creation, registered for this graph.
func (g *Graph) GetParameterByHandle(handle ParameterHandle) *Node {
	g.AssertValid()
	return g.parameters[handle]
}

// GetParameterByName returns the parameter registered with the given name. Returns nil if the parameter
// with the given name hasn't been registered (see Parameter method).
func (g *Graph) GetParameterByName(name string) (node *Node) {
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

// String converts the Graph to a multiline string with a description of the full graph.
func (g *Graph) String() string {
	if g == nil {
		return "Graph(nil)!?"
	}
	if g.backend == nil {
		return "Invalid Graph (already finalized)"
	}
	var compiled string
	if g.executable != nil {
		compiled = " (*)"
	}
	parts := []string{fmt.Sprintf("Graph %q%s: %d nodes, %d parameters", g.name, compiled, len(g.nodes), g.NumParameters())}
	for ii, node := range g.nodes {
		parts = append(parts, fmt.Sprintf("\t#%d\t%s", ii, node))
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
