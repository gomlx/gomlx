// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"
	"weak"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// GraphId is globally unique across graphs.
type GraphId int

// NodeId is a unique NodeId within a GraphImpl.
type NodeId int

// InvalidNodeId indicates a node that failed to be created.
const InvalidNodeId = NodeId(-1)

// ParameterHandle is a key to refer to a graph's input parameters.
type ParameterHandle int

// InvalidParameterHandle represents an invalid (or non-existent) parameter.
const InvalidParameterHandle = ParameterHandle(-1)

// ParamsMap maps parameter nodes to values for RunWithMap.
type ParamsMap map[*NodeImpl]any

// GraphImpl is the internal representation of a computation graph.
type GraphImpl struct {
	public any // Back-link to canonical *graph.Graph wrapper (for 1-to-1 pointer identity).

	backend               compute.Backend
	builder               compute.Builder
	mainFunc, currentFunc *FunctionImpl

	id   GraphId
	name string

	nodes       []*NodeImpl
	functions   map[string]*FunctionImpl
	traced      bool
	aliasToNode map[string]*NodeImpl
	aliasScope  []string

	executable compute.Executable

	distStrategy     distributed.Strategy
	deviceMeshes     []*distributed.DeviceMesh
	deviceAssignment []compute.DeviceNum
	numDevices       int

	state       map[any]any
	uniqueNames map[string]int
}

var (
	muGraphCount sync.Mutex
	graphCount   GraphId
)

type tensorConstCacheEntry struct {
	node    *NodeImpl
	version uint64
}

type tensorConstCache struct {
	mu      sync.Mutex
	entries map[weak.Pointer[tensors.Tensor]]tensorConstCacheEntry
}

func newTensorConstCache() *tensorConstCache {
	return &tensorConstCache{
		entries: make(map[weak.Pointer[tensors.Tensor]]tensorConstCacheEntry),
	}
}

func (c *tensorConstCache) get(t *tensors.Tensor) *NodeImpl {
	if c == nil || t == nil {
		return nil
	}
	wp := weak.Make(t)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return nil
	}
	entry, found := c.entries[wp]
	if !found || entry.version != t.Version() {
		return nil
	}
	return entry.node
}

func (c *tensorConstCache) set(t *tensors.Tensor, node *NodeImpl, version uint64) {
	if c == nil || t == nil {
		return
	}
	wp := weak.Make(t)
	c.mu.Lock()
	if c.entries == nil {
		c.entries = make(map[weak.Pointer[tensors.Tensor]]tensorConstCacheEntry)
	}
	_, existed := c.entries[wp]
	c.entries[wp] = tensorConstCacheEntry{
		node:    node,
		version: version,
	}
	c.mu.Unlock()
	if !existed {
		runtime.AddCleanup(t, func(wp weak.Pointer[tensors.Tensor]) {
			c.mu.Lock()
			delete(c.entries, wp)
			c.mu.Unlock()
		}, wp)
	}
}

func (c *tensorConstCache) clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.entries = nil
	c.mu.Unlock()
}

// NewGraphImpl constructs an empty GraphImpl.
func NewGraphImpl(backend compute.Backend, name string) *GraphImpl {
	muGraphCount.Lock()
	defer muGraphCount.Unlock()

	if name == "" {
		name = fmt.Sprintf("graph_#%d", graphCount)
	}
	g := &GraphImpl{
		backend:      backend,
		id:           graphCount,
		name:         name,
		aliasToNode:  make(map[string]*NodeImpl),
		functions:    make(map[string]*FunctionImpl),
		distStrategy: distributed.None,
		numDevices:   1,
		state:        make(map[any]any),
		uniqueNames:  make(map[string]int),
	}
	graphCount += 1
	return g
}

// Public returns the canonical *graph.Graph wrapper (stored as any), or nil.
func (g *GraphImpl) Public() any {
	if g == nil {
		return nil
	}
	return g.public
}

// SetPublic records the canonical *graph.Graph wrapper on g.
func (g *GraphImpl) SetPublic(pub any) {
	g.public = pub
}

// WithName sets the name of the GraphImpl.
func (g *GraphImpl) WithName(name string) *GraphImpl {
	g.AssertConfiguring()
	g.name = name
	return g
}

// IsBuilding returns whether the GraphImpl is in building mode.
func (g *GraphImpl) IsBuilding() bool {
	return g.IsValid() && g.builder != nil && !g.IsCompiled()
}

func (g *GraphImpl) setBuilding() {
	if !g.IsValid() {
		exceptions.Panicf("Graph is nil or has been finalized already")
	}
	if g.IsCompiled() {
		exceptions.Panicf("Graph already compiled and can't be used for building")
	}
	if g.builder != nil {
		return
	}
	g.builder = g.backend.Builder(g.name)
	g.mainFunc = g.newMainFunc()
	g.currentFunc = g.mainFunc
	if err := g.setupBuilderDistribution(); err != nil {
		panic(err)
	}
}

// Backend returns the compute.Backend used by this GraphImpl.
func (g *GraphImpl) Backend() compute.Backend { return g.backend }

// Builder returns the compute.Builder used by this GraphImpl (nil if not yet building or finalized).
func (g *GraphImpl) Builder() compute.Builder { return g.builder }

// DynamicDimDType returns the DType used by the backend for dynamic dimensions.
func (g *GraphImpl) DynamicDimDType() dtypes.DType {
	if g == nil || g.backend == nil {
		return dtypes.Int64
	}
	return g.backend.Capabilities().DynamicDimDTypeOrDefault()
}

// Executable returns the compiled compute.Executable, or nil.
func (g *GraphImpl) Executable() compute.Executable {
	return g.executable
}

// Name returns the name of the GraphImpl.
func (g *GraphImpl) Name() string { return g.name }

// Finalize frees resources associated with the graph.
func (g *GraphImpl) Finalize() {
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
	for _, f := range g.functions {
		if f != nil {
			f.tensorConstants.clear()
		}
	}
	g.nodes = nil
	g.mainFunc = nil
	g.currentFunc = nil
	g.functions = nil
	g.name = ""
	g.backend = nil
}

// GraphId returns the globally unique id of the graph.
func (g *GraphImpl) GraphId() GraphId {
	return g.id
}

// CurrentFunc returns the current function scope.
func (g *GraphImpl) CurrentFunc() *FunctionImpl {
	return g.currentFunc
}

// IsMainFunc returns whether the current function scope is the main function.
func (g *GraphImpl) IsMainFunc() bool {
	return g.currentFunc.IsMain()
}

// IsValid returns whether the GraphImpl is valid (not nil and not finalized).
func (g *GraphImpl) IsValid() bool {
	return !(g == nil || g.backend == nil)
}

// CheckValid returns an error if the graph is nil or finalized.
func (g *GraphImpl) CheckValid() error {
	if g == nil {
		return errors.Errorf("the Graph is nil")
	}
	if g.backend == nil {
		return errors.Errorf("Graph %q has been finalized already", g.name)
	}
	return nil
}

// AssertValid panics if the graph is nil or finalized.
func (g *GraphImpl) AssertValid() {
	if err := g.CheckValid(); err != nil {
		panic(err)
	}
}

// AssertConfiguring panics if the graph is not in configuring mode.
func (g *GraphImpl) AssertConfiguring() {
	g.AssertValid()
	if g.builder != nil {
		exceptions.Panicf("Graph %q building already started, it can not be further configured", g.name)
	}
	if g.executable != nil {
		exceptions.Panicf("Graph %q is already compiled, it can not be further configured", g.name)
	}
}

// IsCompiled returns whether the graph has been compiled.
func (g *GraphImpl) IsCompiled() bool {
	return g.IsValid() && g.executable != nil
}

// AssertBuilding panics if the graph cannot be built; enters building state if configuring.
func (g *GraphImpl) AssertBuilding() {
	g.AssertValid()
	g.setBuilding()
	g.AssertNotReturned()
}

// AssertNotReturned panics if the current function already returned or graph is compiled.
func (g *GraphImpl) AssertNotReturned() {
	g.AssertValid()
	if g.mainFunc == nil {
		return
	}
	if g.currentFunc.returned || g.IsCompiled() {
		exceptions.Panicf("Graph %q function %q has already returned or been compiled", g.name, g.currentFunc.name)
	}
}

// AssertCompiled panics if the graph is not compiled yet.
func (g *GraphImpl) AssertCompiled() {
	g.AssertValid()
	if !g.IsCompiled() {
		exceptions.Panicf("Graph %q not compiled yet, it can't be used for execution", g.name)
	}
}

// SetTraced enables or disables stack-trace capture at node creation.
func (g *GraphImpl) SetTraced(traced bool) {
	g.AssertBuilding()
	g.traced = traced
}

func (g *GraphImpl) registerNode(node *NodeImpl) (id NodeId) {
	g.AssertBuilding()
	if node.NumOutputs() == 1 && node.DType() == dtypes.InvalidDType {
		exceptions.Panicf("trying to add non-multioutput node with invalid DType: %s", node)
	}

	if len(node.inputNodes) == 0 {
		node.scope = g.currentFunc
	} else {
		scope, err := innermostFunction(node.inputNodes)
		if err != nil {
			exceptions.Panicf("failed to find scope for node %s: %v", node, err)
		}
		if scope == nil {
			node.scope = g.currentFunc
		} else {
			node.scope = scope
		}
	}
	for _, input := range node.inputNodes {
		if input.needsRematerialization {
			node.needsRematerialization = true
			break
		}
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
func (g *GraphImpl) NodeById(id NodeId) *NodeImpl {
	g.AssertBuilding()
	if id == InvalidNodeId || int(id) >= len(g.nodes) {
		exceptions.Panicf("invalid request Graph.NodeById(id=%d): there are only %d nodes", id, len(g.nodes))
	}
	return g.nodes[id]
}

// LastNode returns the last node created, or nil.
func (g *GraphImpl) LastNode() *NodeImpl {
	return xslices.Last(g.nodes)
}

// Nodes returns the slice of all nodes in this graph.
func (g *GraphImpl) Nodes() []*NodeImpl {
	return g.nodes
}

// Compile compiles the graph with the given outputs.
func (g *GraphImpl) Compile(outputs ...*NodeImpl) error {
	return g.CompileWithSharding(outputs, nil)
}

// CompileWithSharding compiles the graph with outputs and optional output shardings.
func (g *GraphImpl) CompileWithSharding(outputs []*NodeImpl, outputShardings []*distributed.ShardingSpec) error {
	g.currentFunc.Return(outputs, outputShardings)

	var err error
	g.executable, err = g.builder.Compile()
	if err != nil {
		return errors.WithMessagef(err, "graph %q failed to compile for the backend %q", g.name, g.backend.Name())
	}
	return nil
}

type donateBuffer struct {
	buffer compute.Buffer
	shape  shapes.Shape
}

// DonateTensorBuffer creates a donation wrapper for a tensor's device buffer.
func DonateTensorBuffer(t *tensors.Tensor, backend compute.Backend, deviceNum compute.DeviceNum) (any, error) {
	d := &donateBuffer{shape: t.Shape()}
	var err error
	d.buffer, err = t.DonateBuffer(backend, deviceNum)
	if err != nil {
		return nil, err
	}
	return d, nil
}

// Run executes the compiled graph with the given inputs on device 0.
func (g *GraphImpl) Run(inputs ...any) (outputs []*tensors.Tensor) {
	g.AssertCompiled()
	return g.RunOnDevice(0, inputs...)
}

// RunOnDevice executes the compiled graph on defaultDeviceNum.
func (g *GraphImpl) RunOnDevice(defaultDeviceNum compute.DeviceNum, inputs ...any) (outputs []*tensors.Tensor) {
	numParams := g.NumParameters()
	numDevices := g.NumDevices()
	if len(inputs) != numParams*numDevices {
		exceptions.Panicf("graph %q takes %d parameters * %d devices, but %d were given to Graph.Run()",
			g.name, numParams, numDevices, len(inputs))
	}
	buffers := make([]compute.Buffer, numParams*numDevices)
	donate := make([]bool, numParams*numDevices)
	if g.deviceMeshes == nil {
		deviceNum := defaultDeviceNum
		for ii, input := range inputs {
			buffers[ii], _, donate[ii] = anyToDeviceBuffer(g.backend, deviceNum, input)
		}
	} else {
		deviceAssignment := g.deviceAssignment
		for ii, input := range inputs {
			deviceIndex := ii / numParams
			deviceNum := deviceAssignment[deviceIndex]
			buffers[ii], _, donate[ii] = anyToDeviceBuffer(g.backend, deviceNum, input)
		}
	}
	return g.RunWithBuffers(buffers, donate, defaultDeviceNum)
}

// RunWithMap executes the compiled graph using a map of parameter nodes to values.
func (g *GraphImpl) RunWithMap(inputs ParamsMap) (outputs []*tensors.Tensor) {
	g.AssertCompiled()
	deviceNum := compute.DeviceNum(0)

	numParams := g.NumParameters()
	if len(inputs) != numParams {
		exceptions.Panicf(
			"graph %q takes %d parameters, but %d were given to RunWithMap()",
			g.name,
			numParams,
			len(inputs),
		)
	}
	for node := range inputs {
		if node.Type() != NodeTypeParameter {
			exceptions.Panicf("graph %q RunWithMap() received a non-parameter node as key to an input", g.name)
		}
	}
	buffers := make([]compute.Buffer, g.NumParameters())
	donate := make([]bool, g.NumParameters())
	for node, value := range inputs {
		handle := node.GetParameterHandle()
		if buffers[handle] != nil {
			exceptions.Panicf("Graph %q input for node %q defined more than once", g.name, node)
		}
		buffers[handle], _, donate[handle] = anyToDeviceBuffer(g.backend, deviceNum, value)
	}
	return g.RunWithBuffers(buffers, donate, deviceNum)
}

// RunWithBuffers executes the graph using on-device buffers.
func (g *GraphImpl) RunWithBuffers(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) (
	outputs []*tensors.Tensor) {
	g.AssertCompiled()
	numParams := g.NumParameters()
	numDevices := g.NumDevices()
	if len(inputs) != numParams*numDevices {
		exceptions.Panicf("graph %q takes %d parameters * %d devices, but %d were given to RunWithBuffers()",
			g.name, numParams, numDevices, len(inputs))
	}
	if len(donate) != numParams*numDevices {
		exceptions.Panicf("graph %q takes %d * %d donate values for the input parameters, "+
			"but %d were given to RunWithBuffers()", g.name, numParams, numDevices, len(donate))
	}
	var results []compute.Buffer
	var err error
	if klog.V(1).Enabled() {
		start := time.Now()
		results, err = g.executable.Execute(inputs, donate, defaultDevice)
		elapsed := time.Since(start)
		klog.V(1).Infof("Graph.RunWithBuffers: %s elapsed", elapsed)
	} else {
		results, err = g.executable.Execute(inputs, donate, defaultDevice)
	}
	if err != nil {
		panic(errors.WithMessagef(err, "Graph failed to execute"))
	}
	outputs = xslices.Map(
		results,
		func(buf compute.Buffer) *tensors.Tensor {
			t, err := tensors.FromBuffer(buf)
			if err != nil {
				panic(err)
			}
			return t
		},
	)
	return
}

func must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func anyToDeviceBuffer(
	backend compute.Backend,
	deviceNum compute.DeviceNum,
	value any,
) (compute.Buffer, shapes.Shape, bool) {
	if t, ok := value.(*tensors.Tensor); ok {
		buf, shape, err := tensorToDeviceBuffer(backend, deviceNum, t)
		if err != nil {
			panic(err)
		}
		return buf, shape, false
	}
	if b, ok := value.(*donateBuffer); ok {
		return b.buffer, b.shape, true
	}
	t := tensors.MustFromAnyValue(value)
	shape := t.Shape()
	return must1(t.DonateBuffer(backend, deviceNum)), shape, true
}

func tensorToDeviceBuffer(
	backend compute.Backend,
	deviceNum compute.DeviceNum,
	t *tensors.Tensor,
) (compute.Buffer, shapes.Shape, error) {
	var shape shapes.Shape
	err := t.CheckValid()
	if err != nil {
		return nil, shape, err
	}
	currentDevice, err := t.Device()
	if err == nil {
		if currentDevice == deviceNum {
			buf, err := t.Buffer(backend, deviceNum)
			if err != nil {
				return nil, shape, err
			}
			return buf, t.Shape(), nil
		}
		return nil, shape, errors.Errorf(
			"tensor stored on the wrong deviceNum #%d, expected it to be on device #%d where it's going "+
				"to be used -- this is likely a logic error, so it is not transferred automatically",
			currentDevice, deviceNum)
	}
	if !t.IsLocal() {
		return nil, shape, errors.Errorf(
			"tensor %q is not local and not on a device, cannot transfer to deviceNum #%d", t.Shape(), deviceNum)
	}
	buf, err := t.Buffer(backend, deviceNum)
	if err != nil {
		return nil, shape, err
	}
	return buf, t.Shape(), nil
}

// NumParameters returns the number of parameters created for this graph.
func (g *GraphImpl) NumParameters() int {
	g.AssertValid()
	if g.mainFunc == nil {
		return 0
	}
	return len(g.mainFunc.parameters)
}

// GetParameterByHandle returns the ii-th parameter registered for this graph.
func (g *GraphImpl) GetParameterByHandle(handle ParameterHandle) *NodeImpl {
	g.AssertValid()
	if g.mainFunc == nil {
		return nil
	}
	return g.mainFunc.parameters[handle]
}

// GetParameterByName returns the parameter registered with the given name, or nil.
func (g *GraphImpl) GetParameterByName(name string) (node *NodeImpl) {
	g.AssertValid()
	if name == "" || g.mainFunc == nil {
		return
	}
	handle, ok := g.mainFunc.parameterNameToHandle[name]
	if !ok {
		return
	}
	return g.mainFunc.parameters[handle]
}

// String returns a human-readable summary of the graph.
func (g *GraphImpl) String() string {
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
	parts := []string{
		fmt.Sprintf("Graph %q%s: %d nodes, %d parameters", g.name, compiled, len(g.nodes), g.NumParameters()),
	}
	for ii, node := range g.nodes {
		parts = append(parts, fmt.Sprintf("\t#%d\t%s", ii, node))
	}
	return strings.Join(parts, "\n")
}

// LoggedNodes returns all nodes marked to be logged.
func (g *GraphImpl) LoggedNodes() (nodes []*NodeImpl) {
	for _, node := range g.nodes {
		if node.IsLogged() {
			nodes = append(nodes, node)
		}
	}
	return
}

type scalarCache map[dtypes.DType]map[float64]*NodeImpl

// GetScalarConst returns or creates a cached scalar constant node in the current function scope.
func (g *GraphImpl) GetScalarConst(dtype dtypes.DType, value float64) (output *NodeImpl) {
	g.AssertBuilding()
	f := g.currentFunc
	dtypeMap, found := f.scalars[dtype]
	if !found {
		dtypeMap = make(map[float64]*NodeImpl)
		f.scalars[dtype] = dtypeMap
	}
	output, found = dtypeMap[value]
	if found {
		return output
	}
	output = Const(g, shapes.CastAsDType(value, dtype))
	dtypeMap[value] = output
	return output
}

// AttachState attaches an arbitrary opaque state object to the graph.
func (g *GraphImpl) AttachState(key any, value any) {
	if key == nil {
		panic(errors.New("key cannot be nil"))
	}
	if value == nil {
		delete(g.state, key)
		return
	}
	g.state[key] = value
}

// State retrieves an opaque state object attached to the graph.
func (g *GraphImpl) State(key any) any {
	return g.state[key]
}

// UniqueName generates a graph-unique name with the given prefix.
func (g *GraphImpl) UniqueName(base string) string {
	if g.uniqueNames == nil {
		g.uniqueNames = make(map[string]int)
	}
	count, found := g.uniqueNames[base]
	g.uniqueNames[base] = count + 1
	if !found {
		return base
	}
	return fmt.Sprintf("%s_%d", base, count)
}
