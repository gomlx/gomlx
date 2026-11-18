// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package graph is the core package for GoMLX. It is used to create and run computation graphs on different compute backends.
//
// The main elements in the package (or related) are:
//
//   - Exec is the driver that manages the lifecycle (Graph creation, compilation, caching, and execution) across
//     different input shapes. This is where most use cases start.
//
//   - Graph is the blueprint for a specific computation with specific input shapes.
//     It's usually created by an Exec object, built by an ExecGraphFn, and then cached and executed by the Exec.
//
//   - Node represents a symbolic value in the computation. This can be an input parameter, a variable/constant,
//     or the result of an operation ("op" for short, e.g.: Add, Sub, Mul, Sigmoid, Reshape, etc.).
package graph

//go:generate go run ../../internal/cmd/graph_generator -public
//go:generate go run ../../internal/cmd/constraints_generator -graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// Graph with the operations and dependencies needed to run a computation.
//
// See details in the package graph documentation.
type Graph struct {
	impl *graphimpl.GraphImpl
}

func wrapGraph(impl *graphimpl.GraphImpl) *Graph {
	if impl == nil {
		return nil
	}
	if pub := impl.Public(); pub != nil {
		return pub.(*Graph)
	}
	g := &Graph{impl: impl}
	impl.SetPublic(g)
	return g
}

func unwrapGraph(g *Graph) *graphimpl.GraphImpl {
	if g == nil {
		return nil
	}
	return g.impl
}

// NewGraph constructs an empty Graph.
//
// An empty Graph can be further configured (e.g., with Graph.WithName) until one starts building a computation.
//
// After building a computation, it can be compiled (see Graph.Compile), at which point the Graph becomes immutable
// and can only be executed.
//
// If it is finalized (see Graph.Finalize), resources are released immediately (instead of waiting for the GC), and
// the Graph can no longer be used.
func NewGraph(backend compute.Backend, name string) *Graph {
	return wrapGraph(graphimpl.NewGraphImpl(backend, name))
}

// GraphId is a unique Graph id within a manager.
type GraphId = graphimpl.GraphId

// NodeId is a unique NodeId within a Graph.
type NodeId = graphimpl.NodeId

// InvalidNodeId indicates a node that failed to be created.
const InvalidNodeId = graphimpl.InvalidNodeId

// ParameterHandle is a key to be used by Graph implementations to refer to its internal parameters.
type ParameterHandle = graphimpl.ParameterHandle

// InvalidParameterHandle represents an invalid (or non-existent) parameter.
const InvalidParameterHandle = graphimpl.InvalidParameterHandle

// GraphProvider is an interface for objects that have a reference to a Graph.
type GraphProvider interface {
	Graph() *Graph
}

// Graph returns itself, to satisfy the GraphProvider interface.
func (g *Graph) Graph() *Graph {
	return g
}

// WithName sets the name of the Graph.
//
// It can only be called before starting to build a Graph.
// It returns the graph passed, so configuring methods can be cascaded.
func (g *Graph) WithName(name string) *Graph {
	g.impl.WithName(name)
	return g
}

// IsBuilding returns whether the Graph already started building a computation.
func (g *Graph) IsBuilding() bool {
	return g.impl.IsBuilding()
}

// Backend this Graph is using.
func (g *Graph) Backend() compute.Backend {
	return g.impl.Backend()
}

// DynamicDimDType returns the DType used by the backend for dynamic dimensions.
// Defaults to dtypes.Int64 if the backend is nil or the field is unset.
func (g *Graph) DynamicDimDType() dtypes.DType {
	return g.impl.DynamicDimDType()
}

// Name of the computation this Graph defines, set during its construction.
func (g *Graph) Name() string {
	return g.impl.Name()
}

// Finalize frees the associated data with the compiled graph (if it is compiled) and all the nodes.
func (g *Graph) Finalize() {
	if g == nil {
		return
	}
	g.impl.Finalize()
}

// GraphId is a globally unique id (even across different Backends) of the graph.
func (g *Graph) GraphId() GraphId {
	return g.impl.GraphId()
}

// IsValid returns whether the Graph is in a valid state.
func (g *Graph) IsValid() bool {
	if g == nil {
		return false
	}
	return g.impl.IsValid()
}

// CheckValid returns an error if the graph is nil or if it has already been finalized.
func (g *Graph) CheckValid() error {
	if g == nil {
		return (*graphimpl.GraphImpl)(nil).CheckValid()
	}
	return g.impl.CheckValid()
}

// AssertValid panics if the graph is nil or if it has already been finalized.
func (g *Graph) AssertValid() {
	if g == nil {
		(*graphimpl.GraphImpl)(nil).AssertValid()
	}
	g.impl.AssertValid()
}

// IsCompiled returns whether the Graph has been compiled (immutable).
func (g *Graph) IsCompiled() bool {
	if g == nil {
		return false
	}
	return g.impl.IsCompiled()
}

// SetTraced defines whether each node creation is traced.
func (g *Graph) SetTraced(traced bool) {
	g.impl.SetTraced(traced)
}

// NodeById returns the node for the given id.
func (g *Graph) NodeById(id NodeId) *Node {
	return wrapNode(g.impl.NodeById(id))
}

// LastNode returns the last node created, or nil.
func (g *Graph) LastNode() *Node {
	return wrapNode(g.impl.LastNode())
}

// Nodes returns a slice of all nodes.
func (g *Graph) Nodes() []*Node {
	return wrapNodes(g.impl.Nodes())
}

// Compile just-in-time (JIT) compiles the Graph into a computation that can be executed.
func (g *Graph) Compile(outputs ...*Node) error {
	return g.impl.Compile(unwrapNodes(outputs)...)
}

// CompileWithSharding compiles the graph with the given outputs and optional sharding specifications.
func (g *Graph) CompileWithSharding(outputs []*Node, outputShardings []*distributed.ShardingSpec) error {
	return g.impl.CompileWithSharding(unwrapNodes(outputs), outputShardings)
}

// DonateTensorBuffer marks t to donate its on-device buffer to Graph.Run / Exec.Call.
func DonateTensorBuffer(t *tensors.Tensor, backend compute.Backend, deviceNum compute.DeviceNum) (any, error) {
	return graphimpl.DonateTensorBuffer(t, backend, deviceNum)
}

// Run the compiled Graph with the inputs given in order.
func (g *Graph) Run(inputs ...any) []*tensors.Tensor {
	return g.impl.Run(inputs...)
}

// RunOnDevice runs the compiled Graph using defaultDeviceNum.
func (g *Graph) RunOnDevice(defaultDeviceNum compute.DeviceNum, inputs ...any) []*tensors.Tensor {
	return g.impl.RunOnDevice(defaultDeviceNum, inputs...)
}

// RunWithBuffers executes the graph using on-device buffers.
func (g *Graph) RunWithBuffers(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) []*tensors.Tensor {
	return g.impl.RunWithBuffers(inputs, donate, defaultDevice)
}

// NumParameters returns the number of parameters created for this graph.
func (g *Graph) NumParameters() int {
	return g.impl.NumParameters()
}

// GetParameterByHandle returns the ii-th parameter, in order of creation, registered for this graph.
func (g *Graph) GetParameterByHandle(handle ParameterHandle) *Node {
	return wrapNode(g.impl.GetParameterByHandle(handle))
}

// GetParameterByName returns the parameter registered with the given name, or nil if not found.
func (g *Graph) GetParameterByName(name string) *Node {
	return wrapNode(g.impl.GetParameterByName(name))
}

// GetParam returns the parameter registered with the given name, or nil if not found.
// This is a shorthand alias for [Graph.GetParameterByName].
func (g *Graph) GetParam(name string) *Node {
	return g.GetParameterByName(name)
}

// String converts the Graph to a multiline string description.
func (g *Graph) String() string {
	if g == nil {
		return "Graph(nil)!?"
	}
	return g.impl.String()
}

// LoggedNodes returns all nodes from the graph marked to be logged.
func (g *Graph) LoggedNodes() []*Node {
	return wrapNodes(g.impl.LoggedNodes())
}

// UniqueName generates a graph-unique name with the given base string as prefix.
func (g *Graph) UniqueName(base string) string {
	return g.impl.UniqueName(base)
}
