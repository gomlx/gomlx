// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"
	"reflect"
	"runtime"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// SideParamsFn is a function that sets side parameters during execution for Graphs that define them.
type SideParamsFn func(graph *Graph, inputBuffers []compute.Buffer, donate []bool) error

// LoggerFn is the function used to log nodes marked for logging.
type LoggerFn func(graph *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId)

// ErrExecFinalized is returned when an operation (such as Call or Compile) is attempted on a finalized Exec.
var ErrExecFinalized = graphimpl.ErrExecFinalized

// DefaultExecMaxCacheSize is the default max cache size for new Exec objects.
const DefaultExecMaxCacheSize = graphimpl.DefaultExecMaxCacheSize

// Exec creates and executes computation graphs as needed based on the input shapes.
type Exec struct {
	impl     *graphimpl.ExecImpl
	loggerFn LoggerFn
}

func wrapExec(impl *graphimpl.ExecImpl) *Exec {
	if impl == nil {
		return nil
	}
	if pub := impl.Public(); pub != nil {
		return pub.(*Exec)
	}
	e := &Exec{impl: impl, loggerFn: DefaultNodeLogger}
	impl.SetPublic(e)
	return e
}

func unwrapExec(e *Exec) *graphimpl.ExecImpl {
	if e == nil {
		return nil
	}
	return e.impl
}

// NewExecCanonical constructs an Exec object that uses graphFn to build computation graphs.
func NewExecCanonical(backend compute.Backend, graphFn CanonicalExecGraphFn, numInputs, numOutputs int, inputIsGraph, inputAsSlice, outputAsSlice bool) (*Exec, error) {
	implFn := func(ig *graphimpl.GraphImpl, iArgs []*graphimpl.NodeImpl) []*graphimpl.NodeImpl {
		return unwrapNodes(graphFn(wrapGraph(ig), wrapNodes(iArgs)))
	}
	impl, err := graphimpl.NewExecCanonicalImpl(backend, implFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice)
	if err != nil {
		return nil, err
	}
	return wrapExec(impl), nil
}

// NewExec constructs an Exec object that uses graphFn to build computation graphs.
func NewExec[F ExecGraphFn](backend compute.Backend, graphFn F) (*Exec, error) {
	canonicalFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice := convertExecFn(graphFn)
	exec, err := NewExecCanonical(backend, canonicalFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice)
	if err != nil {
		return nil, err
	}
	funcName := runtime.FuncForPC(reflect.ValueOf(graphFn).Pointer()).Name()
	exec.WithName(fmt.Sprintf("Exec:%s", funcName))
	return exec, nil
}

// WithName sets the name of Exec, used as a prefix for created graphs.
func (e *Exec) WithName(name string) *Exec {
	e.impl.WithName(name)
	return e
}

// WithDynamicAxes sets the names of axes treated as dynamic for each input parameter.
func (e *Exec) WithDynamicAxes(axisNames ...[]string) *Exec {
	e.impl.WithDynamicAxes(axisNames...)
	return e
}

// WithDeviceAssignment specifies which concrete devices to use when compiling computation graphs.
func (e *Exec) WithDeviceAssignment(devices []compute.DeviceNum) *Exec {
	e.impl.WithDeviceAssignment(devices)
	return e
}

// DeviceAssignment returns the current device assignment used by this Exec.
func (e *Exec) DeviceAssignment() []compute.DeviceNum {
	return e.impl.DeviceAssignment()
}

// DistributionStrategy returns the distribution strategy used by this Exec.
func (e *Exec) DistributionStrategy() distributed.Strategy {
	return e.impl.DistributionStrategy()
}

// SPMD sets the distribution strategy to SPMD.
func (e *Exec) SPMD(mesh *distributed.DeviceMesh) *Exec {
	e.impl.SPMD(mesh)
	return e
}

// AutoSharding sets the distribution strategy to AutoSharding.
func (e *Exec) AutoSharding(meshes ...*distributed.DeviceMesh) *Exec {
	e.impl.AutoSharding(meshes...)
	return e
}

// Meshes returns the slice of currently configured meshes.
func (e *Exec) Meshes() []*distributed.DeviceMesh {
	return e.impl.Meshes()
}

// WithInputShardingSpecs sets the sharding specs for the inputs.
func (e *Exec) WithInputShardingSpecs(specs ...*distributed.ShardingSpec) *Exec {
	e.impl.WithInputShardingSpecs(specs...)
	return e
}

// WithOutputShardingSpecs sets the sharding specs for the outputs.
func (e *Exec) WithOutputShardingSpecs(specs ...*distributed.ShardingSpec) *Exec {
	e.impl.WithOutputShardingSpecs(specs...)
	return e
}

// NumDevices returns the number of devices used by the current strategy.
func (e *Exec) NumDevices() int {
	return e.impl.NumDevices()
}

// Name returns the Exec name.
func (e *Exec) Name() string {
	return e.impl.Name()
}

// SetMaxCache sets the maximum size of the cache (-1 for unlimited).
func (e *Exec) SetMaxCache(maxCacheSize int) *Exec {
	e.impl.SetMaxCache(maxCacheSize)
	return e
}

// SetSideParamsHook configures a function to be called just before executing a graph to set side parameters.
func (e *Exec) SetSideParamsHook(fn SideParamsFn) *Exec {
	if fn == nil {
		e.impl.SetSideParamsHook(nil)
		return e
	}
	e.impl.SetSideParamsHook(func(ig *graphimpl.GraphImpl, inputBuffers []compute.Buffer, donate []bool) error {
		return fn(wrapGraph(ig), inputBuffers, donate)
	})
	return e
}

// SetNodeLogger sets the function to be called for nodes marked for logging during execution.
func (e *Exec) SetNodeLogger(loggerFn LoggerFn) {
	e.loggerFn = loggerFn
	if loggerFn == nil {
		e.impl.SetNodeLogger(nil)
		return
	}
	e.impl.SetNodeLogger(func(ig *graphimpl.GraphImpl, messages []string, values []*tensors.Tensor, nodes []NodeId) {
		loggerFn(wrapGraph(ig), messages, values, nodes)
	})
}

// GetNodeLogger returns the currently registered LoggerFn.
func (e *Exec) GetNodeLogger() LoggerFn {
	return e.loggerFn
}

// CallWithGraphOnDevice compiles/executes the graph on defaultDevice and returns outputs and the Graph.
func (e *Exec) CallWithGraphOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, *Graph, error) {
	outputs, ig, err := e.impl.CallWithGraphOnDevice(defaultDevice, args...)
	if err != nil {
		return nil, nil, err
	}
	return outputs, wrapGraph(ig), nil
}

// ExecWithGraphOnDevice is a deprecated alias to CallWithGraphOnDevice.
//
// Deprecated: use CallWithGraphOnDevice instead.
func (e *Exec) ExecWithGraphOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, *Graph, error) {
	return e.CallWithGraphOnDevice(defaultDevice, args...)
}

// Compile creates (if not yet created) and compiles the computation graph for the given input shapes.
func (e *Exec) Compile(inputShapes ...shapes.Shape) (*Graph, error) {
	ig, err := e.impl.Compile(inputShapes...)
	if err != nil {
		return nil, err
	}
	return wrapGraph(ig), nil
}

// CompileStandalone builds and compiles a fresh graph with this executor's configuration.
// The caller owns the returned graph and must call Graph.Finalize when finished.
func (e *Exec) CompileStandalone(inputShapes ...shapes.Shape) (*Graph, error) {
	ig, err := e.impl.CompileStandalone(inputShapes...)
	if err != nil {
		return nil, err
	}
	return wrapGraph(ig), nil
}

// Finalize clears the cache and finalizes compiled graphs.
func (e *Exec) Finalize() {
	e.impl.Finalize()
}

// DefaultNodeLogger prints logged nodes to stdout.
func DefaultNodeLogger(g *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId) {
	graphimpl.DefaultNodeLogger(unwrapGraph(g), messages, values, nodes)
}
