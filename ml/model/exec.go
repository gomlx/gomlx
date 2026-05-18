// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"reflect"
	"runtime"
	"slices"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Exec creates and executes computation graphs that take as input a
// Store as needed based on the inputs shapes, to allow the function
// to access (both read and set) variables and everything in the Store.
// Otherwise, very similar to graph.Exec.
type Exec struct {
	backend compute.Backend
	scope   *Scope
	exec    *graph.Exec

	// Original function that takes ctx and the converted closure
	// that only takes *Node as input.
	ctxGraphFn, graphFn                       any
	inputIsGraph, inputAsSlice, outputAsSlice bool
	inputShardingSpecs                        []*distributed.ShardingSpec
	outputShardingSpecs                       []*distributed.ShardingSpec

	// changedVars maps each graph's GraphId to their list of modified variables.
	// It's used to update the variables in the Store after the graph execution -- these variables are added
	// as extra outputs.
	changedVars   map[graph.GraphId][]*Variable
	muChangedVars sync.Mutex

	// isInitializeVariablesExec indicates this executor is being used to initialize variables.
	// Initializing variables within the cxtGraphFn would lead to an infinite recursion.
	// This checks for that.
	isInitializeVariablesExec bool
}

// NewExecAny constructs an Exec object for the given scopeOrStore and symbolic computation function ctxGraphFn.
//
// The scopeOrStore can be a *Scope or a *Store. If it is a *Store, it is converted to its RootScope().
// If nil, it automatically creates a new empty store and uses its RootScope().
//
// The ctxGraphFn is called to build the computation graphs with a Scope.
// It must take a *Scope input parameter followed by one or more *Node parameters as input and return one or more *Node.
// Alternatively, it can, instead of *Node inputs, take a *Graph object, when there are no input tensors.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetValueGraph) are updated also during execution.
// More details see Exec.
func NewExecAny(backend compute.Backend, scopeOrStore any, ctxGraphFn any) (*Exec, error) {
	var scope *Scope
	if scopeOrStore == nil {
		scope = NewStore().RootScope()
	} else {
		switch v := scopeOrStore.(type) {
		case *Store:
			scope = v.RootScope()
		case *Scope:
			scope = v
		default:
			return nil, errors.Errorf("scopeOrStore must be a *model.Scope or *model.Store, got %T instead", scopeOrStore)
		}
	}
	e := &Exec{
		backend:     backend,
		scope:       scope,
		ctxGraphFn:  ctxGraphFn,
		changedVars: make(map[graph.GraphId][]*Variable),
	}
	ctxGraphFnT := reflect.TypeOf(ctxGraphFn)
	if ctxGraphFnT.Kind() != reflect.Func {
		return nil, errors.Errorf("ctxGraphFn must be a function")
	}
	nodeType := reflect.TypeFor[*Node]()
	scopeType := reflect.TypeFor[*Scope]()
	graphType := reflect.TypeFor[*Graph]()

	// Must have at least 2 arguments, and the first must be of type *Scope.
	if ctxGraphFnT.NumIn() < 2 {
		return nil, errors.Errorf("at least *Scope and one input argument required")
	}
	if ctxGraphFnT.In(0) != scopeType {
		return nil, errors.Errorf(
			"the first argument for ctxGraphFn must be a *Scope, got %s instead",
			ctxGraphFnT.In(0),
		)
	}

	// Check other arguments.
	for ii := 1; ii < ctxGraphFnT.NumIn(); ii++ {
		if ctxGraphFnT.In(ii).Kind() == reflect.Slice && ctxGraphFnT.In(ii).Elem() == nodeType {
			// Case 1: []*Node
			if ctxGraphFnT.NumIn() != 2 {
				return nil, errors.Errorf(
					"[]*Node parameters are only accepted if they are the only input besides the Scope, got function type %s instead",
					ctxGraphFnT,
				)
			}
			e.inputAsSlice = true
			break
		}
		if ctxGraphFnT.In(ii) == graphType {
			// Case 2: *Graph
			if ctxGraphFnT.NumIn() != 2 {
				return nil, errors.Errorf(
					"*Graph argument is only accepted if it is the only input besides the Scope, got function type %s instead",
					ctxGraphFnT,
				)
			}
			e.inputIsGraph = true
			break
		}
		if ctxGraphFnT.In(ii) != nodeType {
			return nil, errors.Errorf("input parameter %d is not of type *Node", ii)
		}
	}
	for ii := 0; ii < ctxGraphFnT.NumOut(); ii++ {
		if ctxGraphFnT.Out(ii).Kind() == reflect.Slice && ctxGraphFnT.Out(ii).Elem() == nodeType {
			if ctxGraphFnT.NumOut() != 1 {
				return nil, errors.Errorf(
					"[]*Node parameters are only accepted as output if they are the only output, got function type %s instead",
					ctxGraphFnT,
				)
			}
			e.outputAsSlice = true
			break
		}
		if ctxGraphFnT.Out(ii) != nodeType {
			return nil, errors.Errorf("output parameter %d is not of type *Node", ii)
		}
	}

	e.buildGraphFn()
	e.exec = graph.MustNewExecAny(backend, e.graphFn)
	funcName := runtime.FuncForPC(reflect.ValueOf(ctxGraphFn).Pointer()).Name()
	e.exec.WithName(fmt.Sprintf("Store.Exec:%s", funcName))
	e.exec.SetSideParamsHook(e.setSideParams)
	return e, nil
}

// buildGraphFn constructs a function graphFn that can be passed to the wrapped Exec.
// This function is a closure that will call the ctxGraphFn provided by the user with the
// extra *model.Scope argument, plus it prepends the output with the updated values --
// so it can behind the scenes update the variables to the user.
func (e *Exec) buildGraphFn() {
	ctxGraphFnT := reflect.TypeOf(e.ctxGraphFn)
	numIn := ctxGraphFnT.NumIn() - 1
	nodeT := reflect.TypeFor[*Node]()
	nodeSliceT := reflect.TypeFor[[]*Node]()
	graphT := reflect.TypeFor[*Graph]()

	// Build input types for new graphFn: same as ctxGraphFn, but without the Scope.
	var inT []reflect.Type
	if e.inputIsGraph {
		// The only input is a graph.
		inT = []reflect.Type{graphT}
	} else if e.inputAsSlice {
		// The only input is a []*Node.
		inT = []reflect.Type{nodeSliceT}
	} else {
		inT = make([]reflect.Type, numIn)
		for ii := range numIn {
			inT[ii] = nodeT
		}
	}

	// Output types for a new graphFn: it is converted to a []*Node, because we will prepend
	// the changed variables as extra outputs.
	outT := []reflect.Type{nodeSliceT}

	// Builds the function that will be called without Scope, by computation.Exec. It will take as
	// input a slice of *Node (or only a *computation.Graph), and as output also a slice of *Node.
	graphFnT := reflect.FuncOf(inT, outT, false)
	e.graphFn = reflect.MakeFunc(graphFnT, func(args []reflect.Value) (results []reflect.Value) {
		// Inputs for the original ctxGraphFn: we prepend the scope to the arguments.
		argsWithScope := make([]reflect.Value, len(args)+1)
		argsWithScope[0] = reflect.ValueOf(e.scope.Store().Scope(e.scope.Scope()))
		copy(argsWithScope[1:], args)

		// Call ctxGraphFn, the results will be a slice of *Node.
		ctxGraphFnResults := reflect.ValueOf(e.ctxGraphFn).Call(argsWithScope)

		// Find the graph.
		var g *Graph
		if e.inputIsGraph {
			g = args[0].Interface().(*Graph)
		} else if e.inputAsSlice {
			nodes := args[0].Interface().([]*Node)
			g = nodes[0].Graph()
		} else {
			node := args[0].Interface().(*Node)
			g = node.Graph()
		}
		graphId := g.GraphId()

		// Find variables that were changed and their updated graph values (*Node).
		var changedVars []*Variable
		var allValues []*Node
		for _, v := range e.scope.store.variables {
			if v.ChangedInGraph(g) {
				changedVars = append(changedVars, v)
				allValues = append(allValues, v.NodeValue(g))
			}
		}
		{
			// Save list of variables changed.
			e.muChangedVars.Lock()
			e.changedVars[graphId] = changedVars
			e.muChangedVars.Unlock()
		}
		// Append ctxGraphFnResults to allValues.
		if e.outputAsSlice {
			// cxtGraphResults returns one value, a []*Node, easy to append.
			allValues = append(allValues, ctxGraphFnResults[0].Interface().([]*Node)...)
		} else {
			// Append one result at a time (it's ok if there are no results).
			for _, r := range ctxGraphFnResults {
				allValues = append(allValues, r.Interface().(*Node))
			}
		}

		// the results will be a []*Node, which will hold all the values.
		results = []reflect.Value{reflect.ValueOf(allValues)}
		return
	}).Interface()
}

// Finalize clears the cache, finalizing and releasing the memory for all compiled graphs.
// The Exec object shouldn't be used after that.
func (e *Exec) Finalize() {
	e.exec.Finalize()
}

// setSideParams is used by computation.Exec.SetSideParamsHook to set up
// the variable values as parameters just before graph execution.
func (e *Exec) setSideParams(g *Graph, inputBuffers []compute.Buffer, donate []bool) error {
	// Initialize variables if needed.
	store := e.scope.store
	if !e.isInitializeVariablesExec && store.needsInitialization {
		err := store.InitializeVariables(e.backend, func(initExec *Exec) error {
			return initExec.ConfigureDistributionFrom(e)
		})
		if err != nil {
			return errors.WithMessagef(err, "failed to initialize variables")
		}
	}

	numDevices := e.exec.NumDevices()
	if numDevices > 1 {
		return e.setSideParamsDistributed(g, inputBuffers, donate)
	}
	return e.setSideParamsSingleDevice(g, inputBuffers, donate)
}

// setSideParamsSingleDevice sets the side parameters for single-device execution.
func (e *Exec) setSideParamsSingleDevice(g *Graph, inputBuffers []compute.Buffer, donate []bool) error {
	store := e.scope.store
	graphId := g.GraphId()
	deviceAssignment := e.exec.DeviceAssignment()
	deviceNum := compute.DeviceNum(0)
	if len(deviceAssignment) > 0 {
		deviceNum = deviceAssignment[0]
	}

	for _, v := range store.variables {
		nodes, found := v.graphToNodes.Load(graphId)
		if !found {
			continue
		}
		if nodes == nil || nodes.paramNode == nil || nodes.paramNode.Type() != graph.NodeTypeParameter {
			return errors.Errorf("invalid paramNode for variable %q", v.ParameterName())
		}
		handle := nodes.paramNode.GetParameterHandle()

		if v.ChangedInGraph(g) {
			// We donate the buffer, since we are getting a new one on the output.
			value, err := v.Value()
			if err != nil {
				return err
			}
			inputBuffers[handle], err = value.DonateBuffer(e.backend, deviceNum)
			if err != nil {
				return err
			}
			err = v.Reset()
			if err != nil {
				return err
			}
			donate[handle] = true
		} else {
			if !v.HasValue() {
				if e.isInitializeVariablesExec {
					Panicf("variable %q used and not initialized during variable initialization", v.ScopeAndName())
				} else {
					Panicf("variable %q failed to initialize", v.ScopeAndName())
				}
			}
			value, err := v.Value()
			if err != nil {
				return err
			}
			inputBuffers[handle], err = value.Buffer(e.backend, deviceNum)
			if err != nil {
				return err
			}
			donate[handle] = false
		}
	}
	return nil
}

// setSideParamsDistributed sets the side parameters for distributed execution.
func (e *Exec) setSideParamsDistributed(g *Graph, inputBuffers []compute.Buffer, donate []bool) error {
	store := e.scope.store
	graphId := g.GraphId()
	numDevices := e.exec.NumDevices()
	numParams := g.NumParameters()
	deviceAssignment := e.exec.DeviceAssignment()

	for _, v := range store.variables {
		nodes, found := v.graphToNodes.Load(graphId)
		if !found {
			continue
		}
		if nodes == nil || nodes.paramNode == nil || nodes.paramNode.Type() != graph.NodeTypeParameter {
			return errors.Errorf("invalid paramNode for variable %q", v.ParameterName())
		}
		handle := int(nodes.paramNode.GetParameterHandle())

		if !v.HasValue() {
			if e.isInitializeVariablesExec {
				return errors.Errorf("variable %q used and not initialized during variable initialization", v.ScopeAndName())
			} else {
				return errors.Errorf("variable %q failed to initialize", v.ScopeAndName())
			}
		}

		dTensor, err := v.DistributedValue()
		if err != nil {
			return errors.WithMessagef(err, "failed to get distributed value for variable %q", v.ScopeAndName())
		}
		shards := dTensor.Shards()
		changedInGraph := v.ChangedInGraph(g)
		if changedInGraph {
			// Donate buffers since we'll get new ones on output.
			for deviceIdx := range numDevices {
				bufIdx := deviceIdx*numParams + handle
				deviceNum := compute.DeviceNum(deviceIdx)
				if len(deviceAssignment) > deviceIdx {
					deviceNum = deviceAssignment[deviceIdx]
				}
				inputBuffers[bufIdx], err = shards[deviceIdx].DonateBuffer(e.backend, deviceNum)
				if err != nil {
					return errors.WithMessagef(err, "failed to donate buffer for variable %q on device %d",
						v.ScopeAndName(), deviceIdx)
				}
				donate[bufIdx] = true
			}
			// Reset the variable since we donated all shards.
			err = v.Reset()
			if err != nil {
				return err
			}
		} else {
			for deviceIdx := range numDevices {
				bufIdx := deviceIdx*numParams + handle
				deviceNum := compute.DeviceNum(deviceIdx)
				if len(deviceAssignment) > deviceIdx {
					deviceNum = deviceAssignment[deviceIdx]
				}
				inputBuffers[bufIdx], err = shards[deviceIdx].Buffer(e.backend, deviceNum)
				if err != nil {
					return errors.WithMessagef(err, "failed to get buffer for variable %q on device %d",
						v.ScopeAndName(), deviceIdx)
				}
				donate[bufIdx] = false
			}
		}
	}
	return nil
}

// ConfigureDistributionFrom configures the distribution of the executor from another executor.
func (e *Exec) ConfigureDistributionFrom(e2 *Exec) error {
	switch e2.exec.DistributionStrategy() {
	case distributed.None:
		return nil
	case distributed.AutoSharding:
		e.AutoSharding(e2.Meshes()...)
	case distributed.SPMD:
		e.SPMD(e2.Meshes()[0])
	}
	deviceAssignment := e2.DeviceAssignment()
	e.WithDeviceAssignment(deviceAssignment)
	return nil
}

// SetNodeLogger sets the node logger.
func (e *Exec) SetNodeLogger(loggerFn graph.LoggerFn) {
	e.exec.SetNodeLogger(loggerFn)
}

// GetNodeLogger returns the currently registered LoggerFn.
func (e *Exec) GetNodeLogger() graph.LoggerFn {
	return e.exec.GetNodeLogger()
}

// WithDeviceAssignment specifies which concrete devices to use.
func (e *Exec) WithDeviceAssignment(devices []compute.DeviceNum) *Exec {
	e.exec.WithDeviceAssignment(devices)
	return e
}

// DeviceAssignment returns the current device assignment.
func (e *Exec) DeviceAssignment() []compute.DeviceNum {
	return e.exec.DeviceAssignment()
}

// DistributionStrategy returns the distribution strategy.
func (e *Exec) DistributionStrategy() distributed.Strategy {
	return e.exec.DistributionStrategy()
}

// NumDevices returns the number of devices used.
func (e *Exec) NumDevices() int {
	return e.exec.NumDevices()
}

// SPMD sets the distribution strategy to SPMD.
func (e *Exec) SPMD(mesh *distributed.DeviceMesh) *Exec {
	e.exec.SPMD(mesh)
	e.scope.store.defaultShardingSpec = distributed.NewReplicatedShardingSpec(mesh)
	return e
}

// AutoSharding sets the distribution strategy to AutoSharding.
func (e *Exec) AutoSharding(meshes ...*distributed.DeviceMesh) *Exec {
	e.exec.AutoSharding(meshes...)
	e.scope.store.defaultShardingSpec = distributed.NewReplicatedShardingSpec(meshes[0])
	return e
}

// Meshes returns the slice of currently configured meshes.
func (e *Exec) Meshes() []*distributed.DeviceMesh {
	return e.exec.Meshes()
}

// WithInputShardingSpecs sets the sharding specs for the inputs.
func (e *Exec) WithInputShardingSpecs(specs ...*distributed.ShardingSpec) *Exec {
	e.inputShardingSpecs = specs
	e.exec.WithInputShardingSpecs(specs...)
	return e
}

// WithOutputShardingSpecs sets the sharding specs for the outputs.
func (e *Exec) WithOutputShardingSpecs(specs ...*distributed.ShardingSpec) *Exec {
	e.outputShardingSpecs = specs
	e.exec.WithOutputShardingSpecs(specs...)
	return e
}

// DefaultShardingSpec returns the default sharding spec.
func (e *Exec) DefaultShardingSpec() *distributed.ShardingSpec {
	return e.scope.store.defaultShardingSpec
}

// SetDefaultShardingSpec sets the default sharding spec for the associated Store object.
func (e *Exec) SetDefaultShardingSpec(spec *distributed.ShardingSpec) error {
	if spec != nil {
		if e.exec.DistributionStrategy() == distributed.None {
			return errors.Errorf("cannot set non-nil default sharding spec for non-distributed execution")
		}
		mesh := spec.Mesh
		execMeshes := e.exec.Meshes()
		if slices.Index(execMeshes, mesh) == -1 {
			return errors.Errorf("spec given uses a mesh %q that was not configured", mesh.Name())
		}
		if err := spec.Validate(); err != nil {
			return err
		}
	} else {
		if e.exec.DistributionStrategy() != distributed.None {
			spec = distributed.NewReplicatedShardingSpec(e.exec.Meshes()[0])
		}
	}
	e.scope.store.defaultShardingSpec = spec
	return nil
}

// WithName sets the name of Exec.
func (e *Exec) WithName(name string) *Exec {
	e.exec.WithName(name)
	return e
}

// Name returns the Exec name.
func (e *Exec) Name() string {
	return e.exec.Name()
}

// SetMaxCache sets the maximum size of the cache.
func (e *Exec) SetMaxCache(maxCacheSize int) *Exec {
	e.exec.SetMaxCache(maxCacheSize)
	return e
}

// Store returns the associated Store object.
func (e *Exec) Store() *Store {
	return e.scope.store
}

// SetStore associates the given Store with the Exec object.
func (e *Exec) SetStore(store *Store) *Exec {
	e.scope = store.Scope(e.scope.Scope())
	return e
}

// Exec parses the arguments and executes the graph.
func (e *Exec) Exec(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.ExecWithGraph(args...)
	return outputs, err
}

// ExecWithGraph is similar to Exec, but it also returns the computation graph used in the call.
func (e *Exec) ExecWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph, err error) {
	outputs, g, err = e.exec.ExecWithGraph(args...)
	if err != nil {
		return nil, nil, err
	}

	e.muChangedVars.Lock()
	changedVars := e.changedVars[g.GraphId()]
	e.muChangedVars.Unlock()
	numDevices := e.exec.NumDevices()
	numOutputsPerDevice := len(outputs) / numDevices

	if len(changedVars) > numOutputsPerDevice {
		return nil, nil, errors.Errorf("not enough outputs of the graph for updated variables")
	}

	if numDevices > 1 {
		outputs, err = e.collectOutputsForDistributed(outputs, changedVars, numDevices, numOutputsPerDevice)
	} else {
		outputs, err = e.collectOutputs(outputs, changedVars)
	}
	if err != nil {
		return nil, nil, err
	}
	return
}

// collectOutputs processes outputs for single-device execution.
func (e *Exec) collectOutputs(outputs []*tensors.Tensor, changedVars []*Variable) ([]*tensors.Tensor, error) {
	var firstErr error
	for ii, v := range changedVars {
		if !v.shape.Equal(outputs[ii].Shape()) {
			return nil, errors.Errorf("variable %q changed shape in graph execution", v.ScopeAndName())
		}
		err := v.SetValue(outputs[ii])
		if err != nil {
			err = errors.WithMessagef(err, "failed updating value for %q", v.ScopeAndName())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return outputs[len(changedVars):], nil
}

// collectOutputsForDistributed processes outputs for distributed execution.
func (e *Exec) collectOutputsForDistributed(
	outputs []*tensors.Tensor, changedVars []*Variable, numDevices, numOutputsPerDevice int) (
	[]*tensors.Tensor, error) {
	var firstErr error

	for varIdx, v := range changedVars {
		shards := make([]*tensors.Tensor, numDevices)
		for deviceIdx := range numDevices {
			shards[deviceIdx] = outputs[deviceIdx*numOutputsPerDevice+varIdx]
		}

		shardingSpec := v.shardingSpec
		if shardingSpec == nil {
			shardingSpec = e.scope.store.defaultShardingSpec
		}
		if shardingSpec == nil {
			err := errors.Errorf("variable %q has no sharding spec", v.ScopeAndName())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
			continue
		}

		distValue, err := dtensor.NewTensor(shardingSpec, shards)
		if err != nil {
			err = errors.WithMessagef(err, "failed to create distributed tensor for variable %q", v.ScopeAndName())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
			continue
		}

		err = v.SetDistributedValue(distValue)
		if err != nil {
			err = errors.WithMessagef(err, "failed updating distributed value for %q", v.ScopeAndName())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
		}
	}

	if firstErr != nil {
		return nil, firstErr
	}

	numParamsPerDevice := numOutputsPerDevice - len(changedVars)
	newOutputs := make([]*tensors.Tensor, numDevices*numParamsPerDevice)
	for deviceIdx := range numDevices {
		for paramIdx := range numParamsPerDevice {
			srcIdx := deviceIdx*numOutputsPerDevice + len(changedVars) + paramIdx
			dstIdx := deviceIdx*numParamsPerDevice + paramIdx
			newOutputs[dstIdx] = outputs[srcIdx]
		}
	}
	return newOutputs, nil
}

// PreCompile will build the computation graph, JIT-compile and cache it, but not yet execute.
func (e *Exec) PreCompile(args ...any) error {
	return e.exec.PreCompile(args...)
}
