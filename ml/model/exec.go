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
//
// Notice that the associated Store is linked to the graphs built with it
// (for each different shape of inputs), and is available through model.GetStore(g)
// any time -- but just the Store, you don't get a Scope with it.
type Exec struct {
	backend compute.Backend
	store   *Store
	exec    *graph.Exec

	// Original function that takes scope and the converted closure
	// that only takes *Node as input.
	modelGraphFn                              CanonicalExecGraphFn
	inputIsGraph, inputAsSlice, outputAsSlice bool
	inputShardingSpecs                        []*distributed.ShardingSpec
	outputShardingSpecs                       []*distributed.ShardingSpec

	// numInputs, numOutputs of graphFn, not counting extra logged nodes, which may vary
	// per instance of the graph (entry in the cache).
	// This is relative if inputAsSlice is set to true, in which case graphFn's number of inputs/outputs vary
	// per cache entry.
	numInputs, numOutputs int

	// changedVars maps each graph's GraphId to their list of modified variables.
	// It's used to update the variables in the Store after the graph execution -- these variables are added
	// as extra outputs.
	changedVars   map[graph.GraphId][]*Variable
	muChangedVars sync.Mutex

	// isInitializeVariablesExec indicates this executor is being used to initialize variables.
	// Initializing variables within the modelGraphFn would lead to an infinite recursion.
	// This checks for that.
	isInitializeVariablesExec bool
}

// NewExecCanonical constructs an Exec object for the given model store and symbolic computation function modelGraphFn.
//
// The modelGraphFn must be a CanonicalExecGraphFn.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetNodeValue) are updated also during execution.
// More details see Exec.
func NewExecCanonical(backend compute.Backend, store *Store, modelGraphFn CanonicalExecGraphFn, numInputs, numOutputs int, inputIsGraph, inputAsSlice, outputAsSlice bool) (*Exec, error) {
	if store == nil {
		return nil, errors.Errorf("model.NewExec: store cannot be nil, you can create an empty one with model.NewStore")
	}
	e := &Exec{
		backend:       backend,
		store:         store,
		modelGraphFn:  modelGraphFn,
		numInputs:     numInputs,
		numOutputs:    numOutputs,
		inputIsGraph:  inputIsGraph,
		inputAsSlice:  inputAsSlice,
		outputAsSlice: outputAsSlice,
		changedVars:   make(map[graph.GraphId][]*Variable),
	}

	graphFn := e.buildGraphFn()
	// numInputs, inputIsGraph and inputAsSlice are the same for the wrapped graph.Exec.
	// But numOutputs is variable (slice) because of the changed variables being prepended.
	e.exec = graph.MustNewExecCanonical(backend, graphFn, numInputs, -1, inputIsGraph, inputAsSlice, true)
	funcName := runtime.FuncForPC(reflect.ValueOf(modelGraphFn).Pointer()).Name()
	e.exec.WithName(fmt.Sprintf("Store.Exec:%s", funcName))
	e.exec.SetSideParamsHook(e.setSideParams)
	return e, nil
}

// buildGraphFn constructs a function graphFn that can be passed to the wrapped Exec.
// This function is a closure that will call the modelGraphFn provided by the user with the
// extra *model.Scope argument, plus it prepends the output with the updated values --
// so it can behind the scenes update the variables to the user.
func (e *Exec) buildGraphFn() graph.CanonicalExecGraphFn {
	return func(g *graph.Graph, inputs []*graph.Node) []*graph.Node {
		// Initialize the graph parameters store for this graph.
		g.AttachState(graphState{}, newGraphStore())

		// Attach a link to the model.Store used by model.Exec.
		g.AttachState(graphStoreLink{}, e.store)

		// Call modelGraphFn, the results will be a slice of *Node.
		modelGraphFnResults := e.modelGraphFn(e.store.RootScope(), g, inputs)
		graphId := g.GraphId()

		// Find variables that were changed and their updated graph values (*Node).
		var changedVars []*Variable
		var allValues []*graph.Node
		for _, v := range e.store.variables {
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
		// Append modelGraphFnResults to allValues.
		allValues = append(allValues, modelGraphFnResults...)

		// the results will be a []*Node, which will hold all the values.
		return allValues
	}
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
	if !e.isInitializeVariablesExec && e.store.needsInitialization {
		err := e.store.InitializeVariables(e.backend, func(initExec *Exec) error {
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
	gs := getGraphStore(g)
	deviceAssignment := e.exec.DeviceAssignment()
	deviceNum := compute.DeviceNum(0)
	if len(deviceAssignment) > 0 {
		deviceNum = deviceAssignment[0]
	}

	for _, nodes := range gs.IterVariables() {
		v := nodes.variable
		if nodes == nil || nodes.paramNode == nil || nodes.paramNode.Type() != graph.NodeTypeParameter {
			return errors.Errorf("invalid paramNode for variable %q", v.Path())
		}
		handle := nodes.paramNode.GetParameterHandle()

		if nodes.paramNode != nodes.valueNode {
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
					Panicf("variable %q used and not initialized during variable initialization", v.Path())
				} else {
					Panicf("variable %q failed to initialize", v.Path())
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
	gs := getGraphStore(g)
	numDevices := e.exec.NumDevices()
	numParams := g.NumParameters()
	deviceAssignment := e.exec.DeviceAssignment()

	for _, nodes := range gs.IterVariables() {
		v := nodes.variable
		if nodes == nil || nodes.paramNode == nil || nodes.paramNode.Type() != graph.NodeTypeParameter {
			return errors.Errorf("invalid paramNode for variable %q", v.Path())
		}
		handle := int(nodes.paramNode.GetParameterHandle())

		if !v.HasValue() {
			if e.isInitializeVariablesExec {
				return errors.Errorf("variable %q used and not initialized during variable initialization", v.Path())
			} else {
				return errors.Errorf("variable %q failed to initialize", v.Path())
			}
		}

		dTensor, err := v.DistributedValue()
		if err != nil {
			return errors.WithMessagef(err, "failed to get distributed value for variable %q", v.Path())
		}
		shards := dTensor.Shards()
		changedInGraph := nodes.paramNode != nodes.valueNode
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
						v.Path(), deviceIdx)
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
						v.Path(), deviceIdx)
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
	e.store.defaultShardingSpec = distributed.NewReplicatedShardingSpec(mesh)
	return e
}

// AutoSharding sets the distribution strategy to AutoSharding.
func (e *Exec) AutoSharding(meshes ...*distributed.DeviceMesh) *Exec {
	e.exec.AutoSharding(meshes...)
	e.store.defaultShardingSpec = distributed.NewReplicatedShardingSpec(meshes[0])
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
	return e.store.defaultShardingSpec
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
	e.store.defaultShardingSpec = spec
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
	return e.store
}

// SetStore associates the given Store with the Exec object.
func (e *Exec) SetStore(store *Store) *Exec {
	e.store = store
	return e
}

// Exec parses the arguments and executes the graph.
func (e *Exec) Exec(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.CallWithGraph(args...)
	return outputs, err
}

// CallWithGraph is similar to Exec, but it also returns the computation graph used in the call.
func (e *Exec) CallWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph, err error) {
	outputs, g, err = e.exec.CallWithGraph(args...)
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

// ExecWithGraph is similar to Exec, but it also returns the computation graph used in the call.
//
// Deprecated: please use CallWithGraph instead.
func (e *Exec) ExecWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph, err error) {
	return e.CallWithGraph(args...)
}

// collectOutputs processes outputs for single-device execution.
func (e *Exec) collectOutputs(outputs []*tensors.Tensor, changedVars []*Variable) ([]*tensors.Tensor, error) {
	var firstErr error
	for ii, v := range changedVars {
		if !v.shape.Equal(outputs[ii].Shape()) {
			return nil, errors.Errorf("variable %q changed shape in graph execution", v.Path())
		}
		err := v.SetValue(outputs[ii])
		if err != nil {
			err = errors.WithMessagef(err, "failed updating value for %q", v.Path())
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
			shardingSpec = e.store.defaultShardingSpec
		}
		if shardingSpec == nil {
			err := errors.Errorf("variable %q has no sharding spec", v.Path())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
			continue
		}

		distValue, err := dtensor.NewTensor(shardingSpec, shards)
		if err != nil {
			err = errors.WithMessagef(err, "failed to create distributed tensor for variable %q", v.Path())
			if firstErr == nil {
				firstErr = err
			} else {
				klog.Errorf("Exec error: %v", err)
			}
			continue
		}

		err = v.SetDistributedValue(distValue)
		if err != nil {
			err = errors.WithMessagef(err, "failed updating distributed value for %q", v.Path())
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
