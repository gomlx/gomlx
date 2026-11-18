// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// SideParamsFn sets side parameters during graph execution.
type SideParamsFn func(graph *GraphImpl, inputBuffers []compute.Buffer, donate []bool) error

// LoggerFn logs marked nodes after ExecImpl execution.
type LoggerFn func(graph *GraphImpl, messages []string, values []*tensors.Tensor, nodes []NodeId)

// CanonicalExecGraphFn is the canonical graph-building function type for ExecImpl.
type CanonicalExecGraphFn func(*GraphImpl, []*NodeImpl) []*NodeImpl

// ErrExecFinalized is returned when calling or compiling a finalized ExecImpl.
var ErrExecFinalized = errors.New("exec is finalized")

// DefaultExecMaxCacheSize is the default max cache size for new ExecImpl instances.
const DefaultExecMaxCacheSize = 32

// ExecImpl manages graph creation, compilation, caching, and execution across input shapes.
type ExecImpl struct {
	public any // Back-link to canonical *graph.Exec wrapper.

	name    string
	backend compute.Backend

	distStrategy     distributed.Strategy
	meshes           []*distributed.DeviceMesh
	deviceAssignment []compute.DeviceNum
	numDevices       int

	graphFn CanonicalExecGraphFn

	numInputs, numOutputs int

	inputAsSlice, outputAsSlice bool
	inputIsGraph                bool
	inputShardingSpecs          []*distributed.ShardingSpec
	outputShardingSpecs         []*distributed.ShardingSpec

	maxCacheSize int
	dynamicAxes  [][]string

	setSideParams SideParamsFn
	loggerFn      LoggerFn

	cacheMu sync.Mutex
	cache   []*execGraphCacheEntry
	pending []*pendingCompilation

	finalized  atomic.Bool
	muInFlight sync.RWMutex
}

type pendingCompilation struct {
	argsShapes []shapes.Shape
	done       chan struct{}
	entry      *execGraphCacheEntry
	err        error
}

type execGraphCacheEntry struct {
	argsShapes        []shapes.Shape
	graph             *GraphImpl
	numGraphFnOutputs int
	numAllOutputs     int
	loggedMessages    []string
	loggedNodeIDs     []NodeId
}

// NewExecCanonicalImpl constructs a new ExecImpl.
func NewExecCanonicalImpl(backend compute.Backend, graphFn CanonicalExecGraphFn, numInputs, numOutputs int, inputIsGraph, inputAsSlice, outputAsSlice bool) (*ExecImpl, error) {
	return &ExecImpl{
		backend:       backend,
		name:          "Exec",
		graphFn:       graphFn,
		maxCacheSize:  DefaultExecMaxCacheSize,
		loggerFn:      DefaultNodeLogger,
		numDevices:    1,
		numInputs:     numInputs,
		numOutputs:    numOutputs,
		inputIsGraph:  inputIsGraph,
		inputAsSlice:  inputAsSlice,
		outputAsSlice: outputAsSlice,
	}, nil
}

// Public returns the canonical *graph.Exec wrapper (stored as any), or nil.
func (e *ExecImpl) Public() any {
	if e == nil {
		return nil
	}
	return e.public
}

// SetPublic records the canonical *graph.Exec wrapper on e.
func (e *ExecImpl) SetPublic(pub any) {
	e.public = pub
}

// CacheLen returns the current number of cached compiled graphs (used in tests).
func (e *ExecImpl) CacheLen() int {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()
	return len(e.cache)
}

func (e *ExecImpl) WithName(name string) *ExecImpl {
	e.name = name
	return e
}

func (e *ExecImpl) WithDynamicAxes(axisNames ...[]string) *ExecImpl {
	e.dynamicAxes = make([][]string, len(axisNames))
	for i, names := range axisNames {
		e.dynamicAxes[i] = slices.Clone(names)
	}
	return e
}

func (e *ExecImpl) WithDeviceAssignment(devices []compute.DeviceNum) *ExecImpl {
	e.deviceAssignment = devices
	return e
}

func (e *ExecImpl) DeviceAssignment() []compute.DeviceNum {
	return e.deviceAssignment
}

func (e *ExecImpl) DistributionStrategy() distributed.Strategy {
	return e.distStrategy
}

func (e *ExecImpl) SPMD(mesh *distributed.DeviceMesh) *ExecImpl {
	e.distStrategy = distributed.SPMD
	e.meshes = []*distributed.DeviceMesh{mesh}
	e.numDevices = mesh.NumDevices()
	return e
}

func (e *ExecImpl) AutoSharding(meshes ...*distributed.DeviceMesh) *ExecImpl {
	e.distStrategy = distributed.AutoSharding
	e.meshes = slices.Clone(meshes)
	e.numDevices = 0
	for _, mesh := range meshes {
		e.numDevices = max(e.numDevices, mesh.NumDevices())
	}
	return e
}

func (e *ExecImpl) Meshes() []*distributed.DeviceMesh {
	return e.meshes
}

func (e *ExecImpl) WithInputShardingSpecs(specs ...*distributed.ShardingSpec) *ExecImpl {
	e.inputShardingSpecs = specs
	return e
}

func (e *ExecImpl) WithOutputShardingSpecs(specs ...*distributed.ShardingSpec) *ExecImpl {
	e.outputShardingSpecs = specs
	return e
}

func (e *ExecImpl) NumDevices() int {
	return e.numDevices
}

func (e *ExecImpl) Name() string {
	return e.name
}

func (e *ExecImpl) SetMaxCache(maxCacheSize int) *ExecImpl {
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()
	e.maxCacheSize = maxCacheSize
	return e
}

func (e *ExecImpl) SetSideParamsHook(fn SideParamsFn) *ExecImpl {
	e.setSideParams = fn
	return e
}

func (e *ExecImpl) SetNodeLogger(loggerFn LoggerFn) {
	e.loggerFn = loggerFn
}

func (e *ExecImpl) GetNodeLogger() LoggerFn {
	return e.loggerFn
}

func (e *ExecImpl) Call(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.compileAndExecute(true, 0, args...)
	return outputs, err
}

func (e *ExecImpl) CallOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.compileAndExecute(true, defaultDevice, args...)
	return outputs, err
}

func (e *ExecImpl) CallWithGraph(args ...any) ([]*tensors.Tensor, *GraphImpl, error) {
	return e.compileAndExecute(true, 0, args...)
}

func (e *ExecImpl) CallWithGraphOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, *GraphImpl, error) {
	outputs, g, err := e.compileAndExecute(true, defaultDevice, args...)
	if err != nil {
		return nil, nil, err
	}
	return outputs, g, nil
}

func (e *ExecImpl) DistributedCall(args ...any) ([]*dtensor.Tensor, error) {
	outputs, err := e.Call(args...)
	if err != nil {
		return nil, err
	}
	return e.AggregateShards(outputs)
}

func (e *ExecImpl) AggregateShards(shards []*tensors.Tensor) ([]*dtensor.Tensor, error) {
	numDevices := e.NumDevices()
	if numDevices == 1 {
		return nil, errors.New("distributed execution requires more than one device")
	}
	if len(e.meshes) == 0 {
		return nil, errors.New("meshes are not set for distributed execution")
	}
	defaultMesh := e.meshes[0]
	replicatedSpec := distributed.NewReplicatedShardingSpec(defaultMesh)
	numOutputs := len(shards) / numDevices
	distributedOutputs := make([]*dtensor.Tensor, numOutputs)
	var err error
	for outputIdx := range numOutputs {
		outputShards := make([]*tensors.Tensor, numDevices)
		for deviceIdx := range numDevices {
			outputShards[deviceIdx] = shards[deviceIdx*numOutputs+outputIdx]
		}
		shardingSpec := replicatedSpec
		if outputIdx < len(e.outputShardingSpecs) {
			shardingSpec = e.outputShardingSpecs[outputIdx]
		}
		distributedOutputs[outputIdx], err = dtensor.NewTensor(shardingSpec, outputShards)
		if err != nil {
			return nil, err
		}
	}
	return distributedOutputs, nil
}

func unwrapListOfTensors(args []any) []any {
	if len(args) != 1 {
		return args
	}
	if list, ok := args[0].([]*tensors.Tensor); ok {
		return xslices.Map(list, func(x *tensors.Tensor) any { return x })
	}
	return args
}

func unwrapDistributedTensors(numDevices int, args []any) ([]any, error) {
	if len(args) == 0 {
		return args, nil
	}
	if _, ok := args[0].(*dtensor.Tensor); !ok {
		return args, nil
	}
	unwrapped := make([]any, numDevices*len(args))
	for argIdx, arg := range args {
		dTensor, ok := arg.(*dtensor.Tensor)
		if !ok {
			return nil, errors.Errorf(
				"argument #%d is not a dtensor.Tensor -- if using dtensor.Tensor as input, all arguments "+
					"must be dtensor.Tensors, or none of them", argIdx)
		}
		shards := dTensor.Shards()
		if len(shards) != numDevices {
			return nil, errors.Errorf(
				"dtensor.Tensor #%d has %d shards, but graph has %d devices and those many shards  are expected",
				argIdx, len(shards), numDevices)
		}
		for shardIdx, shard := range shards {
			unwrapped[shardIdx*len(args)+argIdx] = shard
		}
	}
	return unwrapped, nil
}

func (e *ExecImpl) validateArgs(args []any) error {
	if len(args) == 0 {
		if !e.inputAsSlice && e.numInputs != 0 {
			return errors.Errorf("no arguments passed to Exec for %q, but graphFn expects %d inputs",
				e.Name(), e.numInputs)
		}
		return nil
	}
	numDevices := e.numDevices
	if numDevices > 1 && len(args)%numDevices != 0 {
		return errors.Errorf("number of arguments for execution (%d) doesn't match number of devices (%d) for %q",
			len(args), numDevices, e.Name())
	}
	numArgsPerDevice := len(args) / numDevices
	if !e.inputAsSlice && numArgsPerDevice != e.numInputs {
		return errors.Errorf(
			"# of arguments to call (#args=%d, %d per device) don't match # arguments to the graph function "+
				"(#args=%d) for %q",
			len(args), numArgsPerDevice, e.numInputs, e.Name())
	}
	return nil
}

func (e *ExecImpl) convertArgsToBuffers(args []any, defaultDevice compute.DeviceNum) (
	argsAsBuffers []compute.Buffer, argsShapes []shapes.Shape, argsDonate []bool, err error) {
	argsAsBuffers = make([]compute.Buffer, len(args))
	argsShapes = make([]shapes.Shape, len(args))
	argsDonate = make([]bool, len(args))
	numDevices := e.numDevices
	argsPerDevice := len(args) / numDevices
	var argIdx int
	argDeviceNum := defaultDevice
	for argDeviceIdx := range numDevices {
		if numDevices > 1 {
			if len(e.deviceAssignment) <= argDeviceIdx {
				argDeviceNum = compute.DeviceNum(argDeviceIdx)
			} else {
				argDeviceNum = e.deviceAssignment[argDeviceIdx]
			}
		}
		for range argsPerDevice {
			arg := args[argIdx]
			err := exceptions.TryCatch[error](func() {
				argsAsBuffers[argIdx], argsShapes[argIdx], argsDonate[argIdx] = anyToDeviceBuffer(
					e.backend,
					argDeviceNum,
					arg,
				)
			})
			if err != nil {
				return nil, nil, nil, errors.WithMessagef(
					err, "Failed to convert argument #%d of %d to device(%d) -- type %T: %v",
					argIdx, len(args), argDeviceNum, args[argIdx], args[argIdx])
			}
			argIdx++
		}
	}
	return argsAsBuffers, argsShapes, argsDonate, nil
}

func (e *ExecImpl) expandArgsToTotalParams(g *GraphImpl, argsAsBuffers []compute.Buffer, argsDonate []bool) (
	[]compute.Buffer, []bool) {
	newParamsPerDevice := g.NumParameters()
	totalParams := newParamsPerDevice * e.numDevices
	if totalParams <= len(argsAsBuffers) {
		return argsAsBuffers, argsDonate
	}

	oldParamsPerDevice := len(argsAsBuffers) / e.numDevices
	numNew := totalParams - len(argsAsBuffers)
	argsAsBuffers = slices.Grow(argsAsBuffers, numNew)
	argsAsBuffers = argsAsBuffers[:totalParams]
	argsDonate = slices.Grow(argsDonate, numNew)
	argsDonate = argsDonate[:totalParams]

	for deviceIdx := e.numDevices - 1; deviceIdx >= 0; deviceIdx-- {
		oldStart := deviceIdx * oldParamsPerDevice
		newStart := deviceIdx * newParamsPerDevice
		copy(argsAsBuffers[newStart:newStart+oldParamsPerDevice], argsAsBuffers[oldStart:oldStart+oldParamsPerDevice])
		copy(argsDonate[newStart:newStart+oldParamsPerDevice], argsDonate[oldStart:oldStart+oldParamsPerDevice])
		for i := newStart + oldParamsPerDevice; i < newStart+newParamsPerDevice; i++ {
			argsAsBuffers[i] = nil
			argsDonate[i] = false
		}
	}
	return argsAsBuffers, argsDonate
}

func (e *ExecImpl) Compile(inputShapes ...shapes.Shape) (*GraphImpl, error) {
	if e.finalized.Load() {
		return nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}
	e.muInFlight.RLock()
	defer e.muInFlight.RUnlock()
	if e.finalized.Load() {
		return nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}
	entry, err := e.compile(inputShapes...)
	if err != nil {
		return nil, err
	}
	return entry.graph, nil
}

func (e *ExecImpl) CompileStandalone(inputShapes ...shapes.Shape) (*GraphImpl, error) {
	if e.finalized.Load() {
		return nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}
	e.muInFlight.RLock()
	defer e.muInFlight.RUnlock()
	if e.finalized.Load() {
		return nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}
	templateShapes, err := e.makeTemplateShapes(inputShapes)
	if err != nil {
		return nil, err
	}
	entry, err := e.buildAndCompileGraph(templateShapes, -1)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed during standalone graph construction/compilation")
	}
	return entry.graph, nil
}

func (e *ExecImpl) compile(inputShapes ...shapes.Shape) (*execGraphCacheEntry, error) {
	entry, err := e.findOrCreateGraph(inputShapes)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed during graph construction/compilation")
	}
	if entry == nil {
		return nil, errors.Errorf(
			"maximum cache size of %d reached for %q, cannot create another graph -- "+
				"a new computation graph needs to be created+compiled for each different shape of "+
				"the input, consider using padding, or if this is not a concern change "+
				"the cache size with executable.SetMaxCache()", e.maxCacheSize, e.Name())
	}
	return entry, nil
}

func (e *ExecImpl) compileAndExecute(execute bool, defaultDevice compute.DeviceNum, args ...any) (
	[]*tensors.Tensor, *GraphImpl, error) {
	if e.finalized.Load() {
		return nil, nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}
	e.muInFlight.RLock()
	defer e.muInFlight.RUnlock()
	if e.finalized.Load() {
		return nil, nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.Name())
	}

	args = unwrapListOfTensors(args)
	var err error
	args, err = unwrapDistributedTensors(e.numDevices, args)
	if err != nil {
		return nil, nil, err
	}
	err = e.validateArgs(args)
	if err != nil {
		return nil, nil, err
	}
	numDevices := e.numDevices
	argsAsBuffers, argsShapes, argsDonate, err := e.convertArgsToBuffers(args, defaultDevice)
	if err != nil {
		return nil, nil, err
	}

	entry, err := e.compile(argsShapes...)
	if err != nil {
		return nil, nil, err
	}
	g := entry.graph

	argsAsBuffers, argsDonate = e.expandArgsToTotalParams(g, argsAsBuffers, argsDonate)

	if e.setSideParams != nil {
		err := e.setSideParams(g, argsAsBuffers, argsDonate)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "failed to set side parameters for graph %q", g.Name())
		}
	}

	numParams := g.NumParameters()
	for ii, t := range argsAsBuffers {
		if t == nil {
			paramIdx := ii % numParams
			deviceIdx := ii / g.NumParameters()
			if numDevices == 1 {
				return nil, nil, errors.Errorf("parameter #%d (%q) is nil or invalid, maybe a variable value not set as a "+
					"parameter, cannot execute the graph %q",
					paramIdx, g.GetParameterByHandle(ParameterHandle(paramIdx)).GetParameterName(), g.Name())
			}
			return nil, nil, errors.Errorf("parameter #%d (%q) for device #%d is nil or invalid, maybe a variable value not set as a "+
				"parameter, cannot execute the graph %q",
				paramIdx, g.GetParameterByHandle(ParameterHandle(paramIdx)).GetParameterName(), deviceIdx, g.Name())
		}
	}

	if !execute {
		return nil, g, nil
	}

	var outputs []*tensors.Tensor
	err = exceptions.TryCatch[error](func() {
		outputs = g.RunWithBuffers(argsAsBuffers, argsDonate, defaultDevice)
	})
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to execute graph %q", g.Name())
	}
	if len(outputs) != entry.numAllOutputs*e.numDevices {
		return nil, nil, errors.Errorf("expected %d * %d (numDevices) = %d outputs from graph %q, got %d",
			e.numOutputs, e.numDevices, e.numOutputs*e.numDevices, g.Name(), len(outputs))
	}

	numGraphFnOutputs := entry.numAllOutputs - len(entry.loggedMessages)
	if e.loggerFn != nil {
		var loggerOutputs []*tensors.Tensor
		if len(entry.loggedMessages) > 0 {
			loggerOutputs = outputs[numGraphFnOutputs:entry.numAllOutputs]
		}
		e.loggerFn(g, entry.loggedMessages, loggerOutputs, entry.loggedNodeIDs)
	}

	if len(entry.loggedMessages) == 0 {
		return outputs, g, nil
	}
	if numDevices == 1 {
		return outputs[:numGraphFnOutputs], g, nil
	}
	graphFnOutputs := make([]*tensors.Tensor, numDevices*numGraphFnOutputs)
	for deviceIdx := range numDevices {
		copy(graphFnOutputs[deviceIdx*numGraphFnOutputs:(deviceIdx+1)*numGraphFnOutputs],
			outputs[deviceIdx*entry.numAllOutputs:deviceIdx*entry.numAllOutputs+numGraphFnOutputs])
	}
	return graphFnOutputs, g, nil
}

func (e *ExecImpl) buildAndCompileGraph(argsShapes []shapes.Shape, cacheIndex int) (*execGraphCacheEntry, error) {
	entry := &execGraphCacheEntry{graph: NewGraphImpl(e.backend, fmt.Sprintf("%s#%d", e.name, cacheIndex))}
	g := entry.graph
	switch e.distStrategy {
	case distributed.AutoSharding:
		if err := g.SetAutoSharding(e.meshes...); err != nil {
			return nil, errors.WithMessagef(err, "failed to assign AutoSharding for graph %q", g.Name())
		}
	case distributed.SPMD:
		if err := g.SetSPMD(e.meshes[0]); err != nil {
			return nil, errors.WithMessagef(err, "failed to assign SPMD for graph %q", g.Name())
		}
	case distributed.None:
	}
	if e.deviceAssignment != nil {
		if err := g.SetDeviceAssignment(e.deviceAssignment); err != nil {
			return nil, errors.WithMessagef(err, "failed to assign device assignment %v for graph %q",
				e.deviceAssignment, g.Name())
		}
	}

	argsShapesToUse := argsShapes
	if e.numDevices > 1 && (e.distStrategy == distributed.AutoSharding || e.distStrategy == distributed.SPMD) {
		numArgsPerDevice := len(argsShapes) / e.numDevices
		argsShapesToUse = argsShapes[:numArgsPerDevice]
	}

	args := make([]*NodeImpl, 0, len(argsShapesToUse))
	for ii, shape := range argsShapesToUse {
		var spec *distributed.ShardingSpec
		if ii < len(e.inputShardingSpecs) {
			spec = e.inputShardingSpecs[ii]
		} else if e.inputAsSlice && len(e.inputShardingSpecs) > 0 {
			spec = e.inputShardingSpecs[len(e.inputShardingSpecs)-1]
		}
		if spec != nil && e.distStrategy == distributed.AutoSharding {
			shape = spec.LogicalShapeForShard(shape)
		}
		arg := ShardedParameter(g, fmt.Sprintf("arg%d", ii), shape, spec)
		args = append(args, arg)
	}

	var outputs []*NodeImpl
	err := exceptions.TryCatch[error](func() {
		start := time.Now()
		outputs = e.graphFn(g, args)
		if klog.V(1).Enabled() {
			elapsed := time.Since(start)
			klog.Infof("Build graph time for %q: %s", e.name, elapsed)
		}
	})
	if err != nil {
		return nil, errors.WithMessagef(err, "graphFn for %q panicked", e.Name())
	}

	if e.outputAsSlice {
		entry.numGraphFnOutputs = len(outputs)
	} else {
		entry.numGraphFnOutputs = e.numOutputs
		if len(outputs) != e.numOutputs {
			return nil, errors.Errorf(
				"graphFn for %q returned %d results, as opposed to the expected %d results",
				e.Name(), len(outputs), e.numOutputs)
		}
	}

	loggedNodes := g.LoggedNodes()
	entry.loggedMessages = make([]string, 0, len(loggedNodes))
	entry.loggedNodeIDs = make([]NodeId, 0, len(loggedNodes))
	for _, node := range loggedNodes {
		outputs = append(outputs, node)
		entry.loggedMessages = append(entry.loggedMessages, node.LogMessage())
		entry.loggedNodeIDs = append(entry.loggedNodeIDs, node.Id())
	}

	outputShardingSpecs := slices.Clone(e.outputShardingSpecs)
	if len(outputShardingSpecs) > 0 {
		if len(outputShardingSpecs) < entry.numGraphFnOutputs {
			lastSpec := outputShardingSpecs[len(outputShardingSpecs)-1]
			for len(outputShardingSpecs) < entry.numGraphFnOutputs {
				outputShardingSpecs = append(outputShardingSpecs, lastSpec)
			}
		}
		for range len(loggedNodes) {
			outputShardingSpecs = append(outputShardingSpecs, nil)
		}
	}

	err = g.CompileWithSharding(outputs, outputShardingSpecs)
	if err != nil {
		return nil, err
	}
	entry.argsShapes = make([]shapes.Shape, len(argsShapes))
	copy(entry.argsShapes, argsShapes)
	entry.numAllOutputs = len(outputs)
	return entry, nil
}

func shapesMatch(a, b []shapes.Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i, s := range a {
		if !s.Equal(b[i]) {
			return false
		}
	}
	return true
}

func (e *ExecImpl) lockedFindInCache(argsShapes []shapes.Shape) *execGraphCacheEntry {
	for _, entry := range e.cache {
		if shapesMatch(argsShapes, entry.argsShapes) {
			return entry
		}
	}
	return nil
}

func (e *ExecImpl) lockedFindPending(argsShapes []shapes.Shape) *pendingCompilation {
	for _, pc := range e.pending {
		if shapesMatch(argsShapes, pc.argsShapes) {
			return pc
		}
	}
	return nil
}

func (e *ExecImpl) makeTemplateShapes(concreteShapes []shapes.Shape) ([]shapes.Shape, error) {
	if len(e.dynamicAxes) == 0 {
		return concreteShapes, nil
	}
	numDevices := e.numDevices
	if numDevices == 0 {
		numDevices = 1
	}
	argsPerDevice := len(concreteShapes) / numDevices
	templateShapes := make([]shapes.Shape, len(concreteShapes))
	for devIdx := range numDevices {
		for argIdx := range argsPerDevice {
			concreteIdx := devIdx*argsPerDevice + argIdx
			shape := concreteShapes[concreteIdx]
			if argIdx < len(e.dynamicAxes) && len(e.dynamicAxes[argIdx]) > 0 {
				axisNames := e.dynamicAxes[argIdx]
				if len(axisNames) != shape.Rank() {
					return nil, errors.Errorf("Exec.WithDynamicAxes: rank mismatch for argument %d: expected rank %d (from dynamic axes), got %d (from input shape %s)", argIdx, len(axisNames), shape.Rank(), shape)
				}
				dims := make([]int, shape.Rank())
				for axis, name := range axisNames {
					if name != "" {
						dims[axis] = shapes.DynamicDim
					} else {
						dims[axis] = shape.Dimensions[axis]
					}
				}
				templateShapes[concreteIdx] = shapes.MakeDynamic(shape.DType, dims, axisNames)
			} else {
				templateShapes[concreteIdx] = shape
			}
		}
	}
	return templateShapes, nil
}

func (e *ExecImpl) findOrCreateGraph(argsShapes []shapes.Shape) (*execGraphCacheEntry, error) {
	templateShapes, err := e.makeTemplateShapes(argsShapes)
	if err != nil {
		return nil, err
	}

	e.cacheMu.Lock()

	if e.finalized.Load() {
		e.cacheMu.Unlock()
		return nil, errors.Wrapf(ErrExecFinalized, "Exec %q is finalized", e.name)
	}

	if entry := e.lockedFindInCache(templateShapes); entry != nil {
		e.cacheMu.Unlock()
		return entry, nil
	}

	if pc := e.lockedFindPending(templateShapes); pc != nil {
		e.cacheMu.Unlock()
		<-pc.done
		if pc.err != nil {
			return nil, errors.WithMessagef(pc.err, "compilation failed for Exec %q", e.name)
		}
		return pc.entry, nil
	}

	if e.maxCacheSize > 0 && len(e.cache)+len(e.pending) >= e.maxCacheSize {
		e.cacheMu.Unlock()
		return nil, nil
	}

	pc := &pendingCompilation{argsShapes: templateShapes, done: make(chan struct{})}
	e.pending = append(e.pending, pc)
	cacheIndex := len(e.cache) + len(e.pending)
	e.cacheMu.Unlock()

	entry, err := e.buildAndCompileGraph(templateShapes, cacheIndex)
	pc.entry = entry
	pc.err = err

	e.cacheMu.Lock()
	if entry != nil {
		e.cache = append(e.cache, entry)
	}
	e.pending = slices.DeleteFunc(e.pending, func(p *pendingCompilation) bool { return p == pc })
	e.cacheMu.Unlock()

	close(pc.done)

	if err != nil {
		return nil, errors.WithMessagef(err, "compilation failed for Exec %q", e.name)
	}
	return entry, nil
}

func (e *ExecImpl) Finalize() {
	if e.finalized.Swap(true) {
		return
	}
	e.muInFlight.Lock()
	defer e.muInFlight.Unlock()

	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()

	for _, entry := range e.cache {
		entry.graph.Finalize()
		entry.graph = nil
	}
	e.cache = e.cache[:0]
	e.pending = nil
}

// DefaultNodeLogger logs marked nodes during execution.
//
//nolint:forbidigo // Intentionally prints to stdout.
func DefaultNodeLogger(g *GraphImpl, messages []string, values []*tensors.Tensor, nodes []NodeId) {
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
