// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

//=======================================================================================================
//
// Note: This file contains various aliases for execution.
// We separate them from the exec.go file to make it easier to separate the core logic (in exec.go)
// from ergonomics (execaliases.go, this file).
//
//=======================================================================================================

// Call executes the computation with the given arguments.
//
// It the input arguments shape has never been seen before, it JIT-compiles a new computation graph for that shape,
// which can take a while, but is cached and later executions are very fast.
//
// The arguments are first all converted to tensors where needed.
//
// Optionally, use DonateTensorBuffer(value) to mark a tensor as a value to be "donated" to the execution (and potentially save some space).
// See details in DonateTensorBuffer.
//
// It returns the outputs in a slice, even if there is only one output, or an error if it fails. See Exec1-Exec4 for
// aliases that return some exact number of outputs.
//
// Errors (with full stack-traces) are returned on failure.
func (e *Exec) Call(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.CallWithGraph(args...)
	return outputs, err
}

// Exec is a deprecated alias for Exec.Call.
//
// Deprecated: use Exec.Call instead.
func (e *Exec) Exec(args ...any) ([]*tensors.Tensor, error) {
	return e.Call(args...)
}

// DistributedCall is just like Exec, but aggregates the outputs into [dtensor.Tensor].
// Usually, Exec will return a slice of numDevices * numOutputs individual shards ([tensors.Tensor]).
//
// Notice that to actually trigger distributed execution, you must set Exec.AutoSharding (or Exec.SPMD) and
// set the proper sharding specs of the inputs and outputs.
//
// See also Exec.AggregateShards.
func (e *Exec) DistributedCall(args ...any) ([]*dtensor.Tensor, error) {
	shards, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, err
	}
	return e.AggregateShards(shards)
}

// DistributedExec is a deprecated alias for Exec.DistributedCall.
//
// Deprecated: use Exec.DistributedCall instead.
func (e *Exec) DistributedExec(args ...any) ([]*dtensor.Tensor, error) {
	return e.DistributedCall(args...)
}

// AggregateShards returned by Exec into [dtensor.Tensor].
// Usually, Exec will return a slice of numDevices * numOutputs individual shards ([tensors.Tensor]).
//
// See also DistributedExec, which calls Exec and then calls this.
func (e *Exec) AggregateShards(shards []*tensors.Tensor) ([]*dtensor.Tensor, error) {
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
		var shardingSpec = replicatedSpec
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

// CallWithGraph is similar to Exec, but it also returns the computation graph used
// in the call.
//
// Exec creates different computation graphs when the inputs' shapes change,
// so different calls may return different graphs.
//
// It returns the outputs in a slice, even if there is only one output. It also returns the computation graph used.
//
// Distributed execution: if the Exec was configured with AutoSharding(meshes..) or SPMD(mesh),
// then it requires the input values for each device used in the execution.
// So if there are D devices, and I inputs, it required D*I args, organized in a
// "device-major" list (all the inputs to the first device, then the inputs for the second device, and so on).
// Alternatively, you can provide I args of [dtensor.Tensor] (they already include one value per device),
// matching the DistributedMesh provided to Exec.SPMD.
//
// Errors (with full stack-traces) are returned on failure.
func (e *Exec) CallWithGraph(args ...any) ([]*tensors.Tensor, *Graph, error) {
	return e.ExecWithGraphOnDevice(compute.DeviceNum(0), args...)
}

// ExecWithGraph is a deprecated alias for Exec.CallWithGraph.
//
// Deprecated: use CallWithGraph instead.
func (e *Exec) ExecWithGraph(args ...any) ([]*tensors.Tensor, *Graph, error) {
	return e.CallWithGraph(args...)
}

// CallOnDevice behaves like Exec but for portable computations uses the given device for execution.
//
// deafultDevice is used for single-device computations that are portable (no fixed device assignment set
// WithDeviceAssignment). Otherwise, it is ignored.
func (e *Exec) CallOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.CallWithGraph(args...)
	return outputs, err
}

// ExecOnDevice is a deprecated alias for Exec.CallOnDevice.
//
// Deprecated: use CallOnDevice instead.
func (e *Exec) ExecOnDevice(defaultDevice compute.DeviceNum, args ...any) ([]*tensors.Tensor, error) {
	return e.CallOnDevice(defaultDevice, args...)
}

// MustNewExecCanonical constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// `graphFn` must be of type CanonicalExecGraphFn.
//
// It panics if the inputs are invalid.
//
// See also the generics MustNewExec (or MustNewExec for returning an error), which checks for valid graphFn in compile time.
func MustNewExecCanonical(backend compute.Backend, graphFn CanonicalExecGraphFn, numInputs, numOutputs int, inputIsGraph, inputAsSlice, outputAsSlice bool) *Exec {
	return must.M1(NewExecCanonical(backend, graphFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice))
}

// MustNewExec constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// graphFn should take *Node as input and return a *Node -- except if there are no (Node) inputs,
// in which case it should take a single *Graph input.
//
// It's a wrapper for MustNewExecCanonical, but uses generics to type check that
// graphFn is valid.
func MustNewExec[F ExecGraphFn](backend compute.Backend, graphFn F) *Exec {
	return must.M1(NewExec(backend, graphFn))
}

// ExecOnce builds the graph and calls it with the given arguments and returns the one output.
//
// It's short for a call to MustNewExec, Exec.Exec and Exec.Finalize for functions that return only one output.
func ExecOnce[F ExecGraphFnOneOutput](backend compute.Backend, graphFn F, args ...any) (*tensors.Tensor, error) {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	results, err := e.Call(args...)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// ExecOnceN builds the graph and calls it with the given arguments and returns the various outputs.
//
// It's short for a call to MustNewExec, Exec.Exec and Exec.Finalize for functions that return only one output.
//
// See ExecOnce for a more convenient version if you have only one output.
// Also, see MustExecOnceN or MustExecOnce if you want it to panic on error.
func ExecOnceN[F ExecGraphFn](backend compute.Backend, graphFn F, args ...any) ([]*tensors.Tensor, error) {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	results, err := e.Call(args...)
	if err != nil {
		return nil, err
	}
	return results, nil
}

// MustExecOnce builds the graph and calls it with the given arguments and returns the one output.
//
// It's short for a call to MustNewExec, Exec.MustExec and Exec.Finalize for functions that return only one output.
// It panics on error.
//
// See MustExecOnceN if you have multiple outputs.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func MustExecOnce[F ExecGraphFnOneOutput](backend compute.Backend, graphFn F, args ...any) *tensors.Tensor {
	return MustExecOnceN(backend, graphFn, args...)[0]
}

// MustExecOnceN builds the graph and calls it with the given arguments, and returns various outputs.
//
// It's short for a call to MustNewExec, Exec.MustExec and Exec.Finalize.
// It panics on error.
//
// See MustExecOnce for a more convenient version if you have only one output.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func MustExecOnceN[F ExecGraphFn](backend compute.Backend, graphFn F, args ...any) []*tensors.Tensor {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	return e.MustCall(args...)
}

// MustCall calls the computation with the given arguments.
//
// It the input arguments shape has never been seen before, it JIT-compiles a new computation graph for that shape,
// which can take a while, but is cached and later executions are very fast.
//
// The arguments are first all converted to tensors, if they not yet tensors.
//
// Optionally, use DonateTensorBuffer(value) to mark a tensor as a value to be "donated" to the execution (and potentially save some space).
// See details in DonateTensorBuffer.
//
// It returns the outputs in a slice, even if there is only one output, or an error if it fails. See Exec1-Exec4 for
// aliases that return some exact number of outputs.
//
// It panics on errors (with full stack-traces).
func (e *Exec) MustCall(args ...any) []*tensors.Tensor {
	results, _ := e.MustCallWithGraph(args...)
	return results
}

// MustCallWithGraph is similar to MustExec, but it also returns the computation graph used
// in the call.
//
// The underlying Exec creates different computation graphs when the inputs' shapes change,
// so different calls may return different graphs.
//
// It returns the outputs in a slice, even if there is only one output. It also returns the computation graph used.
//
// It panics on errors (with full stack-traces).
func (e *Exec) MustCallWithGraph(args ...any) ([]*tensors.Tensor, *Graph) {
	results, g, err := e.CallWithGraph(args...)
	if err != nil {
		panic(err)
	}
	return results, g
}

// Call1 calls the compiled the graph with the given arguments and returns one output.
//
// It returns an error if the graph doesn't return exactly one output.
//
// See Callfor more details.
func (e *Exec) Call1(args ...any) (*tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, err
	}
	if len(results) != 1 {
		return nil, errors.Errorf(
			"graph %q returned %d results, as opposed to exactly one as expected by Exec1",
			e.Name(),
			len(results),
		)
	}
	return results[0], nil
}

// Call2 calls the compiled graph with the given arguments and returns two outputs.
//
// It returns an error if the graph doesn't return exactly two outputs.
//
// See Call for more details.
func (e *Exec) Call2(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, nil, err
	}
	if len(results) != 2 {
		return nil, nil, errors.Errorf(
			"graph %q returned %d results, as opposed to exactly two as expected by Exec2",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], nil
}

// Call3 calls the compiled graph with the given arguments and returns three outputs.
//
// It returns an error if the graph doesn't return exactly three outputs.
//
// See Call for more details.
func (e *Exec) Call3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(results) != 3 {
		return nil, nil, nil, errors.Errorf(
			"graph %q returned %d results, as opposed to exactly three as expected by Exec3",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2], nil
}

// Call4 calls the compiled graph with the given arguments and returns four outputs.
//
// It returns an error if the graph doesn't return exactly four outputs.
//
// See Exec for more details.
func (e *Exec) Call4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if len(results) != 4 {
		return nil, nil, nil, nil, errors.Errorf(
			"graph %q returned %d results, as opposed to exactly four as expected by Exec4",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2], results[3], nil
}

// MustCall1 executes the graph with the given arguments and returns one output.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly one output.
//
// See MustExec for more details.
func (e *Exec) MustCall1(args ...any) *tensors.Tensor {
	results := e.MustCall(args...)
	if len(results) != 1 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly one as expected by MustExec1",
			e.Name(),
			len(results),
		)
	}
	return results[0]
}

// MustCall2 executes the graph with the given arguments and returns two outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly two outputs.
//
// See MustExec for more details.
func (e *Exec) MustCall2(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	results := e.MustCall(args...)
	if len(results) != 2 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly two as expected by MustExec2",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1]
}

// MustCall3 executes the graph with the given arguments and returns three outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly three outputs.
//
// See MustExec for more details.
func (e *Exec) MustCall3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.MustCall(args...)
	if len(results) != 3 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly three as expected by MustExec3",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2]
}

// MustCall4 executes the graph with the given arguments and returns four outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly four outputs.
//
// See MustExec for more details.
func (e *Exec) MustCall4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.MustCall(args...)
	if len(results) != 4 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly four as expected by MustExec4",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2], results[3]
}

// MustExec is an alias to the corresponding MustCall method.
//
// Deprecated: use the MustCall method instead.
func (e *Exec) MustExec(args ...any) []*tensors.Tensor {
	return e.MustCall(args...)
}

// MustExecWithGraph is an alias to the corresponding MustCallWithGraph method.
//
// Deprecated: use the MustCallWithGraph method instead.
func (e *Exec) MustExecWithGraph(args ...any) ([]*tensors.Tensor, *Graph) {
	return e.MustCallWithGraph(args...)
}

// Exec1 is an alias to the corresponding Call1 method.
//
// Deprecated: use the Call1 method instead.
func (e *Exec) Exec1(args ...any) (*tensors.Tensor, error) {
	return e.Call1(args...)
}

// Exec2 is an alias to the corresponding Call2 method.
//
// Deprecated: use the Call2 method instead.
func (e *Exec) Exec2(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	return e.Call2(args...)
}

// Exec3 is an alias to the corresponding Call3 method.
//
// Deprecated: use the Call3 method instead.
func (e *Exec) Exec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	return e.Call3(args...)
}

// Exec4 is an alias to the corresponding Call4 method.
//
// Deprecated: use the Call4 method instead.
func (e *Exec) Exec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	return e.Call4(args...)
}

// MustExec1 is an alias to the corresponding MustCall1 method.
//
// Deprecated: use the MustCall1 method instead.
func (e *Exec) MustExec1(args ...any) *tensors.Tensor {
	return e.MustCall1(args...)
}

// MustExec2 is an alias to the corresponding MustCall2 method.
//
// Deprecated: use the MustCall2 method instead.
func (e *Exec) MustExec2(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	return e.MustCall2(args...)
}

// MustExec3 is an alias to the corresponding MustCall3 method.
//
// Deprecated: use the MustCall3 method instead.
func (e *Exec) MustExec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return e.MustCall3(args...)
}

// MustExec4 is an alias to the corresponding MustCall4 method.
//
// Deprecated: use the MustCall4 method instead.
func (e *Exec) MustExec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return e.MustCall4(args...)
}
