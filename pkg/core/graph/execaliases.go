package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

//=======================================================================================================
//
// Note: This file contains various aliases for execution.
// We separate them from the exec.go file to make it easier to separate the core logic (in exec.go)
// from ergonomics (execaliases.go, this file).
//
//=======================================================================================================

// Exec executes the computation with the given arguments.
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
func (e *Exec) Exec(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.ExecWithGraph(args...)
	return outputs, err
}

// DistributedExec is just like Exec, but aggregates the outputs into *distributed.Tensor.
// Usually, Exec will return alice of numDevices * nnumOutputs individual shards (*tensors.Tensor).
//
// Notice that to actually trigger distributed execution, you must set Exec.AutoSharding (or Exec.SPMD) and
// set the proper sharding specs of the inputs and outputs.
//
// See also Exec.AggregateShards.
func (e *Exec) DistributedExec(args ...any) ([]*distributed.Tensor, error) {
	shards, _, err := e.ExecWithGraph(args...)
	if err != nil {
		return nil, err
	}
	return e.AggregateShards(shards)
}

// AggregateShards returned by Exec into *distributedTensor outputs.
// Usually, Exec will return alice of numDevices * nnumOutputs individual shards (*tensors.Tensor).
//
// See also DistributedExec, which calls Exec and then calls this.
func (e *Exec) AggregateShards(shards []*tensors.Tensor) ([]*distributed.Tensor, error) {
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
	distributedOutputs := make([]*distributed.Tensor, numOutputs)
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
		distributedOutputs[outputIdx], err = distributed.NewTensor(shardingSpec, outputShards)
		if err != nil {
			return nil, err
		}
	}
	return distributedOutputs, nil
}

// ExecWithGraph is similar to Exec, but it also returns the computation graph used
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
// Alternatively, you can provide I args of distributed.Tensor (they already include one value per device),
// matching the DistributedMesh provided to Exec.SPMD.
//
// Errors (with full stack-traces) are returned on failure.
func (e *Exec) ExecWithGraph(args ...any) ([]*tensors.Tensor, *Graph, error) {
	return e.ExecWithGraphOnDevice(backends.DeviceNum(0), args...)
}

// ExecOnDevice behaves like Exec but for portable computations uses the given device for execution.
//
// deafultDevice is used for single-device computations that are portable (no fixed device assignment set
// WithDeviceAssignment). Otherwise, it is ignored.
func (e *Exec) ExecOnDevice(defaultDevice backends.DeviceNum, args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.ExecWithGraph(args...)
	return outputs, err
}

// MustNewExecAny constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// `graphFn` can take only *Node parameters as input, and it returns one or more *Node.
// Except if there are no inputs, in which case graphFn needs to take a *Graph as the first parameter.
//
// It panics if the inputs are invalid.
//
// See also the generics MustNewExec (or MustNewExec for returning an error), which checks for valid graphFn in compile time.
func MustNewExecAny(backend backends.Backend, graphFn any) *Exec {
	return must.M1(NewExecAny(backend, graphFn))
}

// MustNewExec constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// graphFn should take *Node as input and return a *Node -- except if there are no (Node) inputs,
// in which case it should take a single *Graph input.
//
// It's a wrapper for MustNewExecAny, but uses generics to type check that
// graphFn is valid.
func MustNewExec[F ExecGraphFn](backend backends.Backend, graphFn F) *Exec {
	return MustNewExecAny(backend, graphFn)
}

// ExecOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to MustNewExec, Exec.Exec and Exec.Finalize for functions that return only one output.
func ExecOnce[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) (*tensors.Tensor, error) {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	results, err := e.Exec(args...)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// ExecOnceN builds the graph and executes it with the given arguments and returns the various outputs.
//
// It's short for a call to MustNewExec, Exec.Exec and Exec.Finalize for functions that return only one output.
//
// See ExecOnce for a more convenient version if you have only one output.
// Also, see MustExecOnceN or MustExecOnce if you want it to panic on error.
func ExecOnceN[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) ([]*tensors.Tensor, error) {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	results, err := e.Exec(args...)
	if err != nil {
		return nil, err
	}
	return results, nil
}

// MustExecOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to MustNewExec, Exec.MustExec and Exec.Finalize for functions that return only one output.
// It panics on error.
//
// See MustExecOnceN if you have multiple outputs.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func MustExecOnce[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) *tensors.Tensor {
	return MustExecOnceN(backend, graphFn, args...)[0]
}

// MustExecOnceN builds the graph and executes it with the given arguments, and returns various outputs.
//
// It's short for a call to MustNewExec, Exec.MustExec and Exec.Finalize.
// It panics on error.
//
// See MustExecOnce for a more convenient version if you have only one output.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func MustExecOnceN[F ExecGraphFn](backend backends.Backend, graphFn F, args ...any) []*tensors.Tensor {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	return e.MustExec(args...)
}

// MustExec executes the computation with the given arguments.
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
func (e *Exec) MustExec(args ...any) []*tensors.Tensor {
	results, _ := e.MustExecWithGraph(args...)
	return results
}

// MustExecWithGraph is similar to MustExec, but it also returns the computation graph used
// in the call.
//
// The underlying Exec creates different computation graphs when the inputs' shapes change,
// so different calls may return different graphs.
//
// It returns the outputs in a slice, even if there is only one output. It also returns the computation graph used.
//
// It panics on errors (with full stack-traces).
func (e *Exec) MustExecWithGraph(args ...any) ([]*tensors.Tensor, *Graph) {
	results, g, err := e.ExecWithGraph(args...)
	if err != nil {
		panic(err)
	}
	return results, g
}

// Exec1 executes the graph with the given arguments and returns one output.
//
// It returns an error if the graph doesn't return exactly one output.
//
// See Exec for more details.
func (e *Exec) Exec1(args ...any) (*tensors.Tensor, error) {
	results, _, err := e.ExecWithGraph(args...)
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

// Exec2 executes the graph with the given arguments and returns two outputs.
//
// It returns an error if the graph doesn't return exactly two outputs.
//
// See Exec for more details.
func (e *Exec) Exec2(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.ExecWithGraph(args...)
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

// Exec3 executes the graph with the given arguments and returns three outputs.
//
// It returns an error if the graph doesn't return exactly three outputs.
//
// See Exec for more details.
func (e *Exec) Exec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.ExecWithGraph(args...)
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

// Exec4 executes the graph with the given arguments and returns four outputs.
//
// It returns an error if the graph doesn't return exactly four outputs.
//
// See Exec for more details.
func (e *Exec) Exec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.ExecWithGraph(args...)
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

// MustExec1 executes the graph with the given arguments and returns one output.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly one output.
//
// See MustExec for more details.
func (e *Exec) MustExec1(args ...any) *tensors.Tensor {
	results := e.MustExec(args...)
	if len(results) != 1 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly one as expected by MustExec1",
			e.Name(),
			len(results),
		)
	}
	return results[0]
}

// MustExec2 executes the graph with the given arguments and returns two outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly two outputs.
//
// See MustExec for more details.
func (e *Exec) MustExec2(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	results := e.MustExec(args...)
	if len(results) != 2 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly two as expected by MustExec2",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1]
}

// MustExec3 executes the graph with the given arguments and returns three outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly three outputs.
//
// See MustExec for more details.
func (e *Exec) MustExec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.MustExec(args...)
	if len(results) != 3 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly three as expected by MustExec3",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2]
}

// MustExec4 executes the graph with the given arguments and returns four outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly four outputs.
//
// See MustExec for more details.
func (e *Exec) MustExec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.MustExec(args...)
	if len(results) != 4 {
		exceptions.Panicf(
			"graph %q returned %d results, as opposed to exactly four as expected by MustExec4",
			e.Name(),
			len(results),
		)
	}
	return results[0], results[1], results[2], results[3]
}
