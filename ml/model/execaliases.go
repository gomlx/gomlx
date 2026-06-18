// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// NewExec constructs an Exec object for the given scopeOrStore and symbolic computation function graphFn.
//
// The scopeOrStore can be a *Scope or a *Store. If it is a *Store, it is converted to its RootScope().
// If nil, it automatically creates a new empty store and uses its RootScope().
//
// The graphFn is called to build the computation graphs with a Scope.
// It must take a *Scope input parameter followed by one or more *Node parameters as input and return one or more *Node.
// Alternatively, it can, instead of *Node inputs, take a *Graph object when there are no input tensors.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetNodeValue) are updated also during execution.
// More details see Exec.
//
// This is a generic wrapper around NewExecCanonical that checks that types are
// correct.
func NewExec[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F) (*Exec, error) {
	canonicalFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice := convertExecFn(graphFn)
	return NewExecCanonical(backend, store, canonicalFn, numInputs, numOutputs, inputIsGraph, inputAsSlice, outputAsSlice)
}

// MustNewExec constructs an Exec object for the given scopeOrStore and symbolic computation function graphFn.
//
// The scopeOrStore can be a *Scope or a *Store. If it is a *Store, it is converted to its RootScope().
// If nil, it automatically creates a new empty store and uses its RootScope().
//
// The graphFn is called to build the computation graphs with a Scope.
// It must take a *Scope input parameter followed by one or more *Node parameters as input and return one or more *Node.
// Alternatively, it can, instead of *Node inputs, take a *Graph object when there are no input tensors.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetNodeValue) are updated also during execution.
// More details see Exec.
//
// It panics on error.
func MustNewExec[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F) *Exec {
	return must.M1(NewExec(backend, store, graphFn))
}

// DistributedExec is just like Exec, but aggregates the outputs into *dtensor.Tensor.
// Usually, Exec will return alice of numDevices * nnumOutputs individual shards (*tensors.Tensor).
//
// Notice that to actually trigger distributed execution, you must set Exec.AutoSharding (or Exec.SPMD) and
// set the proper sharding specs of the inputs and outputs.
//
// See also Exec.AggregateShards.
func (e *Exec) DistributedExec(args ...any) ([]*dtensor.Tensor, error) {
	shards, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, err
	}
	return e.exec.AggregateShards(shards)
}

// AggregateShards returned by Exec into *distributedTensor outputs.
// Usually, Exec will return alice of numDevices * nnumOutputs individual shards (*tensors.Tensor).
//
// See also DistributedExec, which calls Exec and then calls this.
func (e *Exec) AggregateShards(shards []*tensors.Tensor) ([]*dtensor.Tensor, error) {
	return e.exec.AggregateShards(shards)
}

// Call parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
//
// Notice it uses the Store object used during creation -- if needed, you can change it with SetStore.
//
// If a graph does not yet exist, one is created (using graphFn provided during creation), compiled, and cached
// for these shapes of the inputs.
//
// It returns the outputs in a slice. See the aliases Call1, Call2, ..., Call4 when you expect a fixed number of outputs.
func (e *Exec) Call(args ...any) ([]*tensors.Tensor, error) {
	outputs, _, err := e.CallWithGraph(args...)
	return outputs, err
}

// MustCall parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
//
// Notice it uses the Store object used during creation -- if needed, you can change it with SetStore.
//
// If a graph does not yet exist, one is created (using graphFn provided during creation), compiled, and cached
// for these shapes of the inputs.
//
// It returns the outputs in a slice. See the aliases MustCall1, MustCall2, ..., MustCall4 when you expect a fixed number of outputs.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) MustCall(args ...any) []*tensors.Tensor {
	outputs, err := e.Call(args...)
	if err != nil {
		panic(err)
	}
	return outputs
}

// MustCallWithGraph is similar to MustExec, but it also returns the computation graph used in the call.
//
// Since Exec creates different computation graphs when the inputs shapes change,
// this can help disambiguate in case the user needs to use the Graph for something else.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) MustCallWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph) {
	var err error
	outputs, g, err = e.CallWithGraph(args...)
	if err != nil {
		panic(err)
	}
	return
}

// CallOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize.
//
// See ExecOnce for a more convenient version if you have only one output.
func CallOnceN[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F, args ...any) ([]*tensors.Tensor, error) {
	e, err := NewExec(backend, store, graphFn)
	if err != nil {
		return nil, err
	}
	defer e.Finalize()
	return e.Call(args...)
}

// MustCallOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize.
//
// See MustExecOnce for a more convenient version if you have only one output.
//
// It panics on error. See ExecOnceN for a version that returns an error.
func MustCallOnceN[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F, args ...any) []*tensors.Tensor {
	outputs, err := CallOnceN(backend, store, graphFn, args...)
	if err != nil {
		panic(err)
	}
	return outputs
}

// CallOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize for functions that return only one output.
//
// See ExecOnceN if you have multiple (or zero) outputs.
func CallOnce[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F, args ...any) (*tensors.Tensor, error) {
	outputs, err := CallOnceN(backend, store, graphFn, args...)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.Errorf("ExecOnce expected one output, got %d", len(outputs))
	}
	return outputs[0], nil
}

// MustCallOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize for functions that return only one output.
//
// See MustExecOnceN if you have multiple outputs.
//
// It panics on error. See ExecOnce for a version that returns an error.
func MustCallOnce[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F, args ...any) *tensors.Tensor {
	output, err := CallOnce(backend, store, graphFn, args...)
	if err != nil {
		panic(err)
	}
	return output
}

// Call1 executes the graph with the given arguments and returns one output.
//
// It returns an error if the graph doesn't return exactly one output.
//
// See Exec for more details.
func (e *Exec) Call1(args ...any) (*tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, err
	}
	if len(results) != 1 {
		return nil, errors.Errorf("graph %q returned %d results, as opposed to exactly one as expected by Exec1", e.Name(), len(results))
	}
	return results[0], nil
}

// Call2 executes the graph with the given arguments and returns two outputs.
//
// It returns an error if the graph doesn't return exactly two outputs.
//
// See Exec for more details.
func (e *Exec) Call2(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, nil, err
	}
	if len(results) != 2 {
		return nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly two as expected by Exec2", e.Name(), len(results))
	}
	return results[0], results[1], nil
}

// Call3 executes the graph with the given arguments and returns three outputs.
//
// It returns an error if the graph doesn't return exactly three outputs.
//
// See Exec for more details.
func (e *Exec) Call3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	results, _, err := e.CallWithGraph(args...)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(results) != 3 {
		return nil, nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly three as expected by Exec3", e.Name(), len(results))
	}
	return results[0], results[1], results[2], nil
}

// Call4 executes the graph with the given arguments and returns four outputs.
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
		return nil, nil, nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly four as expected by Exec4", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly one as expected by MustExec1", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly two as expected by MustExec2", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly three as expected by MustExec3", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly four as expected by MustExec4", e.Name(), len(results))
	}
	return results[0], results[1], results[2], results[3]
}

// MustExec is a deprecated alias to MustCall.
//
// Deprecated: please, use MustCall instead.
func (e *Exec) MustExec(args ...any) []*tensors.Tensor {
	return e.MustCall(args...)
}

// MustExecWithGraph is a deprecated alias to MustCallWithGraph.
//
// Deprecated: please, use MustCallWithGraph instead.
func (e *Exec) MustExecWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph) {
	return e.MustCallWithGraph(args...)
}

// ExecOnceN is a deprecated alias to CallOnceN.
//
// Deprecated: please, use CallOnceN instead.
func ExecOnceN[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F, args ...any) ([]*tensors.Tensor, error) {
	return CallOnceN(backend, store, graphFn, args...)
}

// MustExecOnceN is a deprecated alias to MustCallOnceN.
//
// Deprecated: please, use MustCallOnceN instead.
func MustExecOnceN[F ExecGraphFn](backend compute.Backend, store *Store, graphFn F, args ...any) []*tensors.Tensor {
	return MustCallOnceN(backend, store, graphFn, args...)
}

// ExecOnce is a deprecated alias to CallOnce.
//
// Deprecated: please, use CallOnce instead.
func ExecOnce[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F, args ...any) (*tensors.Tensor, error) {
	return CallOnce(backend, store, graphFn, args...)
}

// MustExecOnce is a deprecated alias to MustCallOnce.
//
// Deprecated: please, use MustCallOnce instead.
func MustExecOnce[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F, args ...any) *tensors.Tensor {
	return MustCallOnce(backend, store, graphFn, args...)
}

// Exec1 is a deprecated alias to Call1.
//
// Deprecated: please, use Call1 instead.
func (e *Exec) Exec1(args ...any) (*tensors.Tensor, error) {
	return e.Call1(args...)
}

// Exec2 is a deprecated alias to Call2.
//
// Deprecated: please, use Call2 instead.
func (e *Exec) Exec2(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	return e.Call2(args...)
}

// Exec3 is a deprecated alias to Call3.
//
// Deprecated: please, use Call3 instead.
func (e *Exec) Exec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	return e.Call3(args...)
}

// Exec4 is a deprecated alias to Call4.
//
// Deprecated: please, use Call4 instead.
func (e *Exec) Exec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	return e.Call4(args...)
}

// MustExec1 is a deprecated alias to MustCall1.
//
// Deprecated: please, use MustCall1 instead.
func (e *Exec) MustExec1(args ...any) *tensors.Tensor {
	return e.MustCall1(args...)
}

// MustExec2 is a deprecated alias to MustCall2.
//
// Deprecated: please, use MustCall2 instead.
func (e *Exec) MustExec2(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	return e.MustCall2(args...)
}

// MustExec3 is a deprecated alias to MustCall3.
//
// Deprecated: please, use MustCall3 instead.
func (e *Exec) MustExec3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return e.MustCall3(args...)
}

// MustExec4 is a deprecated alias to MustCall4.
//
// Deprecated: please, use MustCall4 instead.
func (e *Exec) MustExec4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return e.MustCall4(args...)
}

// ExecOneOutput is a wrapper around Exec for graphs that return exactly one output.
type ExecOneOutput struct {
	*Exec
}

// Call executes the computation with the given arguments and returns one output.
//
// It returns an error if the graph doesn't return exactly one output.
func (e *ExecOneOutput) Call(args ...any) (*tensors.Tensor, error) {
	return e.Call1(args...)
}

// MustCall executes the graph with the given arguments and returns one output.
//
// It panics on errors or if the graph doesn't return exactly one output.
func (e *ExecOneOutput) MustCall(args ...any) *tensors.Tensor {
	return e.MustCall1(args...)
}

// ExecTwoOutputs is a wrapper around Exec for graphs that return exactly two outputs.
type ExecTwoOutputs struct {
	*Exec
}

// Call executes the computation with the given arguments and returns two outputs.
//
// It returns an error if the graph doesn't return exactly two outputs.
func (e *ExecTwoOutputs) Call(args ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	return e.Call2(args...)
}

// MustCall executes the graph with the given arguments and returns two outputs.
//
// It panics on errors or if the graph doesn't return exactly two outputs.
func (e *ExecTwoOutputs) MustCall(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	return e.MustCall2(args...)
}

// ExecThreeOutputs is a wrapper around Exec for graphs that return exactly three outputs.
type ExecThreeOutputs struct {
	*Exec
}

// Call executes the computation with the given arguments and returns three outputs.
//
// It returns an error if the graph doesn't return exactly three outputs.
func (e *ExecThreeOutputs) Call(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	return e.Call3(args...)
}

// MustCall executes the graph with the given arguments and returns three outputs.
//
// It panics on errors or if the graph doesn't return exactly three outputs.
func (e *ExecThreeOutputs) MustCall(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return e.MustCall3(args...)
}

// NewExec1 constructs an ExecOneOutput wrapper around a new Exec object, constraining the graphFn to return a single output.
func NewExec1[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F) (*ExecOneOutput, error) {
	exec, err := NewExec(backend, store, graphFn)
	if err != nil {
		return nil, err
	}
	return &ExecOneOutput{Exec: exec}, nil
}

// NewExec2 constructs an ExecTwoOutputs wrapper around a new Exec object, constraining the graphFn to return two outputs.
func NewExec2[F ExecGraphFnTwoOutputs](backend compute.Backend, store *Store, graphFn F) (*ExecTwoOutputs, error) {
	exec, err := NewExec(backend, store, graphFn)
	if err != nil {
		return nil, err
	}
	return &ExecTwoOutputs{Exec: exec}, nil
}

// NewExec3 constructs an ExecThreeOutputs wrapper around a new Exec object, constraining the graphFn to return three outputs.
func NewExec3[F ExecGraphFnThreeOutputs](backend compute.Backend, store *Store, graphFn F) (*ExecThreeOutputs, error) {
	exec, err := NewExec(backend, store, graphFn)
	if err != nil {
		return nil, err
	}
	return &ExecThreeOutputs{Exec: exec}, nil
}

// MustNewExec1 constructs an ExecOneOutput wrapper around a new Exec object and panics on error.
func MustNewExec1[F ExecGraphFnOneOutput](backend compute.Backend, store *Store, graphFn F) *ExecOneOutput {
	return &ExecOneOutput{Exec: MustNewExec(backend, store, graphFn)}
}

// MustNewExec2 constructs an ExecTwoOutputs wrapper around a new Exec object and panics on error.
func MustNewExec2[F ExecGraphFnTwoOutputs](backend compute.Backend, store *Store, graphFn F) *ExecTwoOutputs {
	return &ExecTwoOutputs{Exec: MustNewExec(backend, store, graphFn)}
}

// MustNewExec3 constructs an ExecThreeOutputs wrapper around a new Exec object and panics on error.
func MustNewExec3[F ExecGraphFnThreeOutputs](backend compute.Backend, store *Store, graphFn F) *ExecThreeOutputs {
	return &ExecThreeOutputs{Exec: MustNewExec(backend, store, graphFn)}
}
