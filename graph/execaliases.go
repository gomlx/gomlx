package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// Note: This file contains various aliases for NewExec and Exec.
// We separate them from the exec.go file to make it easier to separate the core logic (in exec.go)
// from ergonomics (execaliases.go, this file).

// MustNewExecAny constructs an Exec object that uses the given graphFn to build
// computation graphs.
//
// `graphFn` can take only *Node parameters as input and returns one or more *Node.
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

// NewExecOrError is an alias to NewExec.
//
// Deprecated: use NewExec instead.
func NewExecOrError[F ExecGraphFn](backend backends.Backend, graphFn F) (*Exec, error) {
	return NewExec(backend, graphFn)
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
// Also, see CallOnceN or CallOnce if you want it to panic on error.
func ExecOnceN[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) ([]*tensors.Tensor, error) {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	results, err := e.Exec(args...)
	if err != nil {
		return nil, err
	}
	return results, nil
}

// CallOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to MustNewExec, Exec.Call and Exec.Finalize for functions that return only one output.
// It panics on error.
//
// See CallOnceN if you have multiple outputs.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func CallOnce[F ExecGraphFnOneOutput](backend backends.Backend, graphFn F, args ...any) *tensors.Tensor {
	return CallOnceN(backend, graphFn, args...)[0]
}

// CallOnceN builds the graph and executes it with the given arguments, and returns various outputs.
//
// It's short for a call to MustNewExec, Exec.Call and Exec.Finalize.
// It panics on error.
//
// See CallOnce for a more convenient version if you have only one output.
// Also, see ExecOnceN or ExecOnce if you want any errors returned.
func CallOnceN[F ExecGraphFn](backend backends.Backend, graphFn F, args ...any) []*tensors.Tensor {
	e := MustNewExec(backend, graphFn)
	defer e.Finalize()
	return e.Call(args...)
}

// Call executes the computation with the given arguments.
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
func (e *Exec) Call(args ...any) []*tensors.Tensor {
	results, _ := e.CallWithGraph(args...)
	return results
}

// CallOrError is a deprecated alias to Exec.
//
// Deprecated: Use Exec instead.
func (e *Exec) CallOrError(args ...any) (results []*tensors.Tensor, err error) {
	return e.Exec(args...)
}

// CallWithGraph is similar to Call, but it also returns the computation graph used
// in the call.
//
// The underlying Exec creates different computation graphs when the inputs' shapes change,
// so different calls may return different graphs.
//
// It returns the outputs in a slice, even if there is only one output. It also returns the computation graph used.
//
// It panics on errors (with full stack-traces).
func (e *Exec) CallWithGraph(args ...any) (results []*tensors.Tensor, g *Graph) {
	return e.compileAndExecute(true, args...)
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
		return nil, errors.Errorf("graph %q returned %d results, as opposed to exactly one as expected by Exec1", e.Name(), len(results))
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
		return nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly two as expected by Exec2", e.Name(), len(results))
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
		return nil, nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly three as expected by Exec3", e.Name(), len(results))
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
		return nil, nil, nil, nil, errors.Errorf("graph %q returned %d results, as opposed to exactly four as expected by Exec4", e.Name(), len(results))
	}
	return results[0], results[1], results[2], results[3], nil
}

// Call1 executes the graph with the given arguments and returns one output.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly one output.
//
// See Call for more details.
func (e *Exec) Call1(args ...any) *tensors.Tensor {
	results := e.Call(args...)
	if len(results) != 1 {
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly one as expected by Call1", e.Name(), len(results))
	}
	return results[0]
}

// Call2 executes the graph with the given arguments and returns two outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly two outputs.
//
// See Call for more details.
func (e *Exec) Call2(args ...any) (*tensors.Tensor, *tensors.Tensor) {
	results := e.Call(args...)
	if len(results) != 2 {
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly two as expected by Call2", e.Name(), len(results))
	}
	return results[0], results[1]
}

// Call3 executes the graph with the given arguments and returns three outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly three outputs.
//
// See Call for more details.
func (e *Exec) Call3(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.Call(args...)
	if len(results) != 3 {
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly three as expected by Call3", e.Name(), len(results))
	}
	return results[0], results[1], results[2]
}

// Call4 executes the graph with the given arguments and returns four outputs.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly four outputs.
//
// See Call for more details.
func (e *Exec) Call4(args ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	results := e.Call(args...)
	if len(results) != 4 {
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly four as expected by Call4", e.Name(), len(results))
	}
	return results[0], results[1], results[2], results[3]
}
