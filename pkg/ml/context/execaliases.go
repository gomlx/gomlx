package context

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// MustNewExec constructs an Exec object that uses the given ctxGraphFn to build
// computation graphs with a Context. ctxGraphFn must take a *Context input
// parameter followed by one or more *Node parameters as input and return one
// or more *Node.
//
// The Context ctx passed will be passed to all computation graph construction calls
// (ctxGraphFn), as well as during the graph execution later. If set to nil, it automatically
// creates a new one.
//
// Before the execution of a graph, if needed, it initializes the variables in the context.
//
// This is a generic wrapper around NewExecAny that checks that types are
// correct (but doesn't support all possible types of ctxGraphFn).
//
// It panics on error.
func MustNewExec[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F) *Exec {
	e, err := NewExecAny(backend, ctx, ctxGraphFn)
	if err != nil {
		panic(err)
	}
	return e
}

// Call parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
//
// Notice Context shouldn't be passed by Call; it will use automatically the context
// stored in context.Exec -- you can change it with SetContext.
//
// If a graph does not yet exist, one is created, compiled and cached for the shapes
// of the inputs.
// It passes the context to the registered ctxGraphFn. After the very first invocation of Call
// the context is marked as Context.Reuse().
//
// It returns the outputs in a slice, even if there is only one output.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) Call(args ...any) []*tensors.Tensor {
	outputs, _ := e.CallWithGraph(args...)
	return outputs
}

// CallWithGraph is similar to Call, but it also returns the computation graph used in the call.
// Since Exec creates different computation graphs for each different set of parameters,
// this can help disambiguate in case the user needs to use the Graph for something else.
//
// It returns the outputs in a slice (it can be empty even) and the graph used to execute the computation.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) CallWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph) {
	var err error
	outputs, g, err = e.ExecWithGraph(args...)
	if err != nil {
		panic(err)
	}
	return
}

// ExecOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize.
//
// See ExecOnce for a more convenient version if you have only one output.
func ExecOnceN[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) ([]*tensors.Tensor, error) {
	e, err := NewExec(backend, ctx, ctxGraphFn)
	if err != nil {
		return nil, err
	}
	defer e.Finalize()
	return e.Exec(args...)
}

// CallOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize.
//
// See CallOnce for a more convenient version if you have only one output.
//
// It panics on error. See ExecOnceN for a version that returns an error.
func CallOnceN[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) []*tensors.Tensor {
	outputs, err := ExecOnceN(backend, ctx, ctxGraphFn, args...)
	if err != nil {
		panic(err)
	}
	return outputs
}

// ExecOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize for functions that return only one output.
//
// See ExecOnceN if you have multiple (or zero) outputs.
func ExecOnce[F ExecGraphFnOneOutput](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) (*tensors.Tensor, error) {
	outputs, err := ExecOnceN(backend, ctx, ctxGraphFn, args...)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.Errorf("ExecOnce expected one output, got %d", len(outputs))
	}
	return outputs[0], nil
}

// CallOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.Call and Exec.Finalize for functions that return only one output.
//
// See CallOnceN if you have multiple outputs.
//
// It panics on error. See ExecOnce for a version that returns an error.
func CallOnce[F ExecGraphFnOneOutput](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) *tensors.Tensor {
	output, err := ExecOnce(backend, ctx, ctxGraphFn, args...)
	if err != nil {
		panic(err)
	}
	return output
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
