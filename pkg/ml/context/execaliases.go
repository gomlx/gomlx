package context

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// NewExec constructs an Exec object for the given context and symbolic computation function ctxGraphFn.
//
// The ctxGraphFn is called to build the computation graphs with a Context.
// It must take a *Context input parameter followed by one or more *Node parameters as input and return one or more *Node.
// Alternatively, it can, instead of *Node inputs, take a *Graph object when there are no input tensors.
//
// The Context ctx passed in the construction is used in all calls to ctxGraphFn, as well as during the graph execution later.
// If set to nil, it automatically creates a new empty context.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetValueGraph) are updated also during execution.
// More details see Exec.
//
// This is a generic wrapper around NewExecAny that checks that types are
// correct (but doesn't support all possible types of ctxGraphFn).
func NewExec[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F) (*Exec, error) {
	return NewExecAny(backend, ctx, ctxGraphFn)
}

// MustNewExec constructs an Exec object for the given context and symbolic computation function ctxGraphFn.
//
// The ctxGraphFn is called to build the computation graphs with a Context.
// It must take a *Context input parameter followed by one or more *Node parameters as input and return one or more *Node.
// Alternatively, it can, instead of *Node inputs, take a *Graph object when there are no input tensors.
//
// The Context ctx passed in the construction is used in all calls to ctxGraphFn, as well as during the graph execution later.
// If set to nil, it automatically creates a new empty context.
//
// Before the execution of a graph, it initializes the variables as needed, using the configured initializer.
// And variables updated in the graph (using Variable.SetValueGraph) are updated also during execution.
// More details see Exec.
//
// It panics on error.
func MustNewExec[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F) *Exec {
	e, err := NewExecAny(backend, ctx, ctxGraphFn)
	if err != nil {
		panic(err)
	}
	return e
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
	return e.exec.AggregateShards(shards)
}

// AggregateShards returned by Exec into *distributedTensor outputs.
// Usually, Exec will return alice of numDevices * nnumOutputs individual shards (*tensors.Tensor).
//
// See also DistributedExec, which calls Exec and then calls this.
func (e *Exec) AggregateShards(shards []*tensors.Tensor) ([]*distributed.Tensor, error) {
	return e.exec.AggregateShards(shards)
}

// MustExec parses the arguments into tensors (if they are not yet) and executes
// the graph corresponding to the shapes of the arguments.
//
// Notice it uses the Context object used during creation -- if needed, you can change it with SetContext.
//
// If a graph does not yet exist, one is created (using ctxGraphFn provided during creation), compiled, and cached
// for these shapes of the inputs.
// After the very first invocation of Exec, the context is marked as Context.Reuse().
//
// It returns the outputs in a slice. See MustExec1, MustExec2, ..., MustExec4 as aliases when you expect a fixed number of outputs.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) MustExec(args ...any) []*tensors.Tensor {
	outputs, _ := e.MustExecWithGraph(args...)
	return outputs
}

// MustExecWithGraph is similar to MustExec, but it also returns the computation graph used in the call.
//
// Since Exec creates different computation graphs when the inputs shapes change,
// this can help disambiguate in case the user needs to use the Graph for something else.
//
// It panics with an informative error if something goes wrong.
func (e *Exec) MustExecWithGraph(args ...any) (outputs []*tensors.Tensor, g *Graph) {
	var err error
	outputs, g, err = e.ExecWithGraph(args...)
	if err != nil {
		panic(err)
	}
	return
}

// ExecOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize.
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

// MustExecOnceN builds the graph and executes it with the given arguments and returns various output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize.
//
// See MustExecOnce for a more convenient version if you have only one output.
//
// It panics on error. See ExecOnceN for a version that returns an error.
func MustExecOnceN[F ExecGraphFn](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) []*tensors.Tensor {
	outputs, err := ExecOnceN(backend, ctx, ctxGraphFn, args...)
	if err != nil {
		panic(err)
	}
	return outputs
}

// ExecOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize for functions that return only one output.
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

// MustExecOnce builds the graph and executes it with the given arguments and returns the one output.
//
// It's short for a call to NewExec, Exec.MustExec, and Exec.Finalize for functions that return only one output.
//
// See MustExecOnceN if you have multiple outputs.
//
// It panics on error. See ExecOnce for a version that returns an error.
func MustExecOnce[F ExecGraphFnOneOutput](backend backends.Backend, ctx *Context, ctxGraphFn F, args ...any) *tensors.Tensor {
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

// MustExec1 executes the graph with the given arguments and returns one output.
//
// It panics on errors (with full stack-traces) or if the graph doesn't return exactly one output.
//
// See MustExec for more details.
func (e *Exec) MustExec1(args ...any) *tensors.Tensor {
	results := e.MustExec(args...)
	if len(results) != 1 {
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly one as expected by MustExec1", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly two as expected by MustExec2", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly three as expected by MustExec3", e.Name(), len(results))
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
		exceptions.Panicf("graph %q returned %d results, as opposed to exactly four as expected by MustExec4", e.Name(), len(results))
	}
	return results[0], results[1], results[2], results[3]
}
