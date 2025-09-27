package models

import (
	"runtime"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/models/builderiface"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/pkg/errors"
)

// Exec is an executor of models.
//
// It holds the "model" object (passed in NewExec), and for each different shaped inputs of the Call() (or MustCall) method,
// it calls model.Build to rebuild the computation graph for that particular combination of inputs, JIT-compiles it and then executes it.
// If the combination of shapes has already been seen before, it will reuse the pre-compiled graph -- up to a certain cache size.
type Exec struct {
	backend backends.Backend

	// Basic graph executor and the wrapper function with a fixed signature that wraps the model's Build function.
	exec                                *graph.Exec
	numBuilderInputs, numBuilderOutputs int

	// mu protects per-graph information: since there may be different concurrent executions,
	// creating different graphs.
	mu sync.Mutex

	// Graphs created by this executor: it is a pointer to a map, so we can clean up on garbage collection
	// (runtime.AddCleanUp requires a pointer)
	graphs *types.Set[graph.GraphId]

	// List of variables that have been used per graph built, both as input and as output (if they were modified):
	sideInputs, sideOutputs map[graph.GraphId][]*Variable
}

// NewExec creates a new Exec (executor) object taking a closure buildFn that builds the model's computation graph.
//
// Once the model executor is built, use it with Exec.Exec (or Exec.Call to panic on errors).
// Or their variations Exec.Exec1, Exec.Exec2, ..., Exec.Call1, Exec.Call2, etc.
//
// The buildFn closure must be a function or a method of the model object.
//
// The buildFn closure may take a *graph.Graph as a first argument,
// plus zero to four *Node arguments or a "...*Node" argument.
// It must return either 0 to 4 *Node values or a []*Node.
//
// Examples of valid buildFn functions or methods (methods can be passed and Go will create a closure for them) for
// various fictitious models:
//
//	func (m *ComplexModel) Predict(inputs ...*Node) (outputs []*Node) {...}
//	predictExec, err := NewExec(backend, m.Predict)
//
//	NewExec(backend, func(gradients ...*Node) { applyGradients(model, gradients...) })
//
//	func (rng *RngState) RandomUniform(g *graph.Graph) *Node {...}
//	uniformExec, err := NewExec(backend, rng.RandomUniform)
//
//	func Statistics(x *Node) (mean, variance *Node) {...}
//	statsExec, err := NewExec(backend, Statistics)
//
//	func (space *NonEuclideanSpace) Distance(x, y *Node) (distance *Node) {...}
//	myDistanceExec, err := NewExec(backend, space.Distance)
//
// If you are only going to execute the model/function once, you can use ExecOnce
func NewExec[B builderiface.FnSet](backend backends.Backend, builderFn B) (*Exec, error) {
	graphsSet := types.MakeSet[graph.GraphId]()
	e := &Exec{
		backend:     backend,
		graphs:      &graphsSet,
		sideInputs:  make(map[graph.GraphId][]*Variable),
		sideOutputs: make(map[graph.GraphId][]*Variable),
	}
	var err error
	var canonicalBuilderFn builderiface.BuilderFn
	canonicalBuilderFn, e.numBuilderInputs, e.numBuilderOutputs, err = builderiface.ConvertToBuilderFn(builderFn)
	if err != nil {
		return nil, err
	}

	// Build graph executor using the canonicalized builderFn.
	// Notice that variable values will be passed as "side inputs" to the graph.
	if e.numBuilderInputs != 0 {
		// If the model has inputs, we take the graph from them first.
		e.exec, err = graph.NewExecOrError(backend, func(inputs []*graph.Node) []*graph.Node {
			if len(inputs) != e.numBuilderInputs {
				panic(errors.Errorf("wrong number of inputs for model, expected %d, got %d", e.numBuilderInputs, len(inputs)))
			}
			g := inputs[0].Graph()
			e.registerGraphId(g.GraphId())
			return canonicalBuilderFn(g, inputs)
		})
	} else {
		// If the model has no input values (nodes), so we must provide the graph as an input.
		e.exec, err = graph.NewExecOrError(backend, func(g *graph.Graph) []*graph.Node {
			e.registerGraphId(g.GraphId())
			return canonicalBuilderFn(g, nil)
		})
	}
	if err != nil {
		return nil, err
	}
	e.exec.SetSideParamsHook(e.setSideParams)

	// Add cleanup functions to resources that would not be otherwise released.
	runtime.AddCleanup(e, func(registeredGraphIDs *types.Set[graph.GraphId]) {
		for gID := range *registeredGraphIDs {
			removeGraphIds(gID)
		}
	}, e.graphs)
	return e, nil
}

// Finalize frees all resources used by the executor -- and doesn't wait for the garbage collector to do it.
//
// It is safe to call this method multiple times.
func (e *Exec) Finalize() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.exec == nil {
		// Already freed.
		return
	}

	for gID := range *e.graphs {
		removeGraphIds(gID)
	}
	clear(*e.graphs)
	clear(e.sideInputs)
	clear(e.sideOutputs)
	e.exec.Finalize()
	e.exec = nil
}

// Exec executes the model with the given inputs.
//
// The number of inputs must match the number of inputs of the model builder.
//
// It returns an error in case of any issues (either building/JIT-compiling the graph or during the execution).
func (e *Exec) Exec(inputs ...any) ([]*tensors.Tensor, error) {
	if len(inputs) != e.numBuilderInputs {
		return nil, errors.Errorf("wrong number of inputs for model, expected %d, got %d", e.numBuilderInputs, len(inputs))
	}
	return e.exec.CallOrError(inputs...)
}

func (e *Exec) setSideParams(graph *graph.Graph, inputBuffers []backends.Buffer, donate []bool) {
	return
}

// Exec1 executes the model with the given inputs and returns the output directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec1(inputs ...any) (*tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 1 {
		return nil, errors.Errorf("model Build method has %d outputs, cannot use Exec1", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.Errorf("wrong number of outputs for model for Exec1, expected 1, got %d", len(outputs))
	}
	return outputs[0], nil
}

// Exec2 executes the model with the given inputs and returns two outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec2(inputs ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 2 {
		return nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec2", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, err
	}
	if len(outputs) != 2 {
		return nil, nil, errors.Errorf("wrong number of outputs for model for Exec2, expected 2, got %d", len(outputs))
	}
	return outputs[0], outputs[1], nil
}

// Exec3 executes the model with the given inputs and returns three outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec3(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 3 {
		return nil, nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec3", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(outputs) != 3 {
		return nil, nil, nil, errors.Errorf("wrong number of outputs for model for Exec3, expected 3, got %d", len(outputs))
	}
	return outputs[0], outputs[1], outputs[2], nil
}

// Exec4 executes the model with the given inputs and returns four outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec4(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 4 {
		return nil, nil, nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec4", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if len(outputs) != 4 {
		return nil, nil, nil, nil, errors.Errorf("wrong number of outputs for model for Exec4, expected 4, got %d", len(outputs))
	}
	return outputs[0], outputs[1], outputs[2], outputs[3], nil
}

// Call is a variation of Exec that panics if there is an error.
func (e *Exec) Call(inputs ...any) []*tensors.Tensor {
	return must.M1(e.Exec(inputs...))
}

// Call1 is a variation of Exec that panics if there is an error and returns the one output directly (as opposed to a slice of tensors).
func (e *Exec) Call1(inputs ...any) *tensors.Tensor {
	return must.M1(e.Exec1(inputs...))
}

// Call2 is a variation of Exec that panics if there is an error and returns two outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call2(inputs ...any) (*tensors.Tensor, *tensors.Tensor) {
	return must.M2(e.Exec2(inputs...))
}

// Call3 is a variation of Exec that panics if there is an error and returns three outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call3(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return must.M3(e.Exec3(inputs...))
}

// Call4 is a variation of Exec that panics if there is an error and returns four outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call4(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return must.M4(e.Exec4(inputs...))
}
