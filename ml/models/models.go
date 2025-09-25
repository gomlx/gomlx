// Package models provide helpers to build, execute, save and load models and their weights.
//
// It hinges on the following abstraction of "model":
//
//   - Fields with static "hyperparameters" (e.g.: learning rate, batch size, number of layers, etc.)
//   - Fields of the type *Variable with the model's weights (trainable or not).
//   - A method called `Build(...)` that should build the model's computation graph (see package github.com/gomlx/ml/graph).
//     It should optionally take a *graph.Graph as an argument, plus one, two or three *Node arguments or a []*Node argument,
//     and it should return 0 to 3 *Node values, or a []*Node.
//
// Example:
package models

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/models/builderif"
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
	model   any

	// Basic graph executor and the wrapper function with a fixed signature that wraps the model's Build function.
	exec                                *graph.Exec
	builderFn                           builderif.BuilderFn
	numBuilderInputs, numBuilderOutputs int

	// List of variables that have been used per graph built, both as input and as output (if they were modified):
	sideInputs, sideOutputs map[graph.GraphId][]*Variable
}

// NewExec creates a new Exec object using the model object passed.
// It keeps a reference to the model object, and it will use it every time it needs to build a new computation graph.
//
// It returns an error if the model object does not have a valid Builder API.
func NewExec(backend backends.Backend, model any) (*Exec, error) {
	e := &Exec{
		backend:     backend,
		model:       model,
		sideInputs:  make(map[graph.GraphId][]*Variable),
		sideOutputs: make(map[graph.GraphId][]*Variable),
	}
	var err error
	e.builderFn, e.numBuilderInputs, e.numBuilderOutputs, err = builderif.ConvertToBuilderFn(model)
	if err != nil {
		return nil, err
	}

	// Build graph executor based on the model's signature (in e.builderFn).
	// Notice that variable values are passed as "side inputs" to the graph.
	if e.numBuilderInputs != 0 {
		// If the model has inputs, we take the graph from them first.
		e.exec, err = graph.NewExecOrError(backend, func(inputs []*graph.Node) []*graph.Node {
			if len(inputs) != e.numBuilderInputs {
				panic(errors.Errorf("wrong number of inputs for model, expected %d, got %d", e.numBuilderInputs, len(inputs)))
			}
			g := inputs[0].Graph()
			return e.builderFn(g, inputs)
		})
	} else {
		// If the model has no input values (nodes), so we must provide the graph as an input.
		e.exec, err = graph.NewExecOrError(backend, func(g *graph.Graph) []*graph.Node {
			return e.builderFn(g, nil)
		})
	}
	if err != nil {
		return nil, err
	}
	e.exec.SetSideParamsHook(e.setSideParams)
	return e, nil
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
