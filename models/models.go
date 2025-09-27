// Package models provide helpers to build, execute, save and load models and their weights.
//
// Note: This package aims to replace the `ml/context` package and all the layers libraries.
// **It's still in beta**. Any feedback is very welcome.
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
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
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

	// Graphs created by this executor.
	graphs types.Set[graph.GraphId]

	// List of variables that have been used per graph built, both as input and as output (if they were modified):
	sideInputs, sideOutputs map[graph.GraphId][]*Variable
}

// NewExec creates a new Exec (executor) object taking the closure buildFn that builds the model's computation graph.
//
// The buildFn closure may take a *graph.Graph as a first argument,
// plus zero to four *Node arguments or a "...*Node" argument.
// It must return either 0 to 4 *Node values or a []*Node.
//
// Examples of valid buildFn functions or methods (methods can be passed and Go will create a closure for them) for
// various fictitious models:
//
//	func (m *ComplexModel) ComplexModel(inputs ...*Node) (outputs []*Node) {...}
//	func (m *MyModel) ApplyGradients(gradients ...*Node)  {...} // No outputs, this graph updates the weights directly.
//	func (rng *RngState) RandomUniform(g *graph.Graph) *Node {...}
//	func Statistics(x *Node) (mean, variance *Node) {...}
//	func (space *NonEuclideanSpace) Distance(x, y *Node) (distance *Node) {...}
//
// The returned Exec keeps a reference to the model object, and it will use it every time it needs to build a new computation graph.
//
// It returns an error if the model object does not have a valid Builder API.
func NewExec[B builderiface.FnSet](backend backends.Backend, builderFn B) (*Exec, error) {
	e := &Exec{
		backend:     backend,
		graphs:      types.MakeSet[graph.GraphId](),
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
