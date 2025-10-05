package model

import (
	"runtime"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/pkg/errors"
)

// Exec is an executor of models. It works like graph.Exec, but it handles models variables, passing them
// automatically as "side inputs" to the graph -- or as "side outputs" if they are updated in the graph.
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
	graphs *sets.Set[graph.GraphId]

	// List of variables that have been used per graph built, both as input and as output (if they were modified):
	sideInputs, sideOutputs map[graph.GraphId][]*Variable
}

// NewExec creates a new Exec (executor) object taking a builderFn that builds the model's computation graph.
//
// Once the model executor is built, use it with Exec.Exec (or Exec.Call to panic on errors).
// Or their variations Exec.Exec1, Exec.Exec2, ..., Exec.Call1, Exec.Call2, etc.
//
// The builderFn closure must be a function or a method of the model object.
//
// The builderFn closure may take a *graph.Graph as a first argument,
// plus zero to four *Node arguments or a "...*Node" argument.
// It must return either 0 to 4 *Node values or a []*Node.
//
// Examples of valid builderFn functions or methods (methods can be passed and Go will create a closure for them) for
// various fictitious models:
//
//	func (m *ComplexModel) Predict(inputs ...*Node) (outputs []*Node) {...}
//	predictExec, err := MustNewExec(backend, m.Predict)
//
//	MustNewExec(backend, func(gradients ...*Node) { applyGradients(model, gradients...) })
//
//	func (rng *RngState) RandomUniform(g *graph.Graph) *Node {...}
//	uniformExec, err := MustNewExec(backend, rng.RandomUniform)
//
//	func Statistics(x *Node) (mean, variance *Node) {...}
//	statsExec, err := MustNewExec(backend, Statistics)
//
//	func (space *NonEuclideanSpace) Distance(x, y *Node) (distance *Node) {...}
//	myDistanceExec, err := MustNewExec(backend, space.Distance)
//
// If you are only going to execute the model/function once, you can use CallOnce
func NewExec[B BuilderFnSet](backend backends.Backend, builderFn B) (*Exec, error) {
	graphsSet := sets.Make[graph.GraphId]()
	e := &Exec{
		backend:     backend,
		graphs:      &graphsSet,
		sideInputs:  make(map[graph.GraphId][]*Variable),
		sideOutputs: make(map[graph.GraphId][]*Variable),
	}
	var err error
	var canonicalBuilderFn normalizedBuilderFn
	canonicalBuilderFn, e.numBuilderInputs, e.numBuilderOutputs, err = convertToNormalizedBuilderFn(builderFn)
	if err != nil {
		return nil, err
	}

	// Build graph executor using the canonicalized builderFn.
	// Notice that variable values will be passed as "side inputs" to the graph.
	if e.numBuilderInputs != 0 {
		// If the model has inputs, we take the graph from them first.
		e.exec, err = graph.NewExec(backend, func(inputs []*graph.Node) []*graph.Node {
			if len(inputs) != e.numBuilderInputs {
				panic(errors.Errorf("wrong number of inputs for model, expected %d, got %d", e.numBuilderInputs, len(inputs)))
			}
			g := inputs[0].Graph()
			e.registerGraphId(g.GraphId())
			outputs := canonicalBuilderFn(g, inputs)
			return e.appendSideOutputs(g, outputs)
		})
	} else {
		// If the model has no input values (nodes), so we must provide the graph as an input.
		e.exec, err = graph.NewExec(backend, func(g *graph.Graph) []*graph.Node {
			e.registerGraphId(g.GraphId())
			outputs := canonicalBuilderFn(g, nil)
			return e.appendSideOutputs(g, outputs)
		})
	}
	if err != nil {
		return nil, err
	}
	e.exec.SetSideParamsHook(e.setSideParams)

	// Add cleanup functions to resources that would not be otherwise released.
	runtime.AddCleanup(e, func(registeredGraphIDs *sets.Set[graph.GraphId]) {
		for gID := range *registeredGraphIDs {
			removeGraphIds(gID)
		}
	}, e.graphs)
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
	var outputs []*tensors.Tensor
	var g *graph.Graph
	err := exceptions.TryCatch[error](func() {
		outputs, g = e.exec.CallWithGraph(inputs...)
	})
	if err != nil {
		return nil, err
	}
	return e.extractVariableOutputs(g.GraphId(), outputs), nil
}

func (e *Exec) setSideParams(g *graph.Graph, inputBuffers []backends.Buffer, donate []bool) {
	// TODO: Initialize variables if needed.
	gID := g.GraphId()
	sideInputs := e.sideInputs[gID]
	for _, v := range sideInputs {
		nodes, found := v.graphToNodes.Load(gID)
		if !found || nodes == nil || nodes.paramNode == nil {
			exceptions.Panicf("models.Exec input variable %q was marked as needed by the model builder, but "+
				"it has no associated parameter "+
				"in graph #%d-- this is likely a bug in the models system, please report it in github.com/gomlx/gomlx",
				v, gID)
			panic(nil)
		}
		if nodes.paramNode.Type() != graph.NodeTypeParameter {
			exceptions.Panicf("invalid paramNode type %q for variable %q in graph #%d", nodes.paramNode.Type(), v, gID)
		}
		handle := nodes.paramNode.GetParameterHandle()
		if v.ChangedInGraph(g) {
			// We donate the buffer, since we are getting a new one on the output.
			inputBuffers[handle] = v.Value().DonateBuffer(e.backend, e.exec.DeviceNum())
			v.Value().FinalizeAll()
			v.value = nil
			donate[handle] = true
		} else {
			if v.value == nil {
				//if e.isInitializeVariablesExec {
				//	Panicf("variable %q used and not initialized during variable initialization, this would lead to "+
				//		"recursive initialization of variables, and is not supported", v.ScopeAndName())
				//} else {
				exceptions.Panicf("variable %q failed to initialize for graph #%d", v, gID)
				panic(nil)
				//}
			}
			inputBuffers[handle] = v.Value().Buffer(e.backend, e.exec.DeviceNum())
			donate[handle] = false
		}
	}
}

// appendSideOutputs at the end of the computation graph building.
func (e *Exec) appendSideOutputs(g *graph.Graph, outputs []*graph.Node) []*graph.Node {
	sideOutputs := e.sideOutputs[g.GraphId()]
	for _, v := range sideOutputs {
		outputs = append(outputs, v.ValueGraph(g))
	}
	return outputs
}

// extractVariableOutputs extract variable updates
func (e *Exec) extractVariableOutputs(gID graph.GraphId, outputs []*tensors.Tensor) []*tensors.Tensor {
	sideOutputs := e.sideOutputs[gID]
	if len(sideOutputs) == 0 {
		return outputs
	}
	varValues := outputs[len(outputs)-len(sideOutputs):]
	outputs = outputs[:len(outputs)-len(sideOutputs)]
	for idx, v := range sideOutputs {
		v.SetValue(varValues[idx])
	}
	return outputs
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
