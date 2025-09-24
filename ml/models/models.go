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

import "github.com/gomlx/gomlx/ml/models/builderif"

// Exec is an executor of models.
//
// It holds the "model" object (passed in NewExec), and for each different shaped inputs of the Call() (or MustCall) method,
// it calls model.Build to rebuild the computation graph for that particular combination of inputs, JIT-compiles it and then executes it.
// If the combination of shapes has already been seen before, it will reuses the pre-compiled graph -- up to a certain cache size.
type Exec struct {
	model     any
	builderFn builderif.BuilderFn
}

// NewExec creates a new Exec object using the model object passed.
// It keeps a reference to the model object, and it will use it every time it needs to build a new computation graph.
//
// It returns an error if the model object does not have a valid Builder API.
func NewExec(model any) (*Exec, error) {
	builderFn, err := builderif.ConvertToBuilderFn(model)
	if err != nil {
		return nil, err
	}
	return &Exec{
		model:     model,
		builderFn: builderFn,
	}, nil
}
