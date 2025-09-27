// Package models provide helpers to execute, save and load models with their weights and hyperparameters.
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
