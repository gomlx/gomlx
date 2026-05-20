// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package model implements a model's [Store] object (to store variables and
// hyperparameters), [Scope]'s (scope, like a directory "path") within a store,
// and [Exec], an executor that takes as an extra parameter a [Store] and will
// handle passing the variables automatically as extra inputs and outputs to the
// graph being built. Exec simplifies building models using or updateing the
// variables of its [Store].
//
// Graphs ([graph.Graph]) created when using [Exec] will have a link
// to its associated [Store]. You can retreive it using [GetStore].
//
// # Example: create a simple counter in a variable, and count to 10
//
//		func main() {
//			// Initialize a compute backend (e.g., using "go" or "xla:cpu").
//			backend, _ := compute.NewWithConfig(compute.DefaultConfig)
//			store := model.NewStore()  // Store to hold variables and hyperparameters.
//			modelFn := func(scope *model.Scope, g *graph.Graph) *graph.Node {
//				counterVar := scope.VariableWithValue("counter", 0)
//	            counter := graph.AddScalar(counterVar.NodeValue(g), 1)
//				counterVar.SetNodeValue(counter)
//				return newValue
//			}
//
//			exec := model.MustNewExec(backend, store, modelFn)  // Executor.
//			for range 10 {
//				fmt.Printf("Counter: %v\n", exec.MustCall1())  // Call with 1 output.
//			}
//		}
package model

//go:generate go run ../../internal/cmd/constraints_generator -model
