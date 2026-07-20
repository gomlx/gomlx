// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
)

// //md_start:scopes
func denseLayer(scope *model.Scope, x *Node, outputDims int) *Node {
	g := x.Graph()
	dtype := x.DType()
	inputDims := x.Shape().Dimensions[1] // x shape is [batch, inputDims]

	// Create weights and biases in the current scope
	weights := scope.VariableWithShape("weights", shapes.Make(dtype, inputDims, outputDims)).NodeValue(g)
	biases := scope.VariableWithShape("biases", shapes.Make(dtype, 1, outputDims)).NodeValue(g)

	// Compute x * weights + biases
	return Add(Dot(x, weights).Product(), biases)
}

//md_end:scopes

func main() {
	backend := compute.MustNew()

	// Output to counter
	fmt.Println("md:counter")

	//md_start:counter(-1)
	store := model.NewStore()
	counterFn := func(scope *model.Scope, g *Graph) *Node {
		counterVar := scope.VariableWithValue("counter", int32(0))
		counter := AddScalar(counterVar.NodeValue(g), 1)
		counterVar.SetNodeValue(counter)
		return counter
	}

	exec := model.MustNewExec(backend, store, counterFn)
	fmt.Printf("Step 1: %v\n", exec.MustCall1())
	fmt.Printf("Step 2: %v\n", exec.MustCall1())
	fmt.Printf("Step 3: %v\n", exec.MustCall1())
	//md_end:counter

	// Output to scopes
	fmt.Println("md:scopes")

	//md:scopes(-1)
	//md_start:scopes
	modelFn := func(scope *model.Scope, x *Node) *Node {
		// Use scope.In to partition variable names under sub-scopes:
		h := denseLayer(scope.In("layer1"), x, 3) // variables: /layer1/weights, /layer1/biases
		y := denseLayer(scope.In("layer2"), h, 1) // variables: /layer2/weights, /layer2/biases
		return y
	}
	//md_end:scopes

	// Build and run the model execution to instantiate the variables
	modelExec := model.MustNewExec(backend, store, modelFn)
	// Input tensor with values [[1.0, 2.0]]
	inputTensor := tensors.FromValue([][]float32{{1.0, 2.0}})
	_ = modelExec.MustCall1(inputTensor)

	// Output to print_vars
	fmt.Println("md:print_vars")

	//md_start:print_vars
	// We can inspect all variables in the store:
	for v := range store.IterVariables() {
		fmt.Printf("Variable: %s, shape: %s\n", v.Path(), v.Shape())
	}
	//md_end:print_vars
}
