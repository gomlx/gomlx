/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// constraint_generator prints out various lists of constraints used by generics,
// which can then be copy&pasted into the code.
// It is an internal tool, meant to be used by developers of GoMLX.
package main

import (
	"fmt"
	"strings"
)

var baseTypes = []string{
	"bool", "float32", "float64", "int", "int32", "int64", "uint8", "uint32", "uint64", "complex64", "complex128"}

const SliceLevels = 7

func TensorSlicesConstraints() {
	count := 0
	for slices := 0; slices < SliceLevels; slices++ {
		fmt.Printf("\t")
		for idxType, t := range baseTypes {
			if idxType > 0 {
				fmt.Print(" | ")
			}
			for ii := 0; ii < slices; ii++ {
				fmt.Print("[]")
			}
			fmt.Print(t)
			count += 1
		}
		if slices < SliceLevels-1 {
			fmt.Printf(" |")
		}
		fmt.Println()
	}
}

func GraphExecFnConstraints(withContext bool) {
	var parts []string
	possibleInputs := []string{"*Graph", "[]*Node"}
	nodesParams := "*Node"
	for ii := 0; ii < 6; ii++ {
		possibleInputs = append(possibleInputs, nodesParams)
		nodesParams = nodesParams + ", *Node"
	}
	possibleOutputs := []string{}
	if withContext {
		possibleOutputs = append(possibleOutputs, "")
	}
	possibleOutputs = append(possibleOutputs, "*Node", "(*Node, *Node)", "(*Node, *Node, *Node)", "[]*Node")
	var contextInput string
	if withContext {
		contextInput = "*Context, "
	}
	for _, outputs := range possibleOutputs {
		if outputs != "" {
			outputs = " " + outputs
		}
		for _, inputs := range possibleInputs {
			parts = append(parts, fmt.Sprintf("\tfunc (%s%s)%s", contextInput, inputs, outputs))
		}
	}
	fmt.Printf("%s\n", strings.Join(parts, " |\n"))
}

func main() {
	fmt.Println("type Supported interface {")
	fmt.Printf("\t%s\n", strings.Join(baseTypes, " | "))
	fmt.Println("}")
	fmt.Println()

	fmt.Println("type MultiDimensionSlice interface {")
	TensorSlicesConstraints()
	fmt.Println("}")
	fmt.Println()

	fmt.Printf("// For graph/exec.go:\n\n")
	fmt.Println("type ExecGraphFn interface {")
	GraphExecFnConstraints(false)
	fmt.Println("}")
	fmt.Println()

	fmt.Printf("// For ml/context/exec.go:\n\n")
	fmt.Println("type ExecGraphFn interface {")
	GraphExecFnConstraints(true)
	fmt.Println("}")
}
