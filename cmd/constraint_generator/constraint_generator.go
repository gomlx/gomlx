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

package main

import (
	"fmt"
	"strings"
)

var baseTypes = []string{"int", "float32", "float64"}

const SliceLevels = 10

func TensorSlicesConstraints() {
	count := 0
	for slices := 0; slices < SliceLevels; slices++ {
		for _, t := range baseTypes {
			if count > 0 {
				fmt.Print(" | ")
			}
			for ii := 0; ii < slices; ii++ {
				fmt.Print("[]")
			}
			fmt.Print(t)
			count += 1
		}
	}
	fmt.Println()
}

func GraphExecFnConstraints() {
	var parts []string
	possibleInputs := []string{"*Graph", "[]*Node"}
	nodesParams := "*Node"
	for ii := 0; ii < 6; ii++ {
		possibleInputs = append(possibleInputs, nodesParams)
		nodesParams = nodesParams + ", *Node"
	}
	for _, outputs := range []string{"*Node", "(*Node, *Node)", "(*Node, *Node, *Node)", "[]*Node"} {
		for _, inputs := range possibleInputs {
			parts = append(parts, fmt.Sprintf("\t\tfunc (%s) %s", inputs, outputs))
		}
	}
	fmt.Printf("%s\n\n", strings.Join(parts, " |\n"))
}

func main() {
	TensorSlicesConstraints()
	fmt.Println()
	GraphExecFnConstraints()
}
