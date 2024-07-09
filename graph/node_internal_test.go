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

package graph

// This file has tests that run as part of the graph package -- so with internal visibility.

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"testing"
)

// compileAndRun compiles, runs and returns the value on the tensor. Doesn't work for tuples though.
func compileAndRun(g *Graph) any {
	g.Compile()
	device := g.Run(nil)
	got := device.Local().Value()
	return got
}

// buildTestManager using "Host" by default -- can be overwritten by GOMLX_PLATFORM environment variable.
func buildTestManager() *Manager {
	return BuildManager().WithDefaultPlatform("Host").Done()
}

func TestBroadcastInDim(t *testing.T) {
	manager := buildTestManager()
	{
		g := manager.NewGraph("")
		input := Const(g, [][][]float32{{{1.1, 1.2}}}) // Shape [1, 1, 2]
		broadcastInDim(input, shapes.Make(shapes.Float32, 2, 1, 2), []int{0, 1, 2})
		got := compileAndRun(g)
		want := [][][]float32{{{1.1, 1.2}}, {{1.1, 1.2}}} // Shape [2, 1, 2].
		if !xslices.DeepSliceCmp(got, want, xslices.Equal[float32]) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}
