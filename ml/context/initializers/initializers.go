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

// Package initializers include several weight initializers, to be used with context.
// They implement computation.VariableInitializer type.
package initializers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"sync"
)

// VariableInitializer builds a node that returns a value to initialize a variable of the given
// shape. It is defined in the Context.
type VariableInitializer func(graph *Graph, shape shapes.Shape) *Node

// Zero initializes variables with zero.
func Zero(graph *Graph, shape shapes.Shape) *Node {
	return Zeros(graph, shape)
}

// One initializes variables with one.
func One(graph *Graph, shape shapes.Shape) *Node {
	return Ones(graph, shape)
}

var (
	muRngStates sync.Mutex
	rngStates   = make(map[GraphId]*Node)
)

// Finalize will clear the global state kept alive and free up the memory.
// Namely, the random number generator states.
//
// Used for testing and debugging.
func Finalize() {
	muRngStates.Lock()
	defer muRngStates.Unlock()
	rngStates = make(map[GraphId]*Node)
}

// useRngState will provide a state for the current graph, and will update it with
// the returned updated state.
//
// It locks the state, so only one call to it will be happening at a time.
//
// If the graph has an error, the state is not updated.
func useRngState(g *Graph, initialSeed int64, fn func(rngState *Node) (newRngState *Node)) {
	g.AssertValid()
	muRngStates.Lock()
	defer muRngStates.Unlock()

	graphId := g.GraphId()
	rngState, found := rngStates[graphId]
	if !found {
		if initialSeed != 0 {
			rngState = Const(g, RngStateFromSeed(initialSeed))
		} else {
			rngState = Const(g, RngState())
		}
	}
	newRngState := fn(rngState)
	if !rngState.Shape().Eq(newRngState.Shape()) {
		g.SetErrorf("updated rngState for the random number generator has invalid shape: %s (should be %s)",
			newRngState.Shape(), rngState.Shape())
		return
	}
	rngStates[graphId] = newRngState
}

const NoSeed = int64(0)

// RandomNormalFn returns an initializer that generates random normal values with the given standard deviation
// and mean set to 0.
//
// The parameter `initialSeed` is used to initialize the random number generator -- only the first time it is
// used for a graph.
// If it is set to 0 (NoSeed), a random seed is instead generated (from the nanosecond clock).
func RandomNormalFn(initialSeed int64, stddev float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if shape.DType != shapes.F32 && shape.DType != shapes.F64 {
			g.SetErrorf("cannot initialize non-float variable with RandomNormal -- shape requested %s", shape)
			return nil
		}
		var values *Node
		useRngState(g, initialSeed, func(rngState *Node) (newRngState *Node) {
			newRngState, values = RandomNormal(rngState, shape)
			return newRngState
		})
		return MulScalar(values, stddev)
	}
}

// RandomUniformFn return an initializer that generates a random uniform values from [min, max).
//
// The parameter `initialSeed` is used to initialize the random number generator -- only the first time it is
// used for a graph.
// If it is set to 0 (NoSeed), a random seed is instead generated (from the nanosecond clock).
func RandomUniformFn(initialSeed int64, min, max float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if shape.DType != shapes.F32 && shape.DType != shapes.F64 {
			g.SetErrorf("cannot initialize non-float variable with RandomUniform -- shape requested %s", shape)
			return nil
		}
		var values *Node
		useRngState(g, initialSeed, func(rngState *Node) (newRngState *Node) {
			newRngState, values = RandomUniform(rngState, shape)
			return newRngState
		})
		values = MulScalar(values, max-min)
		values = AddScalar(values, min)
		return values
	}
}
