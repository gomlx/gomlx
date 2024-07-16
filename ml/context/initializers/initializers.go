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
	"math"
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
	if !rngState.Shape().Equal(newRngState.Shape()) {
		Panicf("updated rngState for the random number generator has invalid shape: %s (should be %s)",
			newRngState.Shape(), rngState.Shape())
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
//
// Non-float and non-complex variables are initialized with zero instead.
func RandomNormalFn(initialSeed int64, stddev float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if shape.DType != dtypes.Float32 && shape.DType != dtypes.Float64 {
			return Zeros(g, shape)
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
//
// Non-float and non-complex variables are initialized with zero instead.
func RandomUniformFn(initialSeed int64, min, max float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
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

// GlorotUniformFn return a Glorot uniform initializer, also called Xavier uniform initializer.
//
// It can be set to a context with `ctx.WithInitializer(GlorotUniformFn(initialSeed))`,
// where `initialSeed` can be 0 for a random seed to be generated.
//
// For float and complex values, it draws samples from a uniform distribution within
// `[-limit, limit]`, where `limit = sqrt(3 / ((fan_in + fan_out)/2))` (`fan_in` is the number of input units in
// the weight tensor and fan_out is the number of output units).
//
// Since it doesn't have semantic information about the variables being created, it makes
// some assumptions about the shapes of the variables: it assumes either these are weights
// for biases, matrix multiplications or 2D or 3D convolutions.
// Using it for different types of shapes may not get the expected result.
//
// The parameter `initialSeed` is used to initialize the random number generator -- only the first time it is
// used for a graph.
// If it is set to 0 (NoSeed), a random seed is instead generated (from the nanosecond clock).
//
// Non-float and non-complex variables are initialized with zero instead.
func GlorotUniformFn(initialSeed int64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		fanIn, fanOut := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn+fanOut)/2.0)
		limit := math.Sqrt(3.0 / scale)
		var values *Node
		useRngState(g, initialSeed, func(rngState *Node) (newRngState *Node) {
			newRngState, values = RandomUniform(rngState, shape)
			return newRngState
		})
		values = MulScalar(values, 2*limit)
		values = AddScalar(values, -limit)
		return values
	}
}

// computeFanInFanOut of a variable that is expected to be the parameters of
// either a [layers.Dense] or [layers.Convolution].
func computeFanInFanOut(shape shapes.Shape) (fanIn, fanOut int) {
	rank := shape.Rank()
	switch rank {
	case 0: // Scalar.
		fanIn = 1
		fanOut = fanIn
	case 1: // 1D shape, like a bias term in a dense layer.
		fanIn = shape.Dimensions[0]
		fanOut = fanIn
	case 2: // 2D shape, weights of a a dense layer.
		fanIn = shape.Dimensions[0]
		fanOut = shape.Dimensions[1]
	default: // Assuming convolution kernels (2D, 3D, or more):
		receptiveFieldSize := 1
		for dim := range shape.Dimensions[:rank-2] {
			receptiveFieldSize *= dim
		}
		fanIn = shape.Dimensions[rank-2] * receptiveFieldSize
		fanOut = shape.Dimensions[rank-1] * receptiveFieldSize
	}
	return
}
