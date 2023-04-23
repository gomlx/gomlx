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

// RandomNormalFn returns an initializer that generates random normal values with the given standard deviation
// and mean set to 0.
func RandomNormalFn(stddev float64) VariableInitializer {
	stddevClosure := stddev
	return func(graph *Graph, shape shapes.Shape) *Node {
		if shape.DType != shapes.F32 && shape.DType != shapes.F64 {
			graph.SetErrorf("cannot initialize non-float variable with RandomNormal -- shape requested %s", shape)
			return nil
		}
		mu := Zeros(graph, shapes.Make(shape.DType))
		sigma := Const(graph, shapes.CastAsDType(stddevClosure, shape.DType))
		return RngNormal(mu, sigma, shape)
	}
}

// RandomUniformFn return an initializer that generates a random uniform values from [min, max].
func RandomUniformFn(min, max float64) VariableInitializer {
	minClosure, maxClosure := min, max
	return func(graph *Graph, shape shapes.Shape) *Node {
		if shape.DType != shapes.F32 && shape.DType != shapes.F64 {
			graph.SetErrorf("cannot initialize non-float variable with RandomUniform -- shape requested %s", shape)
			return nil
		}
		min := Const(graph, shapes.CastAsDType(minClosure, shape.DType))
		max := Const(graph, shapes.CastAsDType(maxClosure, shape.DType))
		return RngUniform(min, max, shape)
	}
}
