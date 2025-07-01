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

// Package initializers include several weight initializers, to be used with a context.Context.
//
// They construct computation.VariableInitializer closures.
package initializers

import (
	"math"
	"slices"

	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
)

var (
	// ParamInitialSeed is the key for the hyperparameter to use for initial seed (int64). The default is 0,
	// which makes it non-deterministic. Set it to a value different from 0 for a deterministic initialization
	// (as long as the model doesn't change).
	//
	// It is an alias to context.ParamInitialSeed.
	ParamInitialSeed = context.ParamInitialSeed
)

// VariableInitializer builds a node that returns a value to initialize a variable of the given
// shape. It is defined in the Context.
type VariableInitializer = context.VariableInitializer

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
//
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// Non-float and non-complex variables are initialized with zero instead.
func RandomNormalFn(ctx *context.Context, stddev float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if shape.DType != dtypes.Float32 && shape.DType != dtypes.Float64 {
			return Zeros(g, shape)
		}
		values := ctx.RandomNormal(g, shape)
		return MulScalar(values, stddev)
	}
}

// RandomUniformFn return an initializer that generates random uniform values from [min, max).
//
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// Non-float and non-complex variables are initialized with zero instead.
func RandomUniformFn(ctx *context.Context, min, max float64) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		values := ctx.RandomUniform(g, shape)
		values = MulScalar(values, max-min)
		values = AddScalar(values, min)
		return values
	}
}

// GlorotUniformFn return a Glorot uniform initializer, also called Xavier uniform initializer.
//
// It can be set to a context with `ctx.WithInitializer(GlorotUniformFn(ctx))`,
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
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func GlorotUniformFn(ctx *context.Context) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			// Zero-bias.
			return Zeros(g, shape)
		}
		fanIn, fanOut := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn+fanOut)/2.0)
		limit := math.Sqrt(3.0 / scale)
		values := ctx.RandomUniform(g, shape)
		values = MulScalar(values, 2*limit)
		values = AddScalar(values, -limit)
		return values
	}
}

// computeFanInFanOut of a variable expected to be the parameters of
// either layers.Dense or layers.Convolution.
func computeFanInFanOut(shape shapes.Shape) (fanIn, fanOut int) {
	rank := shape.Rank()
	switch rank {
	case 0: // Scalar.
		fanIn = 1
		fanOut = fanIn
	case 1: // 1D shape, like a bias term in a dense layer.
		fanIn = 0
		fanOut = fanIn
	case 2: // 2D shape, weights of a dense layer.
		fanIn = shape.Dimensions[0]
		fanOut = shape.Dimensions[1]
	default: // Assuming convolution kernels (2D, 3D, or more):
		receptiveFieldSize := 1
		for _, dim := range shape.Dimensions[:rank-2] {
			receptiveFieldSize *= dim
		}
		fanIn = shape.Dimensions[rank-2] * receptiveFieldSize
		fanOut = shape.Dimensions[rank-1] * receptiveFieldSize
	}
	return
}

// XavierUniformFn returns an initializer that generates random values with a uniform distribution with a range
// defined by +/- sqrt(6 / (fanIn+fanOut)). See description in https://paperswithcode.com/method/xavier-initialization
//
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func XavierUniformFn(ctx *context.Context) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			// Zero-bias.
			return Zeros(g, shape)
		}
		fanIn, fanOut := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn+fanOut))
		limit := math.Sqrt(6.0 / scale)
		values := ctx.RandomUniform(g, shape)
		values = MulScalar(values, 2*limit)
		values = AddScalar(values, -limit)
		return values
	}
}

// XavierNormalFn returns an initializer that generates random values with a normal distribution with mean in 0
// and stddev of sqrt(2 / (fanIn+fanOut)). See description in https://paperswithcode.com/method/xavier-initialization
//
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func XavierNormalFn(ctx *context.Context) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			// Zero-bias.
			return Zeros(g, shape)
		}
		fanIn, fanOut := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn+fanOut))
		stddev := math.Sqrt(2.0 / scale)
		values := ctx.RandomNormal(g, shape)
		return MulScalar(values, stddev)
	}
}

// HeFn returns the initializer that tries to preserve the variance of 1, calculated for the Relu activation functions.
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// It uses the context's ParamInitialSeed hyperparameter to initialize the random number generator --
// only the first time it is used for a graph.
// If it is set to 0 (NoSeed, the default), a random seed is instead generated (from the nanosecond clock).
//
// [1] https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d
// [2] https://arxiv.org/pdf/1502.01852
func HeFn(ctx *context.Context) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			// Zero-bias.
			return Zeros(g, shape)
		}
		fanIn, _ := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn))
		stddev := math.Sqrt(2.0 / scale)
		values := ctx.RandomNormal(g, shape)
		return MulScalar(values, stddev)
	}
}

// BroadcastTensorToShape is an initializer that takes a constant tensor as baseValue, and during initialization
// it broadcast it to the requested variable shape.
//
// The broadcasting happens only on the prefix dimensions (using graph.BroadcastPrefix), so the
// baseValue tensor's shape must match the last dimensions of the variable's shape.
//
// The baseValue can have a different dtype, in which case it is converted (using graph.ConvertDType) to the
// requested variable dtype.
//
// It also works with a scalar baseValue, which translates to constant value initializer.
func BroadcastTensorToShape(baseValue *tensors.Tensor) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		v := ConstCachedTensor(g, baseValue)
		v = ConvertDType(v, shape.DType)
		if v.Shape().Equal(shape) {
			return v
		}
		if shape.Rank() <= v.Rank() || !slices.Equal(v.Shape().Dimensions, shape.Dimensions[shape.Rank()-v.Rank():]) {
			Panicf("invalid BroadcastTensorToShape: variable being initialized has shape %s (rank %d), but base "+
				"tensor has shape %s (rank %d), which is not a suffix of the requested variable shape",
				shape, shape.Rank(), baseValue.Shape(), baseValue.Rank())
		}
		v = BroadcastPrefix(v, shape.Dimensions[:shape.Rank()-v.Rank()]...)
		return v
	}
}
