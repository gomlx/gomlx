package context

import (
	"math"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
)

// VariableInitializer builds a valueNode that returns a value to initialize a variable of the given
// shape. It is defined in the Context.
type VariableInitializer = func(g *graph.Graph, shape shapes.Shape) *Node

// computeFanInFanOut of a variable expected to be the parameters of
// either layers.Dense or layers.Convolution.
//
// Copied from initializers package to set the DefaultInitializer with a good value.
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

// heInitializer returns the initializer that tries to preserve the variance of 1, calculated for the Relu activation functions.
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// [1] https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d
// [2] https://arxiv.org/pdf/1502.01852
//
// Copy from package initializers, used to populate the DefaultInitializer.
//
// It uses the context random state (as opposed to initializers.HeFn which uses initializers own RNG state).
func heInitializer(ctx *Context) VariableInitializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return graph.Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			// Zero-bias.
			return graph.Zeros(g, shape)
		}
		fanIn, _ := computeFanInFanOut(shape)
		scale := max(1.0, float64(fanIn))
		stddev := math.Sqrt(2.0 / scale)
		values := ctx.RandomNormal(g, shape)
		return graph.MulScalar(values, stddev)
	}
}

// DefaultInitializer is used whenever a new context is created to create a new VariableInitializer.
// You can always set your own initializer with Context.WithInitializer.
//
// See package initializers for various standard initializers.
//
// It defaults to a He initializer (https://arxiv.org/pdf/1502.01852)
var DefaultInitializer func(ctx *Context) VariableInitializer = heInitializer
