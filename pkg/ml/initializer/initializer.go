package initializer

import (
	"math"
	"slices"

	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/model"
	"github.com/gomlx/gomlx/pkg/ml/random"
	"github.com/gomlx/gopjrt/dtypes"
)

// Initializer is the interface for a variable initializer, defined in model.VariableInitializer as
//
//	func(g *graph.Graph, shape shapes.Shape) *graph.Node
type Initializer = model.VariableInitializer

var (
	// Zero initializes variables with zero.
	Zero Initializer = func(graph *Graph, shape shapes.Shape) *Node {
		return Zeros(graph, shape)
	}

	// One initializes variables with one.
	One Initializer = func(graph *Graph, shape shapes.Shape) *Node {
		return Ones(graph, shape)
	}
)

// Normal returns an initializer that generates random normal values with the given standard deviation
// and mean set to 0.
//
// Non-float and non-complex numbers are initialized to 0 instead.
//
// Complex numbers have both the real and imaginary parts sampled uniformly in the same range.
func Normal(rng *random.Random, stddev float64) Initializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		switch shape.DType {
		case dtypes.Float32, dtypes.Float64:
			values := rng.Normal(g, shape)
			return MulScalar(values, stddev)
		case dtypes.Complex64, dtypes.Complex128:
			realShape := shape.Clone()
			realShape.DType = shape.DType.RealDType()
			r, i := rng.Normal(g, realShape), rng.Normal(g, realShape)
			r = MulScalar(r, stddev)
			i = MulScalar(i, stddev)
			return Complex(r, i)
		default:
			return Zeros(g, shape)
		}
	}
}

// Uniform returns an initializer that generates random uniform values from [min, max).
//
// Non-float and non-complex variables are initialized with zero instead.
func Uniform(rng *random.Random, minValue, maxValue float64) Initializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		switch shape.DType {
		case dtypes.Float16, dtypes.BF16, dtypes.Float32, dtypes.Float64, dtypes.Complex64, dtypes.Complex128:
			value := rng.Uniform(g, shape)
			return AddScalar(MulScalar(value, maxValue-minValue), minValue)
		default:
			return Zeros(g, shape)
		}
	}
}

// GlorotUniform returns a Glorot uniform initializer, also called Xavier uniform initializer.
//
// For float and complex values, it draws samples from a uniform distribution within
// `[-limit, limit]`, where `limit = sqrt(3 / ((fan_in + fan_out)/2))` (`fan_in` is the number of input units in
// the weight tensor and fan_out is the number of output units).
//
// Since it doesn't have semantic information about the variables being created, it makes
// some assumptions about the shapes of the variables: it assumes either these are weights
// for biases, matrix multiplications, or 2D or 3D convolutions.
// Using it for different shapes may not get the expected result.
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func GlorotUniform(rng *random.Random) Initializer {
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
		values := rng.Uniform(g, shape)
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

// XavierUniform returns an initializer that generates random values with a uniform distribution with a range
// defined by +/- sqrt(6 / (fanIn+fanOut)).
// See paper and reasoning in https://paperswithcode.com/method/xavier-initialization
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func XavierUniform(rng *random.Random) Initializer {
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
		values := rng.Uniform(g, shape)
		values = MulScalar(values, 2*limit)
		values = AddScalar(values, -limit)
		return values
	}
}

// XavierNormal returns an initializer that generates random values with a normal distribution with mean in 0
// and stddev of sqrt(2 / (fanIn+fanOut)).
// See description in https://paperswithcode.com/method/xavier-initialization
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// Non-float and non-complex variables are initialized with zero instead.
func XavierNormal(rng *random.Random) Initializer {
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
		values := rng.Normal(g, shape)
		return MulScalar(values, stddev)
	}
}

// He returns the initializer that tries to preserve the variance of 1, calculated for the Relu activation functions.
//
// It initializes biases (anything with rank <= 1) to zeros.
//
// [1] https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d
// [2] https://arxiv.org/pdf/1502.01852
func He(rng *random.Random) Initializer {
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
		values := rng.Normal(g, shape)
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
func BroadcastTensorToShape(baseValue *tensors.Tensor) Initializer {
	return func(g *Graph, shape shapes.Shape) *Node {
		v := ConstCachedTensor(g, baseValue)
		v = ConvertDType(v, shape.DType)
		if v.Shape().Equal(shape) {
			return v
		}
		if shape.Rank() <= v.Rank() || !slices.Equal(v.Shape().Dimensions, shape.Dimensions[shape.Rank()-v.Rank():]) {
			exceptions.Panicf("invalid BroadcastTensorToShape: variable being initialized has shape %s (rank %d), but base "+
				"tensor has shape %s (rank %d), which is not a suffix of the requested variable shape",
				shape, shape.Rank(), baseValue.Shape(), baseValue.Rank())
		}
		v = BroadcastPrefix(v, shape.Dimensions[:shape.Rank()-v.Rank()]...)
		return v
	}
}
