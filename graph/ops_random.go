package graph

import (
	"math"
	"math/rand"
	"reflect"
	"time"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"golang.org/x/exp/constraints"
)

var (
	// RngStateShape is the shapes of the random number generator state, used
	// in all Random* functions.
	// This is dependent on the algorithm, that for now is fixed.
	RngStateShape = shapes.Make(dtypes.Uint64, 3)
)

// RngStateFromSeed creates a random number generator (RNG) state based on the static seed.
//
// Notice it returns a concrete tensor value that can be used to set a variable or
// constant to be used in a graph.
//
// Typical use case would be to use like:
//
//	rngState := Const(g, RngStateFromSeed(42))
func RngStateFromSeed(seed int64) *tensors.Tensor {
	rngSrc := rand.NewSource(seed)
	rng := rand.New(rngSrc)
	state := tensors.FromShape(RngStateShape)
	state.MutableFlatData(func(flatAny any) {
		flat := flatAny.([]uint64)
		for ii := range flat {
			flat[ii] = rng.Uint64()
		}
	})
	return state
}

// RngState creates a random number generator (RNG) state initialized from the nanosecond clock
// at the time of the graph creation.
//
// Notice it returns a concrete tensor value that can be used to set a variable or
// constant to be used in a graph.
//
// Typical use case would be to use like:
//
//	rngState := Const(g, RngState())
func RngState() *tensors.Tensor {
	return RngStateFromSeed(time.Now().UTC().UnixNano())
}

// RngStateSplit splits the current state into 2 different states that can be used
// separately and will lead to different random numbers.
func RngStateSplit(rngState *Node) (newRngState1, newRngState2 *Node) {
	validateRngState(rngState)
	return backendRngBitGenerator(rngState, rngState.Shape())
}

func validateRngState(rngState *Node) {
	if !rngState.Shape().Equal(RngStateShape) {
		Panicf("rngState is of the wrong shape (see graph.RngStateShape) -- pls create it with " +
			"something like `Const(g, graph.RngState())` or `Const(g, graph.RngStateFromSeed())`")
	}
}

// RandomUniform generates random uniform values from 0.0 to 1.0 (half-open `[0.0, 1.0)`, so 1.0 is never returned)
// for float numbers in the given shapes.
//
// It will signal an error if the dtype is not float -- see RandomIntN for random integers.
//
// For complex numbers, both the real and the imaginary part are independently sampled from `[0.0, 1.0)`.
//
// It uses and updates the random number generator (RNG) state in `rngState`.
// See RngStateFromSeed or RngState to generate a random state tensor (that can be fed to the computation graph).
//
// Alternatively, if you don't want to worry about carrying around the rngState, use the context.Context.RandomUniform
// version, which stores the rngState as a variable.
//
// Example:
//
//	rngState := Const(g, RngStateFromSeed(42))
//	rngState, values := RandomUniform(rngState, shapes.Make(dtypes.Float32, 3, 2))
func RandomUniform(rngState *Node, shape shapes.Shape) (newRngState, values *Node) {
	validateRngState(rngState)
	if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
		Panicf("RandomUniform only work with float or complex numbers, got shape %s instead -- see RandomIntN for integers", shape)
	}

	switch shape.DType {
	case dtypes.Float64:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint64
		var randomBits *Node
		newRngState, randomBits = backendRngBitGenerator(rngState, bitsShape)
		values = ConvertDType(randomBits, dtypes.Float64)
		values = MulScalar(values, math.Pow(2.0, -64))
		values = MinScalar(values, math.Nextafter(1.0, 0.0))
		values = StopGradient(values)
	case dtypes.Float32:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint32 // XLA will only generate `uint` for random bits.
		var randomBits *Node
		newRngState, randomBits = backendRngBitGenerator(rngState, bitsShape)
		values = ConvertDType(randomBits, dtypes.Float32)
		values = MulScalar(values, 1.0/(float64(1<<32)))
		values = Abs(values)
		values = MinScalar(values, float64(math.Nextafter32(1.0, 0.0)))
		values = StopGradient(values)
	case dtypes.Float16, dtypes.BFloat16:
		shapeF32 := shape.Clone()
		shapeF32.DType = dtypes.Float32
		newRngState, values = RandomUniform(rngState, shapeF32)
		values = ConvertDType(values, shape.DType)
		values = StopGradient(values)
	case dtypes.Complex64:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float32
		var re, im *Node
		newRngState, re = RandomUniform(rngState, componentShape)
		newRngState, im = RandomUniform(rngState, componentShape)
		values = Complex(re, im)
	case dtypes.Complex128:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float64
		var re, im *Node
		newRngState, re = RandomUniform(rngState, componentShape)
		newRngState, im = RandomUniform(rngState, componentShape)
		values = Complex(re, im)
	default:
		Panicf("RandomUniform() only accepts Float16, Float32, Float64, Complex64 and Complex128 dtypes, shapes %s given", shape)
	}
	return
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0 and standard deviation 1.0.
// It generates values with the given shapes, each value pseudo-randomly generated.
//
// If you need a different mean and standard deviation, just do something like the example below, where `mean`
// and `stddev` are the desired mean and standard deviation:
//
//	rngState := Const(g, RngStateFromSeed(42))
//	rngState, values := RandomNormal(rngState, shapes.Make(dtypes.Float32, 3, 2))
//	numbers = AddScalar(MulScalar(values, stddev), mean)
//
// It uses the Box-Muller algorithm (see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform).
// It has some numeric limitations, but works well for most purposes.
//
// It will signal an error if the dtype is not float.
// See also RandomIntN for random integers.
//
// It uses and updates the random number generator (RNG) state in `rngState`.
// See [RngStateFromSeed] or [RngState] to generate a random state tensor (that can be fed to the computation graph).
//
// Alternatively, if you don't want to worry about carrying around the rngState, use the context.Context.RandomNormal
// version, which stores the rngState as a variable.
func RandomNormal(rngState *Node, shape shapes.Shape) (newRngState, values *Node) {
	validateRngState(rngState)
	if !shape.DType.IsFloat() {
		Panicf("RandomNormal only work with float or complex numbers, got shape %s instead -- see RandomIntN for integers", shape)
	}

	g := rngState.Graph()
	var u1, u2 *Node
	newRngState, u1 = RandomUniform(rngState, shape)
	// u1 must never be zero, so we take the smallest positive non-zero value.
	u1 = Max(u1, Const(g, shape.DType.SmallestNonZeroValueForDType()))
	newRngState, u2 = RandomUniform(newRngState, shape)
	values = Mul(
		Sqrt(MulScalar(Log(u1), -2)),
		Cos(MulScalar(u2, 2*math.Pi)))
	return
}

// RandomIntN generates random numbers uniformly from 0 to N-1. It only works for integer types, see RandomUniform for
// float or complex data types. N can be given as a Node, or a static scalar integer value.
//
// Example:
//
//	rngState := Const(g, RngStateFromSeed(42))
//	rngState, D10 := RandomIntN(rngState, 10, shapes.Make(dtypes.Int32))
//
// It uses and updates the random number generator (RNG) state in `rngState`.
// See [RngStateFromSeed] or [RngState] to generate a random state tensor (that can be fed to the computation graph).
//
// Alternatively, if you don't want to worry about carrying around the rngState, use the context.Context.RandomIntN
// version, which stores the rngState as a variable.
func RandomIntN[IntT interface{ *Node | constraints.Integer }](
	rngState *Node, N IntT, shape shapes.Shape) (newRngState, values *Node) {
	validateRngState(rngState)
	if !shape.DType.IsInt() {
		Panicf("RandomIntN only work with integer types, got shape %s instead -- see RandomUniform or RandomNormal for float/complex values", shape)
	}

	g := rngState.Graph()
	var randomBits *Node
	randomBitsShape := shape.Clone()
	randomBitsShape.DType = dtypes.U64
	newRngState, randomBits = backendRngBitGenerator(rngState, randomBitsShape)
	var ratio, maxValue *Node
	switch n := any(N).(type) {
	case *Node:
		ratio = Div(Const(g, uint64(math.MaxUint64)), ConvertDType(n, dtypes.U64))
		maxValue = ConvertDType(AddScalar(n, -1), shape.DType)
	default:
		nUint64 := reflect.ValueOf(n).Convert(reflect.TypeOf(uint64(0))).Interface().(uint64)
		ratio = Scalar(g, dtypes.U64, uint64(math.MaxUint64)/nUint64)
		maxValue = Scalar(g, shape.DType, nUint64-1)
	}
	samples := ConvertDType(Div(randomBits, ratio), shape.DType)
	samples = Min(samples, maxValue) // There is an unlikely random chance of getting a value that will be equal to N.
	return newRngState, samples
}
