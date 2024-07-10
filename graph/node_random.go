package graph

import (
	"math"
	"math/rand"
	"time"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
)

var (
	// RngAlgorithm used in all Random* functions. XLA supports more than one, but we didn't
	// want to make the API more complex for now.
	// It may change in the future.
	RngAlgorithm = xla.RngPhilox

	// RngStateShape is the shape of the random number generator state, used
	// in all Random* functions.
	// This is dependent on the algorithm, that for now is fixed.
	RngStateShape = shapes.Make(dtypes.Uint64, 3)
)

// RngStateFromSeed creates a random number generator (RNG) state based on the static seed.
//
// Notice it returns a concrete tensor value that can be used to set a variable or
// constant to be used in a graph.
func RngStateFromSeed(seed int64) tensors.Tensor {
	rngSrc := rand.NewSource(seed)
	rng := rand.New(rngSrc)
	state := tensors.FromShape(RngStateShape)
	ref := state.AcquireData()
	defer ref.Release()
	data := ref.Flat().([]uint64)
	for ii := range data {
		data[ii] = rng.Uint64()
	}
	return state
}

// RngState creates a random number generator (RNG) state initialized from the nanosecond clock
// at the time of the graph creation.
//
// Notice it returns a concrete tensor value that can be used to set a variable or
// constant to be used in a graph.
func RngState() tensors.Tensor {
	return RngStateFromSeed(time.Now().UTC().UnixNano())
}

// RngBitGeneratorXLA generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RngState or RngStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func RngBitGeneratorXLA(state *Node, shape shapes.Shape) (newState, values *Node) {
	g := validateGraphFromInputs(state)
	pair := newNode(g, &xla.SerializedNode{
		Type:  xla.RngBitGeneratorNode,
		Shape: shape,
	}, []*Node{state})
	parts := SplitTuple(pair)
	if len(parts) != 2 {
		Panicf("xla.RngBitGeneratorNode returned %d components, but only 2 expected!?", len(parts))
	}
	newState, values = parts[0], parts[1]
	return
}

// RngStateSplit splits the current state into 2 different states that can be used
// separately and will lead to different random numbers.
func RngStateSplit(rngState *Node) (newRngState1, newRngState2 *Node) {
	return RngBitGeneratorXLA(rngState, rngState.Shape())
}

// RandomUniform generates random uniform values from 0.0 to 1.0 (half-open `[0.0, 1.0)`, so 1.0 is never returned)
// for float numbers in the given shape.
//
// It will signal an error if the dtype is not float -- see RandomIntN for random integers.
//
// For complex numbers, both the real and the imaginary part are independently sampled from `[0.0, 1.0)`.
//
// It uses and updates the random number generator (RNG) state in `rngState`.
func RandomUniform(rngState *Node, shape shapes.Shape) (newRngState, values *Node) {
	switch shape.DType {
	case dtypes.Float64:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint64
		var randomBits *Node
		newRngState, randomBits = RngBitGeneratorXLA(rngState, bitsShape)
		values = ConvertType(randomBits, dtypes.Float64)
		values = MulScalar(values, math.Pow(2.0, -64))
		values = MinScalar(values, math.Nextafter(1.0, 0.0))
		values = StopGradient(values)
	case dtypes.Float32:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint32
		var randomBits *Node
		newRngState, randomBits = RngBitGeneratorXLA(rngState, bitsShape)
		values = ConvertType(randomBits, dtypes.Float32)
		values = MulScalar(values, 1.0/(float64(1<<32)))
		values = MinScalar(values, float64(math.Nextafter32(1.0, 0.0)))
		values = StopGradient(values)
	case dtypes.Float16:
		shapeF32 := shape.Clone()
		shapeF32.DType = dtypes.Float32
		newRngState, values = RandomUniform(rngState, shapeF32)
		values = ConvertType(values, shape.DType)
		values = StopGradient(values)
	case shapes.Complex64:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float32
		var re, im *Node
		newRngState, re = RandomUniform(rngState, componentShape)
		newRngState, im = RandomUniform(rngState, componentShape)
		values = Complex(re, im)
	case shapes.Complex128:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float64
		var re, im *Node
		newRngState, re = RandomUniform(rngState, componentShape)
		newRngState, im = RandomUniform(rngState, componentShape)
		values = Complex(re, im)
	default:
		Panicf("RandomUniform() only accepts Float16, Float32, Float64, Complex64 and Complex128 dtypes, shape %s given", shape)
	}
	return
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0 and standard deviation 1.0.
// It generates values with the given shape, each value pseudo-randomly generated.
//
// If you need a different mean and standard deviation, just do something like the example below, where `mean`
// and `stddev` are the desired mean and standard deviation:
//
//	rngState, numbers = RandomNormal(rngState, myShape)
//	numbers = AddScalar(MulScalar(numbers, stddev), mean)
//
// It uses the Box-Muller algorithm (see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), which
// has some numeric limitations, but works well for most purposes.
//
// It will signal an error if the dtype is not float -- see RandomIntN for random integers.
//
// It uses and updates the random number generator (RNG) state in `rngState`.
//
// See [RngStateFromSeed] or [RngState] to generate a random state tensor (that can be fed to the computation graph).
func RandomNormal(rngState *Node, shape shapes.Shape) (newRngState, values *Node) {
	g := rngState.Graph()
	var u1, u2 *Node
	newRngState, u1 = RandomUniform(rngState, shape)
	// u1 must never be zero, so we take the smallest positive non-zero value.
	u1 = Max(u1, Const(g, shape.DType.SmallestNonZeroValue()))
	newRngState, u2 = RandomUniform(newRngState, shape)
	values = Mul(
		Sqrt(MulScalar(Log(u1), -2)),
		Cos(MulScalar(u2, 2*math.Pi)))
	return
}
