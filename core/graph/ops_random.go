// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"reflect"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/exceptions"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"golang.org/x/exp/constraints"
)

// RNGAlgorithm specifies the Pseudo-Random Number Generator (PRNG) algorithm used by
// [RNGBitGenerator], [RandomUniform], [RandomNormal], [RandomIntN], and [RNGStateSplit].
//
// This is an alias of [compute.RNGAlgorithm].
type RNGAlgorithm = compute.RNGAlgorithm

const (
	// RNGDefault selects the backend's preferred native Random Number Generator (RNG) algorithm.
	RNGDefault = compute.RNGDefault

	// RNGPhilox selects the counter-based Philox-4x32-10 PRNG ([3]uint64 state).
	RNGPhilox = compute.RNGPhilox

	// RNGThreeFry selects the counter-based ThreeFry-2x32-20 PRNG ([2]uint64 state).
	RNGThreeFry = compute.RNGThreeFry

	// RNGSplitMix64 selects the counter-based 64-bit SplitMix64 PRNG ([2]uint64 state).
	RNGSplitMix64 = compute.RNGSplitMix64

	// RNGPCG selects the 128-bit sequential PCG-DXSM PRNG ([2]uint64 state).
	RNGPCG = compute.RNGPCG

	// RNGChaCha8 selects the counter-mode 8-round ChaCha8 CSPRNG ([5]uint64 state).
	RNGChaCha8 = compute.RNGChaCha8
)

var (
	// RNGStateShape is the legacy Philox state shape ([3]uint64).
	//
	// Deprecated: use [RNGAlgorithm.StateShape] or `backend.Capabilities().RNGStateShape(...)` instead.
	RNGStateShape = compute.RNGStateShape

	// RNGAlgorithmString parses a snake-case string name into an [RNGAlgorithm].
	RNGAlgorithmString = compute.RNGAlgorithmString

	// RNGAlgorithmValues returns all valid [RNGAlgorithm] enum values.
	RNGAlgorithmValues = compute.RNGAlgorithmValues
)

// RNGStateFromSeed creates a deterministic Random Number Generator (RNG) state tensor
// for an explicit [RNGAlgorithm] initialized from seed.
//
// Because [RNGDefault] depends on the target backend, passing [RNGDefault] returns an error;
// use [RNGStateFromSeedForBackend] or [RNGStateFromSeedForGraph] instead.
func RNGStateFromSeed(algo RNGAlgorithm, seed int64) (*tensors.Tensor, error) {
	if algo == RNGDefault {
		return nil, errors.New("RNGStateFromSeed: RNGDefault depends on the backend; use RNGStateFromSeedForBackend or RNGStateFromSeedForGraph")
	}
	flat, _, err := algo.StateFromSeed(seed)
	if err != nil {
		return nil, err
	}
	return tensors.FromAnyValue(flat)
}

// RNGStateFromSeedForBackend creates a deterministic Random Number Generator (RNG) state tensor
// for algo initialized from seed, tailored to backend (including [RNGDefault]).
//
// If algo is an explicit counter-based algorithm not natively supported by backend, it still
// succeeds using the algorithm's canonical state shape so GoMLX's decomposed graph fallback can run it.
func RNGStateFromSeedForBackend(backend compute.Backend, algo RNGAlgorithm, seed int64) (*tensors.Tensor, error) {
	if backend == nil {
		return nil, errors.New("RNGStateFromSeedForBackend: backend must not be nil")
	}
	caps := backend.Capabilities()
	flat, _, err := caps.RNGStateFromSeed(algo, seed)
	if err != nil {
		if compute.IsNotImplemented(err) && algo.IsCounterBased() {
			flat, _, err = algo.StateFromSeed(seed)
			if err != nil {
				return nil, err
			}
			return tensors.FromAnyValue(flat)
		}
		return nil, err
	}
	return tensors.FromAnyValue(flat)
}

func randomStateForExplicitAlgo(algo RNGAlgorithm) (*tensors.Tensor, error) {
	switch algo {
	case RNGPhilox:
		var buf [8]byte
		if err := initializeRandomBytes(buf[:]); err != nil {
			return nil, err
		}
		return tensors.FromFlatDataAndDimensions([]uint64{binary.LittleEndian.Uint64(buf[:]), 0, 0}, 3), nil
	case RNGThreeFry, RNGSplitMix64:
		var buf [8]byte
		if err := initializeRandomBytes(buf[:]); err != nil {
			return nil, err
		}
		return tensors.FromFlatDataAndDimensions([]uint64{binary.LittleEndian.Uint64(buf[:]), 0}, 2), nil
	case RNGChaCha8:
		var buf [32]byte
		if err := initializeRandomBytes(buf[:]); err != nil {
			return nil, err
		}
		state := make([]uint64, 5)
		for i := range 4 {
			state[i] = binary.LittleEndian.Uint64(buf[i*8 : (i+1)*8])
		}
		state[4] = 0
		return tensors.FromFlatDataAndDimensions(state, 5), nil
	case RNGPCG:
		var buf [16]byte
		if err := initializeRandomBytes(buf[:]); err != nil {
			return nil, err
		}
		s1 := binary.LittleEndian.Uint64(buf[0:8])
		s2 := binary.LittleEndian.Uint64(buf[8:16])
		rng := rand.NewPCG(s1, s2)
		encoded, err := rng.MarshalBinary()
		if err != nil {
			return nil, errors.Wrap(err, "RNGState(RNGPCG): MarshalBinary failed")
		}
		hi := binary.BigEndian.Uint64(encoded[4:12])
		lo := binary.BigEndian.Uint64(encoded[12:20])
		return tensors.FromFlatDataAndDimensions([]uint64{hi, lo}, 2), nil
	default:
		return nil, errors.Errorf("RNGState: unsupported RNG algorithm %s", algo)
	}
}

// RNGState creates a Random Number Generator (RNG) state tensor initialized from the OS
// cryptographically secure random source for an explicit [RNGAlgorithm].
//
// Because [RNGDefault] depends on the target backend, passing [RNGDefault] returns an error;
// use [RNGStateForBackend] or [RNGStateForGraph] instead.
func RNGState(algo RNGAlgorithm) (*tensors.Tensor, error) {
	if algo == RNGDefault {
		return nil, errors.New("RNGState: RNGDefault depends on the backend; use RNGStateForBackend or RNGStateForGraph")
	}
	return randomStateForExplicitAlgo(algo)
}

// RNGStateForBackend creates a Random Number Generator (RNG) state tensor initialized from the OS
// cryptographically secure random source for algo, tailored to backend (including [RNGDefault]).
func RNGStateForBackend(backend compute.Backend, algo RNGAlgorithm) (*tensors.Tensor, error) {
	if backend == nil {
		return nil, errors.New("RNGStateForBackend: backend must not be nil")
	}
	caps := backend.Capabilities()
	resolved := caps.ResolveRNGAlgorithm(algo)
	if resolved != RNGDefault {
		if !caps.SupportsRNG(algo) && !algo.IsCounterBased() {
			return nil, errors.Wrapf(compute.ErrNotImplemented, "RNG algorithm %s is not supported by backend %q", algo, backend.Name())
		}
		return randomStateForExplicitAlgo(resolved)
	}

	shape, err := caps.RNGStateShape(algo)
	if err != nil {
		return nil, err
	}
	n := shape.Size()
	raw := make([]byte, n*shape.DType.Size())
	if err := initializeRandomBytes(raw); err != nil {
		return nil, err
	}
	switch shape.DType {
	case dtypes.Uint64:
		vals := make([]uint64, n)
		for i := range vals {
			vals[i] = binary.LittleEndian.Uint64(raw[i*8 : (i+1)*8])
		}
		return tensors.FromFlatDataAndDimensions(vals, shape.Dimensions...), nil
	case dtypes.Int64:
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = int64(binary.LittleEndian.Uint64(raw[i*8 : (i+1)*8]))
		}
		return tensors.FromFlatDataAndDimensions(vals, shape.Dimensions...), nil
	case dtypes.Uint32:
		vals := make([]uint32, n)
		for i := range vals {
			vals[i] = binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		}
		return tensors.FromFlatDataAndDimensions(vals, shape.Dimensions...), nil
	case dtypes.Int32:
		vals := make([]int32, n)
		for i := range vals {
			vals[i] = int32(binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4]))
		}
		return tensors.FromFlatDataAndDimensions(vals, shape.Dimensions...), nil
	default:
		return nil, errors.Errorf("RNGStateForBackend: unsupported DefaultRNGStateShape dtype %s", shape.DType)
	}
}

// RNGStateForGraph creates a Random Number Generator (RNG) state constant node in g for algo,
// initialized using the OS cryptographically secure random source.
func RNGStateForGraph(g *Graph, algo RNGAlgorithm) *Node {
	g.AssertValid()
	state, err := RNGStateForBackend(g.Backend(), algo)
	if err != nil {
		panic(err)
	}
	return Const(g, state)
}

// RNGStateFromSeedForGraph creates a Random Number Generator (RNG) state constant node in g
// for algo, initialized deterministically from seed.
func RNGStateFromSeedForGraph(g *Graph, algo RNGAlgorithm, seed int64) *Node {
	g.AssertValid()
	state, err := RNGStateFromSeedForBackend(g.Backend(), algo, seed)
	if err != nil {
		panic(err)
	}
	return Const(g, state)
}

// RNGBitGenerator generates a tensor of the given unsigned integer shape (Uint8, Uint16,
// Uint32, or Uint64) filled with uniformly distributed random bits using algo and returns
// the updated state alongside the generated values.
//
// If the backend does not natively implement an explicit counter-based algorithm
// ([RNGPhilox], [RNGThreeFry], [RNGSplitMix64], or [RNGChaCha8]), RNGBitGenerator
// automatically falls back to an equivalent decomposed graph implementation.
func RNGBitGenerator(algo RNGAlgorithm, rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := validateBuildingGraphFromInputs(rngState)
	if !shape.Ok() || !shape.DType.IsUnsigned() || shape.DType.Bits() < 8 {
		Panicf("RNGBitGenerator requires an unsigned integer bit dtype (Uint8, Uint16, Uint32, or Uint64), got shape %s", shape)
	}

	resolved := g.Backend().Capabilities().ResolveRNGAlgorithm(algo)
	var err error
	err = exceptions.TryCatch[error](func() {
		stateImpl, valImpl := graphimpl.BackendRNGBitGenerator(unwrapNode(rngState), shape, resolved)
		newRNGState, values = wrapNode(stateImpl), wrapNode(valImpl)
	})
	if err != nil {
		if compute.IsNotImplemented(err) && resolved.IsCounterBased() {
			newRNGState, values = rngBitGeneratorFallback(rngState, shape, resolved)
		} else {
			panic(err)
		}
	}
	return newRNGState, values
}

// RNGStateSplit splits rngState into 2 different states for algo that can be used
// separately and will lead to different random numbers.
func RNGStateSplit(algo RNGAlgorithm, rngState *Node) (newRNGState1, newRNGState2 *Node) {
	return RNGBitGenerator(algo, rngState, rngState.Shape())
}

// RandomUniform generates random uniform values from 0.0 to 1.0 (half-open `[0.0, 1.0)`, so 1.0 is never returned)
// for float or complex numbers in the given shape using algo and rngState.
//
// It will signal an error if the dtype is not float or complex -- see RandomIntN for random integers.
//
// For complex numbers, both the real and the imaginary part are independently sampled from `[0.0, 1.0)`.
//
// Alternatively, if you don't want to worry about carrying around the rngState, use the model.Store.RandomUniform
// version, which stores the rngState and algorithm as variables.
func RandomUniform(algo RNGAlgorithm, rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	_ = validateBuildingGraphFromInputs(rngState)
	if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
		Panicf(
			"RandomUniform only work with float or complex numbers, got shape %s instead -- see RandomIntN for integers",
			shape,
		)
	}

	switch shape.DType {
	case dtypes.Float64:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint64
		var randomBits *Node
		newRNGState, randomBits = RNGBitGenerator(algo, rngState, bitsShape)
		// Shift to keep 53 significand bits before converting to avoid rounding >= 2^64-1024 up to 1.0.
		values = ConvertDType(BitwiseShiftRightLogicalScalar(randomBits, 11), dtypes.Float64)
		values = MulScalar(values, math.Ldexp(1.0, -53))
		values = StopGradient(values)
	case dtypes.Float32:
		bitsShape := shape.Clone()
		bitsShape.DType = dtypes.Uint32
		var randomBits *Node
		newRNGState, randomBits = RNGBitGenerator(algo, rngState, bitsShape)
		// Shift to keep 24 significand bits before converting to avoid rounding up to 1.0.
		values = ConvertDType(BitwiseShiftRightLogicalScalar(randomBits, 8), dtypes.Float32)
		values = MulScalar(values, math.Ldexp(1.0, -24))
		values = StopGradient(values)
	case dtypes.Float16, dtypes.BFloat16:
		shapeF32 := shape.Clone()
		shapeF32.DType = dtypes.Float32
		newRNGState, values = RandomUniform(algo, rngState, shapeF32)
		values = ConvertDType(values, shape.DType)
		values = StopGradient(values)
	case dtypes.Complex64:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float32
		var re, im *Node
		newRNGState, re = RandomUniform(algo, rngState, componentShape)
		newRNGState, im = RandomUniform(algo, newRNGState, componentShape)
		values = Complex(re, im)
	case dtypes.Complex128:
		componentShape := shape.Clone()
		componentShape.DType = dtypes.Float64
		var re, im *Node
		newRNGState, re = RandomUniform(algo, rngState, componentShape)
		newRNGState, im = RandomUniform(algo, newRNGState, componentShape)
		values = Complex(re, im)
	default:
		Panicf(
			"RandomUniform() only accepts Float16, BFloat16, Float32, Float64, Complex64 and Complex128 dtypes, shapes %s given",
			shape,
		)
	}
	return
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0 and standard deviation 1.0,
// using algo and rngState.
//
// If you need a different mean and standard deviation, just do something like the example below, where `mean`
// and `stddev` are the desired mean and standard deviation:
//
//	rngState := RNGStateFromSeedForGraph(g, RNGDefault, 42)
//	rngState, values := RandomNormal(RNGDefault, rngState, shapes.Make(dtypes.Float32, 3, 2))
//	numbers = AddScalar(MulScalar(values, stddev), mean)
//
// It uses the Box-Muller algorithm (see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform).
// It has some numeric limitations, but works well for most purposes.
//
// It will signal an error if the dtype is not float.
// See also RandomIntN for random integers.
func RandomNormal(algo RNGAlgorithm, rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := validateBuildingGraphFromInputs(rngState)
	if !shape.DType.IsFloat() {
		Panicf(
			"RandomNormal only work with float numbers, got shape %s instead -- see RandomIntN for integers",
			shape,
		)
	}

	var u1, u2 *Node
	newRNGState, u1 = RandomUniform(algo, rngState, shape)
	// u1 must never be zero, so we take the smallest positive non-zero value.
	u1 = Max(u1, Const(g, shape.DType.SmallestNonZeroValueForDType()))
	newRNGState, u2 = RandomUniform(algo, newRNGState, shape)
	values = Mul(
		Sqrt(MulScalar(Log(u1), -2)),
		Cos(MulScalar(u2, 2*math.Pi)))
	return
}

// RandomIntN generates random numbers uniformly from 0 to N-1 using algo and rngState.
// It only works for integer types, see RandomUniform for float or complex data types.
// N can be given as a Node, or a static scalar integer value.
//
// Example:
//
//	rngState := RNGStateFromSeedForGraph(g, RNGDefault, 42)
//	rngState, D10 := RandomIntN(RNGDefault, rngState, 10, shapes.Make(dtypes.Int32))
func RandomIntN[IntT interface{ *Node | constraints.Integer }](
	algo RNGAlgorithm, rngState *Node, N IntT, shape shapes.Shape) (newRNGState, values *Node) {
	g := validateBuildingGraphFromInputs(rngState)
	if !shape.DType.IsInt() {
		Panicf(
			"RandomIntN only work with integer types, got shape %s instead -- see RandomUniform or RandomNormal for float/complex values",
			shape,
		)
	}

	var randomBits *Node
	randomBitsShape := shape.Clone()
	randomBitsShape.DType = dtypes.U64
	newRNGState, randomBits = RNGBitGenerator(algo, rngState, randomBitsShape)
	var ratio, maxValue *Node
	switch n := any(N).(type) {
	case *Node:
		ratio = Div(Const(g, uint64(math.MaxUint64)), ConvertDType(n, dtypes.U64))
		maxValue = ConvertDType(AddScalar(n, -1), shape.DType)
	default:
		nUint64 := reflect.ValueOf(n).Convert(reflect.TypeFor[uint64]()).Interface().(uint64)
		ratio = Scalar(g, dtypes.U64, uint64(math.MaxUint64)/nUint64)
		maxValue = Scalar(g, shape.DType, nUint64-1)
	}
	samples := ConvertDType(Div(randomBits, ratio), shape.DType)
	samples = Min(samples, maxValue) // There is an unlikely random chance of getting a value that will be equal to N.
	return newRNGState, samples
}
