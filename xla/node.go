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

package xla

// This file defines an intermediary representation for computation.Graph that
// can be communicated to the xla package, and in turn to the XLA C++ API.
//
// Internal implementation only.
//
// It defines Type for its internal representation.
//
// But mostly this is only needed for those implementing new ops, in which case they need to define
// the new type here.

// NodeType enumerate the various types of Nodes that can be converted to XLA.
type NodeType int32

//go:generate stringer -type=NodeType node.go

// XlaWrapperVersion is the version of the library.
// It should match the C++ one, if they are out-of-sync very odd mistakes happen.
//
// Please bump whenever a new NodeType is created, and keep the C++ (in `c/gomlx/status.h`) and Go version numbers
// in sync.
const XlaWrapperVersion = 14

// NodeType values need to be exactly the same as defined in the C++ code, in `c/gomlx/node.h`
// TODO: keep those in sync using some generator script.
const (
	InvalidNode NodeType = iota

	// Special ops:

	ConstantNode
	IotaNode
	ParameterNode
	ConvertTypeNode
	WhereNode
	TupleNode
	GetTupleElementNode
	ReshapeNode
	BroadcastNode
	BroadcastInDimNode
	ReduceSumNode
	ReduceMaxNode
	ReduceMultiplyNode
	SliceNode
	PadNode
	GatherNode
	ScatterNode
	ConcatenateNode
	ConvGeneralDilatedNode
	ReverseNode
	TransposeNode
	ReduceWindowNode
	SelectAndScatterNode
	BatchNormTrainingNode
	BatchNormInferenceNode
	BatchNormGradNode
	DotGeneralNode
	ArgMinMaxNode
	FftNode

	// One-argument ops:

	AbsNode
	NegNode
	ExpNode
	Expm1Node
	FloorNode
	CeilNode
	RoundNode
	LogNode
	Log1pNode
	LogicalNotNode
	LogisticNode
	SignNode
	ClzNode
	CosNode
	SinNode
	TanhNode
	SqrtNode
	RsqrtNode
	ImagNode
	RealNode
	ConjNode

	// Two-arguments ops:

	AddNode
	MulNode
	SubNode
	DivNode
	RemNode // Notice XLA implements Mod, not IEEE754 Remainder operation.
	AndNode
	OrNode
	XorNode
	DotNode
	MinNode
	MaxNode
	PowNode
	ComplexNode

	// Two-arguments comparison ops:

	EqualNode
	NotEqualNode
	GreaterOrEqualNode
	GreaterThanNode
	LessOrEqualNode
	LessThanNode
	EqualTotalOrderNode
	NotEqualTotalOrderNode
	GreaterOrEqualTotalOrderNode
	GreaterThanTotalOrderNode
	LessOrEqualTotalOrderNode
	LessThanTotalOrderNode

	// Nodes with variable sets of arguments:

	RngBitGeneratorNode
	RngNormalNode
	RngUniformNode
)

// RandomAlgorithm should be aligned with constants in `xla_data.proto` file is the XLA code base.
//
// Each random algorithm entails a different shape of `initialState` that needs to be fed to `RngBitGenerator`.
//
// See details and reference of the algorithms: https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator
//
// Unfortunately there is no documented way of figuring out what is the initial state for `RngDefault` algorithm.
type RandomAlgorithm int

//go:generate stringer -type=RandomAlgorithm node.go

const (
	// RngDefault is the back-end specific algorithm with back-end specific shape requirements.
	// There doesn't seem to be any automatic way of figuring out what it takes for `initialState`.
	RngDefault RandomAlgorithm = iota

	// RngThreeFry counter-based PRNG algorithm. The initial_state shape is U64[2] with arbitrary values.
	// [Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
	RngThreeFry

	// RngPhilox algorithm to generate random numbers in parallel. The initial_state shape is `U64[3]` with arbitrary values.
	// [Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
	RngPhilox
)

// FftType should be aligned with the constants in `xla_data.proto` file in the XLA code base.
//
// The XLA's FFT operator can work i different ways, this defines how.
type FftType int

//go:generate stringer -type=FftType node.go

const (
	// FftForward does a forward FFT: complex in, complex out.
	// FFT in the proto.
	FftForward FftType = iota

	// FftInverse does an inverse FFT: complex in, complex out.
	// IFFT in the proto.
	FftInverse

	// FftForwardReal does a forward real FFT: real in, fft_length / 2 + 1 complex out
	// RFFT in the proto.
	FftForwardReal

	// FftInverseReal does an inverse real FFT: fft_length / 2 + 1 complex in, real out
	// IRFFT in the proto.
	FftInverseReal
)
