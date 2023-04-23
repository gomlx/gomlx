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
// It defines NodeType for its internal representation.
//
// But mostly this is only needed for those implementing new ops, in which case they need to define
// the new type here.

// NodeType enumerate the various types of Nodes that can be converted to XLA.
type NodeType int32

//go:generate stringer -type=NodeType node.go

// XlaWrapperVersion is the version of library. It should match the C++ one, if they are out-of-sync
// very odd mistakes happen.
//
// Please bump whenever a new NodeType is created, and keep the C++ (in `node.h`) and Go version numbers
// in sync.
const XlaWrapperVersion = 10

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

	// Nodes with variable sets of arguments.

	RngNormalNode
	RngUniformNode
)
