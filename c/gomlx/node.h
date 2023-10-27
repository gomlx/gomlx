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

// node.h holds C API structure to serialized nodes, and its NodeType enum.
// It also includes the `Shape` struct, a wrapper for `xla::Shape`,
#ifndef _GOMLX_XLA_NODE_H
#define _GOMLX_XLA_NODE_H

#include "gomlx/shape.h"
#include <stdlib.h>

#ifdef __cplusplus
// C++ only includes: these are not seen by the Go compiler.
#include "xla/client/xla_builder.h"
#include "xla/shape.h"

typedef xla::Literal XlaLiteral;
typedef xla::XlaOp XlaOp;

#else
// C and CGO only code.
typedef _Bool bool;
typedef void XlaLiteral;
typedef void XlaOp;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef XlaOp *XlaOpPtr;
struct Literal;

// NodeType has to be aligned with Go corresponding ir.NodeType.
// If changed here, pls bump the XlaWrapperVersion in status.h.
// TODO: keep those in sync using some generator script.
enum NodeType {
  InvalidNode,

  // Special ops:
  ConstantNode,
  IotaNode,
  ParameterNode,
  ConvertTypeNode,
  WhereNode,
  TupleNode,
  GetTupleElementNode,
  ReshapeNode,
  BroadcastNode,
  BroadcastInDimNode,
  ReduceSumNode,
  ReduceMaxNode,
  ReduceMultiplyNode,
  SliceNode,
  PadNode,
  GatherNode,
  ScatterNode,
  ConcatenateNode,
  ConvGeneralDilatedNode,
  ReverseNode,
  TransposeNode,
  ReduceWindowNode,
  SelectAndScatterNode,
  BatchNormTrainingNode,
  BatchNormInferenceNode,
  BatchNormGradNode,
  DotGeneralNode,
  ArgMinMaxNode,
  FftNode,

  // One-argument ops:
  AbsNode,
  NegNode,
  ExpNode,
  Expm1Node,
  FloorNode,
  CeilNode,
  RoundNode,
  LogNode,
  Log1pNode,
  LogicalNotNode,
  LogisticNode,
  SignNode,
  ClzNode,
  CosNode,
  SinNode,
  TanhNode,
  SqrtNode,
  RsqrtNode,
  ImagNode,
  RealNode,
  ConjNode,

  // Two-arguments ops:
  AddNode,
  MulNode,
  SubNode,
  DivNode,
  // Notice XLA implements Mod, not IEEE754 Remainder operation.
  RemNode,
  AndNode,
  OrNode,
  XorNode,
  DotNode,
  MinNode,
  MaxNode,
  PowNode,
  ComplexNode,

  // Two-arguments comparison ops:
  EqualNode,
  NotEqualNode,
  GreaterOrEqualNode,
  GreaterThanNode,
  LessOrEqualNode,
  LessThanNode,
  EqualTotalOrderNode,
  NotEqualTotalOrderNode,
  GreaterOrEqualTotalOrderNode,
  GreaterThanTotalOrderNode,
  LessOrEqualTotalOrderNode,
  LessThanTotalOrderNode,

  // Nodes with variable sets of arguments.
  RngBitGeneratorNode,
  RngNormalNode,
  RngUniformNode,
};

// SerializedNode represents the Node arguments needed to create an XlaOp. The
// underlying data (pointers) are owned by Go, and shouldn't be freed by C
// functions.
typedef struct {
  int32_t node_type;  // [num_nodes]
  int32_t num_inputs; // [num_nodes]
  XlaOpPtr *inputs;

  // When there is a literal involved.
  struct Literal *literal;

  // Extra arguments that depend on the node type:
  int64_t integer;
  int64_t *integer_array;
  int32_t integer_array_size;
  Shape *shape;
  char *string;
  float float_v;

  // Information about the new op created, filled in by ComputationAddOp. Space
  // allocated in C, but ownership is transferred back to Go.
  XlaOp *new_op;
  Shape *new_shape;

} SerializedNode;

#ifdef __cplusplus
}
#endif

#endif // _GOMLX_XLA_NODE_H
