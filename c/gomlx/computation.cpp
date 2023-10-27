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

#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "gomlx/client.h"
#include "gomlx/literal.h"
#include "gomlx/on_device_buffer.h"
#include "gomlx/status.h"

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/xla_builder.h"
#include "xla/execution_options_util.h"
#include "xla/literal.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

#include "computation.h"

using namespace std;

// ShapeFromXlaShape allocates and sets a new Shape struct set with the same
// shape defined by xla::Shape. C++ only.
extern Shape *ShapeFromXlaShape(const xla::Shape &shape);

void DeleteComputation(void *comp) {
  Computation *computation = static_cast<Computation *>(comp);
  if (computation->xla_comp != nullptr) {
    delete (computation->xla_comp);
    computation->xla_comp = nullptr;
  }
  delete (computation);
}

Computation *NewComputation(char *name) {
  Computation *comp = new Computation();
  comp->builder = new xla::XlaBuilder(name);
  free(name);
  return comp;
}

void DeleteXlaOp(XlaOp *op) { delete (static_cast<xla::XlaOp *>(op)); }

xla::StatusOr<xla::XlaComputation> xlaCompForReduction(xla::XlaBuilder *builder,
                                                       xla::XlaOp &init_op,
                                                       int32_t node_type) {
  auto shape_or = builder->GetShape(init_op);
  if (!shape_or.ok()) {
    return shape_or.status();
  }
  xla::PrimitiveType primitive_type =
      shape_or.value().element_type(); // They are both ints.

  switch (node_type) {
  case ReduceSumNode:
    return CreateScalarAddComputation(primitive_type, builder);
  case ReduceMaxNode:
    return CreateScalarMaxComputation(primitive_type, builder);
  case ReduceMultiplyNode:
    return CreateScalarMultiplyComputation(primitive_type, builder);
  }
  return xla::Status(
      absl::StatusCode::kInvalidArgument,
      absl::StrFormat("invalid node_type=%d for xlaCompForReduction",
                      node_type));
}

XlaStatus *ComputationAddOp(Computation *comp, SerializedNode *node) {
  xla::XlaBuilder *builder = comp->builder;

  // Create new XlaOp.
  // TODO: A registration mechanism, where one can implement different
  // node_types in different files or libraries even.
  xla::XlaOp op;

  // Extract optional parameters.
  xla::XlaOp **inputs = node->inputs;
  absl::Span<const int64_t> list_of_ints(node->integer_array,
                                         node->integer_array_size);
  absl::Span<const int64_t> shape_dimensions;
  xla::Shape shape;
  if (node->shape != nullptr) {
    shape_dimensions =
        absl::Span<const int64_t>(node->shape->dimensions, node->shape->rank);
    shape = MakeXlaShape(node->shape);
  }

  // Tool to decode contents encoded in the integer array.
  int integer_array_pos = 0;
  auto decode = [&integer_array_pos, node]() {
    return node->integer_array[integer_array_pos++];
  };
  auto decodeSpan = [&integer_array_pos, node](int len) {
    const absl::Span<const int64_t> result(
        node->integer_array + integer_array_pos, len);
    integer_array_pos += len;
    return result;
  };

  // Switch for each node type.
  switch (node->node_type) {

  // Special ops:
  case ConstantNode:
    op = xla::ConstantLiteral(builder, *node->literal->literal);
    break;
  case IotaNode:
    op = xla::Iota(builder, shape, node->integer);
    break;
  case ParameterNode:
    op = xla::Parameter(builder, node->integer, shape, node->string);
    break;
  case ConvertTypeNode:
    op = xla::ConvertElementType(
        *inputs[0], static_cast<xla::PrimitiveType>(node->integer));
    break;
  case WhereNode:
    op = xla::Select(*inputs[0], *inputs[1], *inputs[2]);
    break;
  case TupleNode: {
    std::vector<xla::XlaOp> ops;
    ops.reserve(node->num_inputs);
    for (int ii = 0; ii < node->num_inputs; ii++) {
      ops.push_back(xla::XlaOp(*inputs[ii]));
    }
    op = xla::Tuple(builder, ops);
    break;
  }
  case GetTupleElementNode:
    op = xla::GetTupleElement(*inputs[0], node->integer);
    break;
  case ReshapeNode:
    op = xla::Reshape(*inputs[0], shape_dimensions);
    break;
  case BroadcastNode:
    op = xla::Broadcast(*inputs[0], list_of_ints);
    break;
  case BroadcastInDimNode:
    op = xla::BroadcastInDim(*inputs[0], shape_dimensions, list_of_ints);
    break;
  case ReduceSumNode:
  case ReduceMultiplyNode:
  case ReduceMaxNode: {
    auto reduce_comp_or =
        xlaCompForReduction(builder, *inputs[1], node->node_type);
    if (!reduce_comp_or.ok()) {
      return new xla::Status(reduce_comp_or.status());
    }
    auto reduce_comp = std::move(reduce_comp_or.value());
    if (node->integer_array_size > 0) {
      op = xla::Reduce(*inputs[0], *inputs[1], reduce_comp, list_of_ints);
    } else {
      op = xla::ReduceAll(*inputs[0], *inputs[1], reduce_comp);
    }
    break;
  }
  case ArgMinMaxNode: {
    // Inputs:
    //   * inputs[0]: Tensor to `num_inputs_to_reduce` pairs of input/initial
    //   value.
    //   * node->integer: Axis on which to calculate the argmax/argmin.
    //   * node->integer_array[0]: `is_min`, whether to do argmin or argmax.
    //   * node->integer_array[1]: DType of the output.
    int axis(node->integer);
    bool is_min(node->integer_array[0]);
    xla::PrimitiveType output_type =
        static_cast<xla::PrimitiveType>(node->integer_array[1]);
    op = xla::ArgMinMax(*inputs[0], output_type, axis, is_min);
    break;
  }
  case SliceNode: {
    int rank = node->integer_array_size / 3;
    absl::Span<const int64_t> starts(node->integer_array, rank);
    absl::Span<const int64_t> limits(node->integer_array + rank, rank);
    absl::Span<const int64_t> strides(node->integer_array + 2 * rank, rank);
    op = xla::Slice(*inputs[0], starts, limits, strides);
    break;
  }
  case PadNode: {
    auto &operand = *inputs[0];
    auto &pad_value = *inputs[1];

    xla::PaddingConfig config;
    int rank = node->integer_array_size / 3;
    for (int ii = 0; ii < rank; ii++) {
      auto axisConfig = config.add_dimensions();
      axisConfig->set_edge_padding_low(decode());
      axisConfig->set_edge_padding_high(decode());
      axisConfig->set_interior_padding(decode());
    }

    op = xla::Pad(operand, pad_value, config);
    break;
  }
  case GatherNode: {
    xla::GatherDimensionNumbers gather_dims;
    int64_t index_vector_dim = node->integer_array[0];
    gather_dims.set_index_vector_dim(index_vector_dim);
    int64_t len_offset_dims = node->integer_array[1];
    int64_t len_collapsed_slice_dims = node->integer_array[2];
    int64_t len_start_index_map = node->integer_array[3];
    int64_t len_slice_sizes = node->integer_array[4];
    bool indices_are_sorted = bool(node->integer_array[5]);
    int pos = 6;
    for (int ii = 0; ii < len_offset_dims; ii++) {
      gather_dims.mutable_offset_dims()->Add(node->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_collapsed_slice_dims; ii++) {
      gather_dims.mutable_collapsed_slice_dims()->Add(
          node->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_start_index_map; ii++) {
      gather_dims.mutable_start_index_map()->Add(node->integer_array[pos++]);
    }
    // Same for collapsed_slice_dims and start_index_map
    absl::Span<const int64_t> slice_sizes(node->integer_array + pos,
                                          len_slice_sizes);
    op = xla::Gather(*inputs[0], *inputs[1], gather_dims, slice_sizes,
                     indices_are_sorted);
    break;
  }
  case ScatterNode: {
    int pos = 0;
    xla::ScatterDimensionNumbers scatter_dims;
    scatter_dims.set_index_vector_dim(node->integer_array[pos++]);
    bool unique_indices = bool(node->integer_array[pos++]);
    bool indices_are_sorted = bool(node->integer_array[pos++]);
    int64_t len_update_window_dims = node->integer_array[pos++];
    int64_t len_inserted_window_dims = node->integer_array[pos++];
    int64_t len_scatter_dims_to_operand_dims = node->integer_array[pos++];
    for (int ii = 0; ii < len_update_window_dims; ii++) {
      scatter_dims.mutable_update_window_dims()->Add(
          node->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_inserted_window_dims; ii++) {
      scatter_dims.mutable_inserted_window_dims()->Add(
          node->integer_array[pos++]);
    }
    for (int ii = 0; ii < len_scatter_dims_to_operand_dims; ii++) {
      scatter_dims.mutable_scatter_dims_to_operand_dims()->Add(
          node->integer_array[pos++]);
    }
    // Create the update computation: only Add supported for now.
    auto shape_or = builder->GetShape(*inputs[0]);
    if (!shape_or.ok()) {
      return new xla::Status(std::move(shape_or.status()));
    }
    xla::PrimitiveType primitive_type = shape_or.value().element_type();
    auto update_computation =
        CreateScalarAddComputation(primitive_type, builder);
    op = Scatter(*inputs[0], *inputs[1], *inputs[2], update_computation,
                 scatter_dims, indices_are_sorted, unique_indices);
    break;
  }
  case ConcatenateNode: {
    vector<xla::XlaOp> operands;
    for (int ii = 0; ii < node->num_inputs; ii++) {
      operands.push_back(*inputs[ii]);
    }
    op = xla::ConcatInDim(inputs[0]->builder(), operands, node->integer);
    break;
  }
  case ConvGeneralDilatedNode: {
    int64_t num_spatial_dims = decode();
    int64_t filter_group_count = decode();
    int64_t batch_group_count = decode();

    // Array lengths.
    int64_t len_strides = decode();
    int64_t len_padding = decode();
    int64_t len_input_dilation = decode();
    int64_t len_filter_dilation = decode();

    // Decode ConvolutionDimensionNumbers.
    xla::ConvolutionDimensionNumbers conv_dims;
    conv_dims.set_input_batch_dimension(decode());
    conv_dims.set_input_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_input_spatial_dimensions()->Add(decode());
    }

    conv_dims.set_kernel_input_feature_dimension(decode());
    conv_dims.set_kernel_output_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_kernel_spatial_dimensions()->Add(decode());
    }

    conv_dims.set_output_batch_dimension(decode());
    conv_dims.set_output_feature_dimension(decode());
    for (int ii = 0; ii < num_spatial_dims; ii++) {
      conv_dims.mutable_output_spatial_dimensions()->Add(decode());
    }

    // Unpack various arrays.
    absl::Span<const int64_t> window_strides = decodeSpan(len_strides);
    std::vector<std::pair<int64_t, int64_t>> padding(len_padding);
    for (int ii = 0; ii < len_padding; ii++) {
      padding[ii].first = decode();
      padding[ii].second = decode();
    }
    absl::Span<const int64_t> input_dilation = decodeSpan(len_input_dilation);
    absl::Span<const int64_t> filter_dilation = decodeSpan(len_filter_dilation);

    // Other undocumented parameters not used.
    const xla::PrecisionConfig *precision_config = nullptr;
    std::optional<xla::PrimitiveType> preferred_element_type;

    std::optional<std::vector<bool>> window_reversal = std::nullopt;
    op = ConvGeneralDilated(
        *inputs[0], *inputs[1], window_strides,
        /* absl::Span<const std::pair<int64_t, int64_t>> */ padding,
        input_dilation, filter_dilation, conv_dims, filter_group_count,
        batch_group_count, precision_config, preferred_element_type,
        window_reversal);
    break;
  }
  case ReverseNode: {
    op = Rev(*inputs[0], absl::Span<const int64_t>(node->integer_array,
                                                   node->integer_array_size));
    break;
  }
  case TransposeNode: {
    op = Transpose(*inputs[0],
                   absl::Span<const int64_t>(node->integer_array,
                                             node->integer_array_size));
    break;
  }
  case ReduceWindowNode: {
    // Create reduction comp.
    int64_t reduction_type = node->integer;
    auto reduce_comp_or =
        xlaCompForReduction(builder, *inputs[1], reduction_type);
    if (!reduce_comp_or.ok()) {
      return new xla::Status(reduce_comp_or.status());
    }
    auto reduce_comp = std::move(reduce_comp_or.value());

    // Decode parameters.
    int64_t rank = decode();
    int64_t len_base_dilations = decode();
    int64_t len_window_dilations = decode();
    int64_t len_paddings = decode();
    absl::Span<const int64_t> window_dimensions = decodeSpan(rank);
    absl::Span<const int64_t> window_strides = decodeSpan(rank);
    absl::Span<const int64_t> base_dilations = decodeSpan(len_base_dilations);
    absl::Span<const int64_t> window_dilations =
        decodeSpan(len_window_dilations);
    std::vector<std::pair<int64_t, int64_t>> paddings(len_paddings);
    for (int ii = 0; ii < len_paddings; ii++) {
      paddings[ii].first = decode();
      paddings[ii].second = decode();
    }

    op = xla::ReduceWindowWithGeneralPadding(
        *inputs[0], *inputs[1], reduce_comp, window_dimensions, window_strides,
        base_dilations, window_dilations, paddings);
    break;
  }
  case SelectAndScatterNode: {
    // All operands.
    auto &operand = *inputs[0];
    auto &source = *inputs[1];
    auto &init_value = *inputs[2];

    // Create select and scatter comps.
    auto shape_or = builder->GetShape(init_value);
    if (!shape_or.ok()) {
      return new xla::Status(shape_or.status());
    }
    xla::PrimitiveType primitive_type =
        shape_or.value().element_type(); // They are both ints.
    xla::XlaComputation scatter_comp(
        CreateScalarAddComputation(primitive_type, builder));
    xla::XlaComputation select_comp(
        CreateScalarGeComputation(primitive_type, builder));

    // Decode parameters.
    int64_t rank = decode();
    int64_t len_paddings = decode();
    absl::Span<const int64_t> window_dimensions = decodeSpan(rank);
    absl::Span<const int64_t> window_strides = decodeSpan(rank);
    std::vector<std::pair<int64_t, int64_t>> paddings(len_paddings);
    for (int ii = 0; ii < len_paddings; ii++) {
      paddings[ii].first = decode();
      paddings[ii].second = decode();
    }

    op = SelectAndScatterWithGeneralPadding(
        operand, select_comp, window_dimensions, window_strides, paddings,
        source, init_value, scatter_comp);
    break;
  }
  case BatchNormInferenceNode: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &offset = *inputs[2];
    auto &mean = *inputs[3];
    auto &variance = *inputs[4];
    float epsilon = node->float_v;
    int64_t feature_index = node->integer;
    op = xla::BatchNormInference(operand, scale, offset, mean, variance,
                                 epsilon, feature_index);
    break;
  }
  case BatchNormTrainingNode: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &offset = *inputs[2];
    float epsilon = node->float_v;
    int64_t feature_index = node->integer;
    op = xla::BatchNormTraining(operand, scale, offset, epsilon, feature_index);
    break;
  }
  case BatchNormGradNode: {
    auto &operand = *inputs[0];
    auto &scale = *inputs[1];
    auto &batch_mean = *inputs[2];
    auto &batch_var = *inputs[3];
    auto &grad_output = *inputs[4];
    float epsilon = node->float_v;
    int64_t feature_index = node->integer;
    op = xla::BatchNormGrad(operand, scale, batch_mean, batch_var, grad_output,
                            epsilon, feature_index);
    break;
  }
  case DotGeneralNode: {
    auto &lhs = *inputs[0]; // left-hand-side.
    auto &rhs = *inputs[1];
    xla::DotDimensionNumbers dims;
    std::vector<google::protobuf::RepeatedField<google::protobuf::int64> *>
        lists = {dims.mutable_lhs_contracting_dimensions(),
                 dims.mutable_lhs_batch_dimensions(),
                 dims.mutable_rhs_contracting_dimensions(),
                 dims.mutable_rhs_batch_dimensions()};
    std::vector<int> listsLens;
    for (int ii = 0; ii < lists.size(); ii++) {
      listsLens.push_back(decode());
    }
    for (int ii = 0; ii < lists.size(); ii++) {
      auto &list = lists[ii];
      int len = listsLens[ii];
      for (int elem = 0; elem < len; elem++) {
        list->Add(decode());
      }
    }

    const xla::PrecisionConfig *precision_config = nullptr;
    std::optional<xla::PrimitiveType> preferred_element_type;
    op = xla::DotGeneral(lhs, rhs, dims, precision_config,
                         preferred_element_type);
    break;
  }

  // One-argument ops:
  case AbsNode:
    op = xla::Abs(*inputs[0]);
    break;
  case NegNode:
    op = xla::Neg(*inputs[0]);
    break;
  case ExpNode:
    op = xla::Exp(*inputs[0]);
    break;
  case Expm1Node:
    op = xla::Expm1(*inputs[0]);
    break;
  case FloorNode:
    op = xla::Floor(*inputs[0]);
    break;
  case CeilNode:
    op = xla::Ceil(*inputs[0]);
    break;
  case RoundNode:
    op = xla::Round(*inputs[0]);
    break;
  case LogNode:
    op = xla::Log(*inputs[0]);
    break;
  case Log1pNode:
    op = xla::Log1p(*inputs[0]);
    break;
  case LogicalNotNode:
    op = xla::Not(*inputs[0]);
    break;
  case LogisticNode:
    op = xla::Logistic(*inputs[0]);
    break;
  case SignNode:
    op = xla::Sign(*inputs[0]);
    break;
  case ClzNode:
    op = xla::Clz(*inputs[0]);
    break;
  case CosNode:
    op = xla::Cos(*inputs[0]);
    break;
  case SinNode:
    op = xla::Sin(*inputs[0]);
    break;
  case TanhNode:
    op = xla::Tanh(*inputs[0]);
    break;
  case SqrtNode:
    op = xla::Sqrt(*inputs[0]);
    break;
  case RsqrtNode:
    op = xla::Rsqrt(*inputs[0]);
    break;
  case ImagNode:
    op = xla::Imag(*inputs[0]);
    break;
  case RealNode:
    op = xla::Real(*inputs[0]);
    break;
  case ConjNode:
    op = xla::Conj(*inputs[0]);
    break;

  // Two-arguments ops
  case AddNode:
    op = xla::Add(*inputs[0], *inputs[1]);
    break;
  case MulNode:
    op = xla::Mul(*inputs[0], *inputs[1]);
    break;
  case SubNode:
    op = xla::Sub(*inputs[0], *inputs[1]);
    break;
  case DivNode:
    op = xla::Div(*inputs[0], *inputs[1]);
    break;
  case RemNode:
    op = xla::Rem(*inputs[0], *inputs[1]);
    break;
  case AndNode:
    op = xla::And(*inputs[0], *inputs[1]);
    break;
  case OrNode:
    op = xla::Or(*inputs[0], *inputs[1]);
    break;
  case XorNode:
    op = xla::Xor(*inputs[0], *inputs[1]);
    break;
  case DotNode:
    op = xla::Dot(*inputs[0], *inputs[1]);
    break;
  case MinNode:
    op = xla::Min(*inputs[0], *inputs[1]);
    break;
  case MaxNode:
    op = xla::Max(*inputs[0], *inputs[1]);
    break;
  case PowNode:
    op = xla::Pow(*inputs[0], *inputs[1]);
    break;
  case ComplexNode:
    op = xla::Complex(*inputs[0], *inputs[1]);
    break;

  // Nodes with variable sets of arguments.
  case RngNormalNode:
    op = xla::RngNormal(*inputs[0], *inputs[1], shape);
    break;
  case RngUniformNode:
    op = xla::RngUniform(*inputs[0], *inputs[1], shape);
    break;
  case RngBitGeneratorNode: {
    xla::RandomAlgorithm algo =
        static_cast<xla::RandomAlgorithm>(node->integer);
    op = xla::RngBitGenerator(algo, *inputs[0], shape);
    break;
  }
  case FftNode: {
    xla::FftType fft_type = static_cast<xla::FftType>(node->integer);
    op = xla::Fft(*inputs[0], fft_type, list_of_ints);
    break;
  }

  case EqualNode:
    op = xla::Eq(*inputs[0], *inputs[1]);
    break;
  case NotEqualNode:
    op = xla::Ne(*inputs[0], *inputs[1]);
    break;
  case GreaterOrEqualNode:
    op = xla::Ge(*inputs[0], *inputs[1]);
    break;
  case GreaterThanNode:
    op = xla::Gt(*inputs[0], *inputs[1]);
    break;
  case LessOrEqualNode:
    op = xla::Le(*inputs[0], *inputs[1]);
    break;
  case LessThanNode:
    op = xla::Lt(*inputs[0], *inputs[1]);
    break;
  case EqualTotalOrderNode:
    op = xla::EqTotalOrder(*inputs[0], *inputs[1]);
    break;
  case NotEqualTotalOrderNode:
    op = xla::NeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case GreaterOrEqualTotalOrderNode:
    op = xla::GeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case GreaterThanTotalOrderNode:
    op = xla::GtTotalOrder(*inputs[0], *inputs[1]);
    break;
  case LessOrEqualTotalOrderNode:
    op = xla::LeTotalOrder(*inputs[0], *inputs[1]);
    break;
  case LessThanTotalOrderNode:
    op = xla::LtTotalOrder(*inputs[0], *inputs[1]);
    break;

  default:
    return new xla::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("invalid node_type=%d for ComputationAddOp",
                        node->node_type));
  }
  if (!op.valid()) {
    auto status = comp->builder->first_error();
    if (!status.ok()) {
      return new xla::Status(status);
    }
    return new xla::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("failed to convert node to XLA: node_type=%d",
                        node->node_type));
  }
  node->new_op = new xla::XlaOp(op);

  // Also retrieve of the shape of the resulting op.
  auto shape_or = builder->GetShapePtr(*node->new_op);
  if (!shape_or.ok()) {
    return FromStatus(shape_or.status());
  }
  node->new_shape = ShapeFromXlaShape(*shape_or.value());
  return nullptr;
}

XlaStatus *ClientCompileComputation(Client *client, Computation *comp,
                                    int num_params, Shape **param_shapes,
                                    XlaOp *output) {
  // Build XlaComputation.
  auto comp_or = comp->builder->Build(*output);
  if (!comp_or.ok()) {
    return FromStatus(comp_or.status());
  }
  comp->xla_comp = new xla::XlaComputation(std::move(comp_or.value()));

  // Compile it.
  std::vector<xla::Shape> shapes(num_params);
  for (int ii = 0; ii < num_params; ii++) {
    shapes[ii] = MakeXlaShape(param_shapes[ii]);
  }

  // Compile with LocalClient
  xla::ExecutableBuildOptions options;
  std::vector<xla::Shape *> shape_pointers(num_params);
  for (int ii = 0; ii < num_params; ii++) {
    shape_pointers[ii] = &shapes[ii];
  }
  auto status_or =
      client->client->Compile(*comp->xla_comp, shape_pointers, options);
  if (!status_or.ok()) {
    return FromStatus(status_or.status());
  }
  auto local_execs = std::move(status_or.value());
  if (local_execs.size() > 1) {
    return new xla::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("compilation for multiple partitions not allowed, got "
                        "%d partitions, wanted only one",
                        int(local_execs.size())));
  }
  comp->local_exec = std::move(local_execs[0]);
  return nullptr;
}

StatusOr ClientExecuteComputation(Client *client, Computation *comp,
                                  int num_params, XlaShapedBuffer **params) {
  StatusOr r{0, 0};
  absl::Span<xla::ShapedBuffer *const> arguments(params, num_params);
  auto data_or = comp->local_exec->Run(arguments, client->exec_options);
  if (!data_or.ok()) {
    r.status = FromStatus(data_or.status());
    return r;
  }
  xla::ScopedShapedBuffer *ssb =
      new xla::ScopedShapedBuffer(std::move(*data_or));
  OnDeviceBuffer *wrapper = new OnDeviceBuffer();
  wrapper->ssb_buffer = ssb;
  r.value = static_cast<void *>(wrapper);
  return r;
}

void DeleteGlobalData(XlaGlobalData *gd) {
  delete static_cast<xla::GlobalData *>(gd);
}

StatusOr GlobalDataShape(XlaGlobalData *gd, Client *client) {
  StatusOr r{0, 0};
  auto shape_or = client->client->GetShape(*gd);
  if (!shape_or.ok()) {
    r.status = FromStatus(shape_or.status());
    return r;
  }
  auto shape = shape_or.value();
  r.value = static_cast<void *>(ShapeFromXlaShape(shape));
  return r;
}

StatusOr GlobalDataDeconstructTuple(XlaGlobalData *gd, Client *client) {
  StatusOr r{0, 0};
  auto gds_or = client->client->DeconstructTuple(*gd);
  if (!gds_or.ok()) {
    r.status = FromStatus(gds_or.status());
    return r;
  }
  std::vector<std::unique_ptr<xla::GlobalData>> gds = std::move(gds_or.value());
  // Since this data is freed by C.free in Go, we use malloc to allocate it.
  XlaGlobalData **gdsArray =
      (XlaGlobalData **)malloc(sizeof(XlaGlobalData *) * gds.size());
  for (int ii = 0; ii < gds.size(); ii++) {
    gdsArray[ii] = gds[ii].release();
  }
  r.value = static_cast<void *>(gdsArray);
  return r;
}

StatusOr TransferFromServer(XlaGlobalData *gd, Client *client) {
  StatusOr r{0, 0};
  auto literal_or = client->client->Transfer(*gd, nullptr);
  if (!literal_or.ok()) {
    r.status = FromStatus(literal_or.status());
    return r;
  }
  Literal *res =
      XlaLiteralToLiteral(new xla::Literal(std::move(literal_or.value())));
  r.value = static_cast<void *>(res);
  return r;
}

StatusOr TransferToServer(Literal *literal, Client *client) {
  StatusOr r{0, 0};
  auto gd_or = client->client->TransferToServer(*literal->literal, nullptr);
  if (!gd_or.ok()) {
    r.status = FromStatus(gd_or.status());
    return r;
  }
  xla::GlobalData *res = gd_or.value().release();
  r.value = static_cast<void *>(res);
  return r;
}
