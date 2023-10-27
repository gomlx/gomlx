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
// #include "xla/client/client.h"
// #include "xla/client/client_library.h"
// #include "xla/client/lib/arithmetic.h"
// #include "xla/client/xla_builder.h"
// #include "xla/execution_options_util.h"
#include "xla/array.h"
#include "xla/literal.h"
// #include "xla/service/platform_util.h"
// #include "xla/service/shaped_buffer.h"
#include "xla/statusor.h"
#include "xla/types.h"
// #include "xla/xla_data.pb.h"
#include "xla/status.h"

#include "gomlx/shape.h"

using namespace std;

// ShapeFromXlaShape allocates and sets a new Shape struct set with the same
// shape defined by xla::Shape. C++ only.
extern Shape *ShapeFromXlaShape(const xla::Shape &shape);

// ShapeFromXlaShape returns a newly allocated Shape C-struct.
Shape *ShapeFromXlaShape(const xla::Shape &xla_shape) {
  Shape *shape = Malloc<Shape>();
  shape->dtype = int32_t(xla_shape.element_type());
  if (shape->dtype == xla::TUPLE) {
    shape->tuple_size = xla_shape.tuple_shapes_size();
    if (shape->tuple_size > 0) {
      shape->tuple_shapes = Malloc<Shape *>(shape->tuple_size);
      for (int ii = 0; ii < shape->tuple_size; ii++) {
        shape->tuple_shapes[ii] = ShapeFromXlaShape(xla_shape.tuple_shapes(ii));
      }
    }
    return shape;
  }
  if (xla_shape.IsArray()) {
    shape->rank = xla_shape.rank();
    if (shape->rank > 0) {
      shape->dimensions = Malloc<int64_t>(shape->rank);
      const auto xla_shape_dims = xla_shape.dimensions();
      std::copy(xla_shape_dims.begin(), xla_shape_dims.end(),
                shape->dimensions);
    }
  }
  return shape;
}

void DeleteShape(Shape *shape) {
  if (shape == nullptr) {
    return;
  }
  if (shape->dimensions != nullptr) {
    free(shape->dimensions);
    shape->dimensions = 0;
  }
  if (shape->tuple_size > 0 && shape->tuple_shapes != nullptr) {
    for (int ii = 0; ii < shape->tuple_size; ii++) {
      DeleteShape(shape->tuple_shapes[ii]);
      shape->tuple_shapes[ii] = nullptr;
    }
    free(shape->tuple_shapes);
    shape->tuple_shapes = nullptr;
  }
  free(shape);
}

xla::Shape MakeXlaShape(Shape *shape) {
  xla::Shape xla_shape;
  auto primitive_type = static_cast<xla::PrimitiveType>(shape->dtype);

  if (shape->tuple_size > 0) {
    // Create a tuple shape.
    if (shape->tuple_shapes == nullptr) {
      // Shape of elements of the tuple not provided, fail.
      // TODO: Log error.
      return xla_shape;
    }
    std::vector<xla::Shape> tuple_shapes;
    tuple_shapes.reserve(shape->tuple_size);
    for (int ii = 0; ii < shape->tuple_size; ii++) {
      tuple_shapes.push_back(MakeXlaShape(shape->tuple_shapes[ii]));
    }
    return xla::ShapeUtil::MakeTupleShape(tuple_shapes);
  }
  const auto rank = shape->rank;
  if (rank == 0) {
    return xla::ShapeUtil::MakeScalarShape(primitive_type);
  } else {
    absl::Span<const int64_t> dimensions(shape->dimensions, rank);
    return xla::ShapeUtil::MakeShape(primitive_type, dimensions);
  }
  return xla_shape;
}
