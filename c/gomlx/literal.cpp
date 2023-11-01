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
#include "gomlx/on_device_buffer.h"
#include "gomlx/shape.h"
#include "gomlx/status.h"

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/types.h"

#include "gomlx/literal.h"

using namespace std;

// Convert xla::Literal object to our own C-wrapped structure. Ownership is
// transferred.
Literal *XlaLiteralToLiteral(xla::Literal *xla_literal) {
  Literal *literal = new Literal();
  literal->literal = xla_literal;
  const xla::Shape &shape = literal->literal->shape();
  literal->is_tuple = shape.IsTuple();
  literal->shape = ShapeFromXlaShape(shape);
  if (literal->shape->tuple_size > 0) {
    literal->is_tuple = true;
  } else {
    literal->data = literal->literal->untyped_data();
    literal->size = literal->literal->element_count();
    literal->size_bytes = literal->literal->size_bytes();
  }
  return literal;
}

// DecomposeLiteral splits literal into its parts and returns a vector of
// *Literal. The original *Literal is invalidated.
Literal **LiteralDecomposeTuple(Literal *literal) {
  int num_elements = literal->shape->tuple_size;
  if (num_elements == 0) {
    return nullptr;
  }
  Literal **results = Malloc<Literal *>(num_elements);
  auto xla_literals = literal->literal->DecomposeTuple();
  for (int ii = 0; ii < num_elements; ii++) {
    Literal *res =
        XlaLiteralToLiteral(new xla::Literal(std::move(xla_literals[ii])));
    results[ii] = res;
  }
  return results;
}

void DeleteLiteral(Literal *literal) {
  literal->data = nullptr; // Owned by the underlying xla::Literal.
  if (literal->literal != nullptr) {
    delete literal->literal;
    literal->literal = nullptr;
  }
  if (literal->shape != nullptr) {
    DeleteShape(literal->shape);
    literal->shape = nullptr;
  }
  delete literal;
}

Literal *MakeLiteralFromShape(Shape *shape) {
  Literal *literal = new Literal();
  xla::Shape xla_shape = MakeXlaShape(shape);
  literal->shape = shape; // Takes ownership of passed shape.
  literal->literal = new xla::Literal(xla_shape);
  if (literal->shape->tuple_size > 0) {
    literal->is_tuple = true;
  } else {
    literal->data = literal->literal->untyped_data();
    literal->size = literal->literal->element_count();
    literal->size_bytes = literal->literal->size_bytes();
  }
  return literal;
}

Literal *MakeLiteralTuple(Literal **elements, int num_elements) {
  std::vector<const xla::Literal *> literals;
  literals.reserve(num_elements);
  for (int ii = 0; ii < num_elements; ii++) {
    literals.push_back(elements[ii]->literal);
  }
  free(elements);
  return XlaLiteralToLiteral(new xla::Literal(xla::LiteralUtil::MakeTuple(literals)));
}

void XlaLiteralRefreshData(Literal *literal) {
  xla::Shape xla_shape = MakeXlaShape(literal->shape);
  literal->data = literal->literal->untyped_data();
  literal->size_bytes = literal->literal->size_bytes();
  literal->size = literal->literal->element_count();
}

StatusOr LiteralToOnDeviceBuffer(Literal *literal, Client *client,
                                 int device_ordinal) {
  StatusOr r{0, 0};
  auto status_or =
      client->client->LiteralToShapedBuffer(*literal->literal, device_ordinal);
  if (!status_or.ok()) {
    r.status = FromStatus(status_or.status());
    return r;
  }

  xla::ScopedShapedBuffer *ssb =
      new xla::ScopedShapedBuffer(std::move(*status_or));
  OnDeviceBuffer *wrapper = new OnDeviceBuffer();
  wrapper->ssb_buffer = ssb;
  r.value = static_cast<void *>(wrapper);
  return r;
}
