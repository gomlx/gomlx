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

// literal.h holds C API structure to literal values: this header and
// implementation are included in the thiner gomlx_aot library.
#ifndef _GOMLX_XLA_LITERAL_H
#define _GOMLX_XLA_LITERAL_H

#include "gomlx/client.h"
#include "gomlx/status.h"

#ifdef __cplusplus
// C++ only includes: these are not seen by the Go compiler.
#include "xla/client/xla_builder.h"
#include "xla/shape.h"

typedef xla::Literal XlaLiteral;
extern Literal *XlaLiteralToLiteral(xla::Literal *xla_literal);

#else
typedef _Bool bool;
typedef void XlaLiteral;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Literal points to the data in C++ space.
// TODO: add support for layouts. For now assuming major->minor layout, that is
// row-major for 2D tensors.
//
// Memory managed by C++ new/delete.
typedef struct Literal {
  // Pointer to xla::Literal that holds the data.
  XlaLiteral *literal;

  // Tuple objects are a collection of several literals, and don't have data
  // themselves.
  bool is_tuple;

  // Shape information.
  Shape *shape;

  // Consecutive data: array of some value defined by dtype
  // (xla::PrimitiveType).
  void *data;

  // Size of the data (total number of elements) and in bytes.
  int64_t size, size_bytes;
} Literal;

// Delete the given structure, and the xla::Literal held by it.
extern void DeleteLiteral(Literal *literal);

// MakeFromShape create a new literal with the given shape, with uninitialized
// data. It takes ownership of the given `Shape*` pointer, and stores it within
// the returned Literal struct.
extern Literal *MakeLiteralFromShape(Shape *shape);

// Update data, size and size_bytes elements.
extern void XlaLiteralRefreshData(Literal *literal);

// MakeXlaLiteralTuple combine the `elements` into an xla::Literal, and returns
// it converted to a newly allocated Literal structure.
//
// It takes ownership of the literals pointed by elements, and the `elements`
// array itself is freed.
extern Literal *MakeLiteralTuple(Literal **elements, int num_elements);

// TransferToServer takes data from a local Literal to the accelerator server
// used by client. Returns a pointer to a xla::GlobalData on success or the
// status on error.
extern StatusOr TransferToServer(Literal *literal, Client *client);

// LiteralDecomposeTuple splits literal into its parts and returns a vector of
// *Literal. The original *Literal is invalidated, but not deleted (still owned
// by caller). The returned array is owned and should be free by the caller.
extern Literal **LiteralDecomposeTuple(Literal *literal);

// LiteralToOnDeviceBuffer conversion. Either returns a OnDeviceBuffer* or an
// error.
extern StatusOr LiteralToOnDeviceBuffer(Literal *literal, Client *client,
                                        int device_ordinal);

struct OnDeviceBuffer;
typedef struct OnDeviceBuffer OnDeviceBuffer;

// ShapedBufferToLiteral conversion. Either returns a Literal* or an error.
StatusOr OnDeviceBufferToLiteral(OnDeviceBuffer *buffer, Client *client);

#ifdef __cplusplus
}
#endif

#endif
