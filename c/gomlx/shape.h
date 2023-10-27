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
#ifndef _GOMLX_XLA_SHAPE_H
#define _GOMLX_XLA_SHAPE_H

#include <stdlib.h>

#ifdef __cplusplus
// C++ only includes: these are not seen by the Go compiler.
#include "status.h"
#include "xla/shape.h"

#else
// C and CGO only code.
typedef _Bool bool;
typedef void XlaLiteral;
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct Literal;

// Shape C representation.
//
// Memory managed by malloc/free.
struct Shape {
  // Data type.
  int32_t dtype;

  // Tuple-Size, if tuple
  int32_t tuple_size;

  // Number of dimensions.
  int64_t rank;

  // List of dimensions.
  int64_t *dimensions;

  // List of the tuple elements shapes. An array of tuple_size pointers.
  struct Shape **tuple_shapes;
};
typedef struct Shape Shape;

// Delete the given Shape -- it actually uses C's free.
extern void DeleteShape(Shape *shape);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Functionality available only for C++ code:

// MakeXlaShape converts from our C Shape representation to an xla::Shape.
// The `shape` given is not freed.
xla::Shape MakeXlaShape(Shape *shape);

// ShapeFromXlaShape returns a newly allocated Shape C-struct.
// Ownership is returned to the caller.
Shape *ShapeFromXlaShape(const xla::Shape &xla_shape);

#endif

#endif // _GOMLX_XLA_SHAPE_H
