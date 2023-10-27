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

/*
shaped_buffer.h provide a C API around xla::ShapedBuffer, defined in
`tensorflow/compiler/xla/service/shaped_buffer.h`.

Notice the Go API will only see the "C" view of this file (where `__cplusplus`
is not defined). The "C" view should have no includes from tensorflow, so that
the library won't depend on their
*/
#ifndef _GOMLX_ON_DEVICE_BUFFER_H
#define _GOMLX_ON_DEVICE_BUFFER_H

#include "gomlx/shape.h"
#include "gomlx/status.h"

#ifdef __cplusplus
// C++ only includes: these are not seen by the Go compiler.
#include "xla/service/shaped_buffer.h"

typedef xla::ShapedBuffer XlaShapedBuffer;
typedef xla::ScopedShapedBuffer XlaScopedShapedBuffer;

#else
typedef _Bool bool;
typedef void XlaShapedBuffer;
typedef void XlaScopedShapedBuffer;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// OnDeviceBuffer is a C struct wrapper around a xla::ShapedBuffer or a
// xla::ScopeShapedBuffer. The two are used in different times in XLA -- it's
// not documented :( ... Only one of the two are set.
//
// Memory managed by C++ new/delete.
struct OnDeviceBuffer {
  XlaShapedBuffer *sb_buffer;
  XlaScopedShapedBuffer *ssb_buffer;
};
typedef struct OnDeviceBuffer OnDeviceBuffer;

// Deletes ShapedBuffer and all associated resources.
extern void DeleteOnDeviceBuffer(OnDeviceBuffer *b);

// Returns the shape of the on-device. Ownership of Shape is transferred.
extern Shape *OnDeviceBufferShape(OnDeviceBuffer *b);

// OnDeviceBufferSubTree retrieves an element from a nested tuple (tree)
// OnDeviceBuffer. The subtree in the original buffer is converted to null.
extern StatusOr OnDeviceBufferSubTree(OnDeviceBuffer *b, int path_length,
                                      int64_t *path);

// Device number of the buffer.
extern int OnDeviceBufferDeviceOrdinal(OnDeviceBuffer *b);

// Convert to a friendly string. Ownership of char* is transferred.
extern char *OnDeviceBufferToString(OnDeviceBuffer *b);

#ifdef __cplusplus
}
#endif

#endif