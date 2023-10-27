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
#include <string>
#include <vector>

#include "gomlx/on_device_buffer.h"

#include "absl/status/status.h"
#include "gomlx/literal.h"
#include "gomlx/status.h"

using namespace std;

xla::ShapedBuffer *get_shaped_buffer(OnDeviceBuffer *b) {
  if (b == nullptr) {
    return nullptr;
  }
  if (b->sb_buffer != nullptr) {
    return b->sb_buffer;
  } else {
    return b->ssb_buffer;
  }
}

// Deletes ShapedBuffer and all associated resources.
void DeleteOnDeviceBuffer(OnDeviceBuffer *b) {
  if (b == nullptr) {
    return;
  }
  if (b->sb_buffer != nullptr) {
    delete b->sb_buffer;
    b->sb_buffer = nullptr;
  }
  if (b->ssb_buffer != nullptr) {
    delete b->ssb_buffer;
    b->ssb_buffer = nullptr;
  }
  delete b;
}

// Returns the shape of the on-host representation of the data held by this
// ShapedBuffer. Ownership of Shape is transferred.
Shape *OnDeviceBufferShape(OnDeviceBuffer *b) {
  auto sb = get_shaped_buffer(b);
  if (sb == nullptr) {
    return nullptr;
  }
  xla::Shape shape = sb->on_device_shape();
  return ShapeFromXlaShape(shape);
}

// Device number of the buffer.
int OnDeviceBufferDeviceOrdinal(OnDeviceBuffer *b) {
  auto sb = get_shaped_buffer(b);
  if (sb == nullptr) {
    return -1;
  }
  return sb->device_ordinal();
}

// Convert to a friendly string. Ownership of char* is transferred.
char *OnDeviceBufferToString(OnDeviceBuffer *b) {
  auto sb = get_shaped_buffer(b);
  if (sb == nullptr) {
    return nullptr;
  }
  return c_str(sb->ToString());
}

StatusOr OnDeviceBufferSubTree(OnDeviceBuffer *b, int path_length,
                               int64_t *path) {
  StatusOr r{0, 0};
  xla::ShapeIndexView indices_view(path, path_length);
  xla::ShapeIndex indices(indices_view);

  if (b == nullptr) {
    r.status =
        new xla::Status(absl::StatusCode::kInvalidArgument,
                        "cant take sub-tree of OnDeviceBuffer == nullptr");

  } else if (b->sb_buffer != nullptr) {
    auto status_or = b->sb_buffer->SubShapedBuffer(indices);
    if (!status_or.ok()) {
      r.status = FromStatus(status_or.status());
      return r;
    }
    xla::ShapedBuffer *sb = new xla::ShapedBuffer(std::move(*status_or));
    OnDeviceBuffer *wrapper = new OnDeviceBuffer();
    wrapper->sb_buffer = sb;
    r.value = static_cast<void *>(wrapper);

  } else if (b->ssb_buffer != nullptr) {
    xla::ScopedShapedBuffer *ssb = new xla::ScopedShapedBuffer(
        std::move(b->ssb_buffer->TakeSubTree(indices)));
    OnDeviceBuffer *wrapper = new OnDeviceBuffer();
    wrapper->ssb_buffer = ssb;
    r.value = static_cast<void *>(wrapper);

  } else {
    r.status = new xla::Status(absl::StatusCode::kInvalidArgument,
                               "cant take sub-tree of empty OnDeviceBuffer");
  }
  return r;
}

StatusOr OnDeviceBufferToLiteral(OnDeviceBuffer *buffer, Client *client) {
  auto sb = get_shaped_buffer(buffer);
  StatusOr r{0, 0};

  if (sb == nullptr) {
    r.status =
        new xla::Status(absl::StatusCode::kInvalidArgument,
                        "cant convert nullptr OnDeviceBuffer to Literal");

  } else {
    auto status_or = client->client->ShapedBufferToLiteral(*sb);
    if (!status_or.ok()) {
      r.status = FromStatus(status_or.status());
      return r;
    }
    Literal *res =
        XlaLiteralToLiteral(new xla::Literal(std::move(status_or.value())));
    r.value = static_cast<void *>(res);
  }
  return r;
}
