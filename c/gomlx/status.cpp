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

#include <string.h>

#include "third_party/tensorflow/compiler/xla/status.h"
#include "third_party/tcmalloc/malloc_extension.h"
#include "third_party/tensorflow/compiler/xla/statusor.h"

#include "third_party/golang/github_com/gomlx/gomlx/v/v0/c/gomlx/shape.h"
#include "third_party/golang/github_com/gomlx/gomlx/v/v0/c/gomlx/status.h"

const char *TF_LOG_LEVEL_ENV = "TF_CPP_MIN_LOG_LEVEL";

// Set TF_CPP_MIN_LOG_LEVEL to two by default. Notice that if it is set, it is
// not overwritten.
int initSetTfLogs() { return setenv(TF_LOG_LEVEL_ENV, "2", /* overwrite */ 0); }

extern int initCall;
int initCall = initSetTfLogs();

int xla_wrapper_version() { return XlaWrapperVersion; }

// Cast C pointer type to C++ object pointer.
absl::Status *XlaStatusCast(XlaStatus *s) {
  return static_cast<absl::Status *>(s);
}

char *c_str(const std::string &s) { return strdup(s.c_str()); }

VectorData *str_to_bytes(const std::string &s) {
  VectorData *v = Malloc<VectorData>();
  v->count = s.size();
  void *data = malloc(v->count);
  memcpy(data, s.data(), v->count);
  v->data = data;
  return v;
}

VectorPointers *c_vector_str(const std::vector<std::string> &v) {
  VectorPointers *vp = Malloc<VectorPointers>();
  vp->count = v.size();
  if (vp->count > 0) {
    vp->data = Malloc<void *>(vp->count);
    for (int ii = 0; ii < vp->count; ii++) {
      vp->data[ii] = static_cast<void *>(c_str(v[ii]));
    }
  }
  return vp;
}

char *memory_stats() {
#if defined(TCMALLOC)
    std::string src = tcmalloc::MallocExtension::GetStats();
    char *dest = new char[src.length() + 1];
    strcpy(dest, src.c_str());
    return dest;
#else
  const size_t kBufferSize = 10 * 1024 * 1024;
  char *buf = (char *)malloc(kBufferSize);
  MallocExtension::instance()->GetStats(buf, kBufferSize);
  return buf;
#endif
}

size_t memory_usage() {
  const char *kCurrentAllocatedBytes = "generic.current_allocated_bytes";
#if defined(TCMALLOC)
    absl::optional<size_t> current_allocated = tcmalloc::MallocExtension::GetNumericProperty(kCurrentAllocatedBytes);
    if (current_allocated.has_value()) {
      return *current_allocated;
    }
#else
  size_t res;
  if (MallocExtension::instance()->GetNumericProperty(kCurrentAllocatedBytes,
                                                      &res)) {
    return res;
  }
#endif
  return 0;
}

bool heap_checker_no_global_leaks() {
#if defined(TCMALLOC)
    return false;
#else
    return HeapLeakChecker::NoGlobalLeaks();
#endif
}

char *number_to_string(int n) { return c_str(std::to_string(n)); }

bool XlaStatusOk(XlaStatus *status) { return XlaStatusCast(status)->ok(); }
char *XlaStatusErrorMessage(XlaStatus *status) {
  return c_str(XlaStatusCast(status)->message().data());
}

int XlaStatusCode(XlaStatus *status) {
  return int(XlaStatusCast(status)->code());
}

XlaStatus *FromStatus(const absl::Status &status) {
  return static_cast<XlaStatus *>(new absl::Status(status));
}

void DeleteXlaStatus(XlaStatus *xla_status) {
  delete XlaStatusCast(xla_status);
}
