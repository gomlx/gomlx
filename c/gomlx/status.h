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

// status.h holds C/Go conversion utilities, in particular handling of
// xla::Status and xla::StatusOr.
#ifndef _GOMLX_XLA_STATUS_H
#define _GOMLX_XLA_STATUS_H
// status.h holds the simplified C interface to xla::Status and xla::StatusOr
// objects.
#include <stdlib.h>

#ifndef __cplusplus
// Definition only needed in C
typedef _Bool bool;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// XlaStatus behind the scenes is a xla::Status type.
typedef void XlaStatus;

// StatusOr contains status or the value from the C++ StatusOr.
//
// Memory managed by malloc/free.
typedef struct {
  XlaStatus *status; // This is a C++ object, managed by new/delete.
  void *value;
} StatusOr;

// VectorPointers is a generic container for a dynamically allocated vector.
// Usually it has to be deleted after use.
//
// Memory managed by malloc/free.
typedef struct {
  int count;
  void **data; // Will contain <count> elements.
} VectorPointers;

// VectorData is a generic container for a vector of some arbitrary data type.
//
// Memory managed by malloc/free.
typedef struct {
  int count;
  void *data; // Will contain <count> elements.
} VectorData;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
///////////////////////////////////////////////////////////////////////////////////////////
// C++ tools used by the implementations.
///////////////////////////////////////////////////////////////////////////////////////////
#include <memory>
#include <string>
#include <vector>

#include "xla/status.h"
#include "xla/statusor.h"
// #include "xla/xla/shape_util.h"

// Malloc is a convenience allocation of individual items or arrays (for n>1)
// using C malloc, needed to interface in Go.
//
// n is the number of elements allocated, and defaults to 1.
template <typename T> T *Malloc(int n = 1) {
  if (n <= 0) {
    return nullptr;
  }
  size_t bytes = sizeof(T) * n;
  T *data = (T *)malloc(bytes);
  memset(data, 0, bytes);
  return data;
}

// C++ version of the XLA wrapper library.
// Bump the number whenever something changes, and keep in sync with Go file
// xla/node.go.
constexpr int XlaWrapperVersion = 14;

// Converts std::string to an allocated `char *` allocated with malloc that in
// Go can be passed to StrFree. This doesn't work for binary blobs, like
// serialized protos.
extern char *c_str(const std::string &s);

// Converts std::string containing binary data to an allocated VectorData
// object, that points to a newly allocated array of bytes.
extern VectorData *str_to_bytes(const std::string &s);

// Converts a vector<string> to an allocated `VectorPointers *` that in Go can
// be passed to StrVectorFree.
extern VectorPointers *c_vector_str(const std::vector<std::string> &v);

// FromStatus creates a dynamically allocated status (aliased to *XlaStatus)
// from the given one -- contents are transferred.
XlaStatus *FromStatus(const xla::Status &status);

template <typename T>
StatusOr FromStatusOr(xla::StatusOr<std::unique_ptr<T>> &statusor) {
  StatusOr r;
  r.status =
      static_cast<XlaStatus *>(new xla::Status(std::move(statusor.status())));
  if (statusor.ok()) {
    r.value = static_cast<void *>(statusor->get());
    statusor->release(); // Ownership should go to StatusOr.
  }
  return r;
}

template <typename T> StatusOr FromStatusOr(xla::StatusOr<T *> &status_or) {
  StatusOr r;
  r.status =
      static_cast<XlaStatus *>(new xla::Status(std::move(status_or.status())));
  if (status_or.ok()) {
    r.value = static_cast<void *>(status_or.Value());
  }
  return r;
}
#endif

///////////////////////////////////////////////////////////////////////////////////
// Symbols exported in C made available to the Go code.
///////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" {
#endif

// Returns C++ version of the XLA wrapper.
extern int xla_wrapper_version();

// Memory profiling/introspection, using MallocExtension::GetStats()
extern char *memory_stats();
extern size_t memory_usage();
extern bool heap_checker_no_global_leaks();

// NumberToString converts a number to a string. Used for testing.
extern char *number_to_string(int n);

extern bool XlaStatusOk(XlaStatus *status);
extern char *XlaStatusErrorMessage(XlaStatus *status);
extern int XlaStatusCode(XlaStatus *status);
extern void DeleteXlaStatus(XlaStatus *xla_status);

#ifdef __cplusplus
}
#endif

#endif // of #ifndef _GOMLX_XLA_STATUS_H
