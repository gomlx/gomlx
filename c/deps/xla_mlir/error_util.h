/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Renamed as we made a local copy of error_util.c into GoMLX.
//
// --> It also required s/BaseScopedDiagnosticHandler/TmpBaseScopedDiagnosticHandler
//     for some obscure C++ reason -- useless GCC output.
//#ifndef XLA_MLIR_UTILS_ERROR_UTIL_H_
//#define XLA_MLIR_UTILS_ERROR_UTIL_H_
#ifndef DEPS_XLA_MLIR_ERROR_UTIL_H_
#define DEPS_XLA_MLIR_ERROR_UTIL_H_

#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/SourceMgr.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Location.h"  // from @llvm-project
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project

// Error utilities for MLIR when interacting with code using Status returns.
// Temporarily renamed namespace to gomlx_mlir.
namespace mlir {
// Diagnostic handler that collects all the diagnostics reported and can
// produce a Status to return to callers. This is for the case where MLIR
// functions are called from a function that will return a Status: MLIR code
// still uses the default error reporting, and the final return function can
// return the Status constructed from the diagnostics collected.
class TmpBaseScopedDiagnosticHandler : public SourceMgrDiagnosticHandler {
 public:
  explicit TmpBaseScopedDiagnosticHandler(MLIRContext* context,
                                       bool propagate = false,
                                       bool filter_stack = false);
  // On destruction error consumption is verified.
  ~TmpBaseScopedDiagnosticHandler();
  // Returns whether any errors were reported.
  bool ok() const;

  // Returns Status corresponding to the diagnostics reported. This consumes
  // the diagnostics reported and returns a Status of type Unknown. It is
  // required to consume the error status, if there is one, before destroying
  // the object.
  absl::Status ConsumeStatus();

  // Returns the combination of the passed in status and consumed diagnostics.
  // This consumes the diagnostics reported and either appends the diagnostics
  // to the error message of 'status' (if 'status' is already an error state),
  // or returns an Unknown status (if diagnostics reported), otherwise OK.
  absl::Status Combine(absl::Status status);

 protected:
  LogicalResult handler(Diagnostic* diag);

  // String stream to assemble the final error message.
  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;

  // A SourceMgr to use for the base handler class.
  llvm::SourceMgr source_mgr_;

  // Whether to propagate diagnostics to the old diagnostic handler.
  bool propagate_;
};
}  // namespace mlir

#endif  // DEPS_XLA_MLIR_ERROR_UTIL_H_
