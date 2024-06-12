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

// client.h holds C API wrapper for xla::LocalClient, which can be used by Go.
// This file only contains things needed by inference in AOT mode.
#ifndef _GOMLX_XLA_AOT_COMPILE_H
#define _GOMLX_XLA_AOT_COMPILE_H

#include <stdlib.h>

#include "gomlx/client.h"
#include "gomlx/computation.h"
#include "gomlx/status.h"

#ifndef __cplusplus
typedef _Bool bool;
#endif

#ifdef __cplusplus
#include "mlir/IR/BuiltinOps.h"
// #include "xla/client/client.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/service/backend.h"

// StableHLOHolder holds a StableHLO object and its Context.
struct StableHLOHolder {
  mlir::ModuleOp stable_hlo;

  // context where the MLIR's StableHLO is defined. If delete, the stable_hlo
  // becomes invalid.
  std::unique_ptr<mlir::MLIRContext> context;
};

extern "C" {

#else  // __cplusplus
// Forward ceclarations for C.
struct StableHLOHolder;
typedef struct StableHLOHolder StableHLOHolder;
#endif // __cpluplus

// ConvertComputationToStableHLO converts a **compiled** computation graph to
// the StableHLO representation. It returns either an error or a
// `StableHLOHolder*` that holds the StableHLO C++ object. Returned
// StableHLOHolder object is owned and needs to be deleted by the caller.
extern StatusOr ConvertComputationToStableHLO(Computation *comp);

// DeleteStableHLOHolder and its contained data.
extern void DeleteStableHLOHolder(StableHLOHolder *holder);

// StableHLOToString print-prints the StableHLO for human consumption.
// Returned string is owned and needs to be freed by the caller.
extern char *StableHLOToString(StableHLOHolder *holder);

// StableHLOCurrentVersion returns the current supported version of StableHLO.
// Returned string is owned and needs to be freed by the caller.
extern char *StableHLOCurrentVersion();

// SerializeStableHLO to bytecode that can presumably be used by PjRT and IREE,
// as well as embedded in one of the TensorFlow SavedModel formats.(??)
//
// Return true if it succeeds, false if failed.
//
// Probably you want to use StableHLOCurrentVersion() for `version`. The string
// will be freed.
//
// The file_descriptor is not closed at the end for the call (but written
// content is flushed).
extern bool SerializeStableHLO(StableHLOHolder *holder, char *version,
                               int file_descriptor);

// UnserializeStableHLO takes a byte array in VectorData and constructs a
// StableHLO. It creates a new MLIRContext to hold it -- meaning that for now it
// does not support unserializing more than one StableHLO program to the same
// context.
//
// VectorData and its associated data is freed before returning.
//
// It returns an error or a StableHLOHolder.
extern StatusOr UnserializeStableHLO(VectorData *serialized);

#ifdef __cplusplus
}
#endif

#endif // _GOMLX_XLA_AOT_COMPILE_H