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

// aot_exec.h holds C API wrapper for xla::LocalClient and method to run
// Ahead-Of-Time (AOT) compiled code.
#ifndef _GOMLX_XLA_AOT_EXEC_H
#define _GOMLX_XLA_AOT_EXEC_H

#include <stdlib.h>

#include "gomlx/client.h"
#include "gomlx/on_device_buffer.h"
#include "gomlx/status.h"

#ifdef __cplusplus
#include "xla/client/client.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"

// AOTExecutable wraps XLA LocalExecutable, to execute AOT graphs.
struct AOTExecutable {
  std::unique_ptr<xla::LocalExecutable> local_executable;
};

extern "C" {

#else

// C version: includes missing types and forward declarations.
typedef _Bool bool;

struct AOTExecutable;
typedef struct AOTExecutable AOTExecutable;
struct Shape;
typedef struct Shape Shape;

#endif // #ifdef __cplusplus

// NewAOTExecutable (or an error) given the `Client` and a serialized
// xla::AotResult (these are created by `ClientAOTCompileComputation`).
// The `serialized_aot_result` ownership is transferred, and it will be deleted
// after being used.
StatusOr NewAOTExecutable(Client *client, VectorData *serialized_aot_result);

// ExecuteAOT and returns a ShapedBuffer pointer or an error.
// `num_params` and `params` hold the pointers to parameters, its ownership
// is *not* transferred.
StatusOr ExecuteAOT(Client *client, AOTExecutable *exec, int num_params,
                    XlaShapedBuffer **params);

#ifdef __cplusplus
} // extern "C" {
#endif

#endif // #ifndef _GOMLX_XLA_AOT_EXEC_H
