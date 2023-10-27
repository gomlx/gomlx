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

// computation.h holds C API wrapper for xla::Builder, xla::Computation,
// xla::Literal, xla::GlobalData and related functionality.
#ifndef _GOMLX_XLA_COMPUTATION_H
#define _GOMLX_XLA_COMPUTATION_H

// #include "gomlx/aot/aot.h"
#include "gomlx/client.h"
#include "gomlx/literal.h"
#include "gomlx/node.h"
#include "gomlx/on_device_buffer.h"
#include "gomlx/status.h"

#ifdef __cplusplus
// C++ dependencies.
#include "xla/client/xla_builder.h"
#include "xla/execution_options_util.h"

// Alias to xla::Literal.
typedef xla::GlobalData XlaGlobalData;
typedef xla::XlaOp XlaOp;

#else
typedef _Bool bool;
typedef void XlaGlobalData;

// Forward reference of C++ types.
struct Computation;
typedef struct Computation Computation;
#endif

#ifdef __cplusplus
extern "C" {

// Computation, internal representation to the C++ binding code. In
// contains the builder and later the compiled computation.
struct Computation {
  ~Computation() {
    if (builder != nullptr) {
      delete builder;
    }
  }

  // Builder, set while the Computation is being built.
  xla::XlaBuilder *builder;

  // XlaComputation, available only after the Computation has been built.
  // Computation owns the memory, so if Computation is deleted, this has to be
  // freed as well.
  xla::XlaComputation *xla_comp;

  // Compiled computation.
  xla::ExecutionHandle exec;

  // Compiled computation.
  std::unique_ptr<xla::LocalExecutable> local_exec;
};

#endif

// NewComputation creates a C++ handle to a computation building structure. It
// owns the xla::XlaBuilder, and eventually the xla::XlaComputation and
// xla::ExecutionHandle.
extern Computation *NewComputation(char *name);

// ComputationAddOp creates an xla::XlaOp for the given node description.
// Returns the new op and its shape in the fields `node.new_op` and
// `node.new_shape`. Ownership of the memory is transferred back.
extern XlaStatus *ComputationAddOp(Computation *comp, SerializedNode *node);

// DeleteComputation will destroy associated resources.
extern void DeleteComputation(void *comp);

// DeleteXlaOp delete XlaOp reference.
extern void DeleteXlaOp(XlaOp *op);

// DeleteGlobalData when no longer needed.
extern void DeleteGlobalData(XlaGlobalData *gd);

// TransferGlobalData brings data from accelerator server. Returns a Literal* or
// an error (Status).
extern StatusOr TransferFromServer(XlaGlobalData *gd, Client *client);

// GlobalDataShape retrieves only the shape for the GlobalData. Returns a Shape*
// or an error (Status).
extern StatusOr GlobalDataShape(XlaGlobalData *gd, Client *client);

// GlobalDataDecomposeTuple splits tuple into its components. Return array of
// XlaGlobalData or an error. Size of the array is given by the gd's
// shape->tuple_size.
extern StatusOr GlobalDataDeconstructTuple(XlaGlobalData *gd, Client *client);

// ClientCompileComputation should be called after all the ops are added to the
// computation, it will finalize the building and compile the computation graph.
// No more ops can be added after this. The ownership of the array param_shapes
// is not transferred. Returns nullptr if there were no errors.
extern XlaStatus *ClientCompileComputation(Client *client, Computation *comp,
                                           int num_params, Shape **param_shapes,
                                           XlaOp *output);

// ComputationExecuteComputation executes a previously compiled computation. It
// returns a ShapedBuffer pointer or an error. `num_params` and `params` hold
// the pointers to parameters, its ownership is *not* transferred.
extern StatusOr ClientExecuteComputation(Client *client, Computation *comp,
                                         int num_params,
                                         XlaShapedBuffer **params);

#ifdef __cplusplus
}
#endif

#endif