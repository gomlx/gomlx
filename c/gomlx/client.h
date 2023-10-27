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
#ifndef _GOMLX_XLA_CLIENT_H
#define _GOMLX_XLA_CLIENT_H

#include <stdlib.h>

#include "gomlx/on_device_buffer.h"
#include "gomlx/status.h"

#ifndef __cplusplus
typedef _Bool bool;
#endif

#ifdef __cplusplus
// #include "xla/client/client.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/service/backend.h"

// Client wraps XLA stuff needed to compile and execute computations. Stuff that
// are shared across multiple computation graphs. Still lots of guess work here
// (XLA is poorly documented).
struct Client {
  // xla::LocalClient is owned by XLA, only a reference is kept here.
  xla::LocalClient *client;

  // Cliend data.
  xla::ExecutableRunOptions exec_options;
  int device_count, default_device_ordinal;
};

// Forward declaration, defined in computation.cpp.
class Computation;

extern "C" {

#else
struct Client;
typedef struct Client Client;

// Forward declaration, defined elsewhere.
struct Computation;
typedef struct Computation Computation;
struct Shape;
typedef struct Shape Shape;
#endif

// GetPlatforms enumerates list of platform names available. Return StatusOr of
// a VectorPointers of strings.
StatusOr GetPlatforms();

// NewClient returns a wrapper object that holds XLA::Client and allow execution
// of computations. The ownership of `platform_name` is transferred.
StatusOr NewClient(char *platform_name, int num_replicas, int num_threads);

// DeleteClient deletes the client and associated data. Notice that all
// computations compiled with this client will become invalid.
void DeleteClient(Client *client);

// ClientDevices sets device_count and default_device_ordinal.
void ClientDevices(Client *client, int64_t *device_count,
                   int64_t *default_device_ordinal);

#ifdef __cplusplus
}
#endif

#endif