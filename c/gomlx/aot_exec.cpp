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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "gomlx/client.h"
#include "gomlx/status.h"

// #include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/execution_options_util.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/types.h"
// #include "xla/xla_data.pb.h"

#include "aot_exec.h"

// #include "xla/service/cpu/executable.pb.h"

using namespace std;

StatusOr NewAOTExecutable(Client *client, VectorData *serialized_aot_result) {
  StatusOr r{0, 0};

  int num_bytes = serialized_aot_result->count;
  string aot_result_str(static_cast<const char *>(serialized_aot_result->data),
                        num_bytes);
  free(serialized_aot_result->data);
  free(serialized_aot_result);

  //    xla::cpu::XlaRuntimeCpuExecutableProto exec_proto;
  //    cerr << "NewAOTExecutable(): parsing executable -> " <<
  //    exec_proto.ParseFromString(aot_result_str) << endl; cerr <<
  //    "\tSerialized aot result (" << num_bytes << " bytes): " <<
  //    aot_result_str.size() << endl;
  //
  // Create executable.
  xla::ExecutableBuildOptions executable_build_options;
  cerr << "\tClient platform: " << client->client->platform()->Name() << endl;
  auto executable_or =
      client->client->Load(aot_result_str, executable_build_options);
  if (!executable_or.ok()) {
    r.status = FromStatus(executable_or.status());
    return r;
  }
  cerr << "\tClient platform: " << client->client->platform()->Name() << endl;

  AOTExecutable *exec = new AOTExecutable;
  exec->local_executable = std::move(executable_or.value());
  r.value = exec;
  cerr << "\tLoaded executable!" << endl;
  return r;
}

StatusOr ExecuteAOT(Client *client, AOTExecutable *exec, int num_params,
                    XlaShapedBuffer **params) {
  StatusOr r{0, 0};
  absl::Span<xla::ShapedBuffer *const> arguments(params, num_params);
  auto data_or = exec->local_executable->Run(arguments, client->exec_options);
  if (!data_or.ok()) {
    r.status = FromStatus(data_or.status());
    return r;
  }
  xla::ScopedShapedBuffer *ssb =
      new xla::ScopedShapedBuffer(std::move(*data_or));
  OnDeviceBuffer *wrapper = new OnDeviceBuffer();
  wrapper->ssb_buffer = ssb;
  r.value = static_cast<void *>(wrapper);
  return r;
}