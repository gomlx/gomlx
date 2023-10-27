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

#include "gomlx/status.h"

// #include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/execution_options_util.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

#include "client.h"

using namespace std;

StatusOr GetPlatforms() {
  StatusOr r{0, 0};
  auto platforms_or = xla::PlatformUtil::GetSupportedPlatforms();
  if (!platforms_or.ok()) {
    r.status = FromStatus(platforms_or.status());
    return r;
  }

  auto platforms = platforms_or.value();
  std::vector<std::string> names;
  for (const auto &platform : platforms) {
    if (platform->VisibleDeviceCount() > 0) {
      names.push_back(platform->Name());
    }
  }
  r.value = c_vector_str(names);
  return r;
}

StatusOr NewClient(char *platform_name, int num_replicas, int num_threads) {
  StatusOr r{0, 0};

  // Get selected platform by name.
  auto platform_or = xla::PlatformUtil::GetPlatform(platform_name);
  free(platform_name);
  if (!platform_or.ok()) {
    r.status = FromStatus(platform_or.status());
    return r;
  }
  auto platform = platform_or.value();

  // Create client with given options.
  xla::LocalClientOptions opt(platform, num_replicas, num_threads);
  Client *client = new Client();
  auto client_or = xla::ClientLibrary::GetOrCreateLocalClient(opt);
  if (!client_or.ok()) {
    r.status = FromStatus(client_or.status());
    return r;
  }
  client->client = client_or.value();
  const xla::Backend &backend = client->client->backend();
  client->device_count = backend.device_count();
  client->default_device_ordinal = backend.default_device_ordinal();

  // Create ExecutableRunOptions.
  client->exec_options.set_allocator(backend.memory_allocator());
  client->exec_options.set_device_ordinal(client->default_device_ordinal);
  client->exec_options.set_intra_op_thread_pool(
      backend.eigen_intra_op_thread_pool_device());

  r.value = client;
  return r;
}

void ClientDevices(Client *client, int64_t *device_count,
                   int64_t *default_device_ordinal) {
  *default_device_ordinal =
      static_cast<int64_t>(client->default_device_ordinal);
  *device_count = static_cast<int64_t>(client->device_count);
}

void DeleteClient(Client *client) {
  if (client->client != nullptr) {
    delete client->client;
    client->client = nullptr;
  }
  delete client;
}
