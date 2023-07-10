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

/*
    This file has the implementations of AOT (Ahead-Of-Time) Compilation of computation graphs, for various
    devices.

    The header for this file is in client.h.
*/

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "gomlx/aot_compile.h"

#include "gomlx/client.h"
#include "gomlx/literal.h"
#include "gomlx/status.h"
#include "gomlx/on_device_buffer.h"

#include "absl/types/span.h"
#include "absl/types/span.h"
#include "absl/strings/str_format.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h" // from @llvm-project

#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/api/PortableApi.h"

#include "xla/array.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/xla_builder.h"
#include "xla/debug_options_flags.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/literal.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "xla/status.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"

using namespace std;

// StableHLOToStringImplementation calls the print() method of the StableHLO given in a ModuleOp to a string buffer.
std::string StableHLOToStringImplementation(mlir::ModuleOp module) {
    std::string moduleStr;
    llvm::raw_string_ostream stringStream(moduleStr);
    module.print(stringStream);
    stringStream.flush();
    return moduleStr;
}

// ConvertXlaComputationToStableHLOImplementation returns teh StableHLO referenced by an mlir::ModuleOp within the given `mlir::MLIRContext`.
// If the context is destroyed the returned StableHLO becomes invalid.
xla::StatusOr<mlir::ModuleOp> ConvertXlaComputationToStableHLOImplementation(const xla::XlaComputation &xla_comp, mlir::MLIRContext &context) {
    xla::HloModuleProto hlo_module_proto = xla_comp.proto();
    TF_ASSIGN_OR_RETURN(
        xla::ProgramShape program_shape,
        xla_comp.GetProgramShape()
    );
    xla::HloModuleConfig module_config(program_shape);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::HloModule> hlo_module,
        xla::HloModule::CreateFromProto(hlo_module_proto, module_config)
    );

    // Convert to MHLO (MLIR HLO)
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    auto status = ConvertHloToMlirHlo(module, &hlo_module_proto, true);
    if (!status.ok()) {
        return status;
    }

    mlir::BaseScopedDiagnosticHandler diagnostic(&context);  // Error collector.
    if (!mlir::verify(module).succeeded()) {
        return xla::FromAbslStatus(diagnostic.ConsumeStatus());
    }

    // Convert to StableHLO.
    mlir::PassManager pm(&context);
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
    if (!mlir::succeeded(pm.run(module))) {
        return xla::FromAbslStatus(diagnostic.ConsumeStatus());
    }
    return module;
}

StatusOr ConvertComputationToStableHLO(Computation *comp)  {
    StatusOr r{0, 0};
    if (comp->xla_comp == nullptr) {
        r.status = new xla::Status(absl::StatusCode::kInvalidArgument, "Computation hasn't been compiled yet");
        return r;
    }

    std::unique_ptr<StableHLOHolder> holder(new StableHLOHolder());
    holder->context.reset(new mlir::MLIRContext());
    auto status_or = ConvertXlaComputationToStableHLOImplementation(*comp->xla_comp, *holder->context);
    if (!status_or.ok()) {
        r.status = FromStatus(status_or.status());
        return r;
    }
    holder->stable_hlo = status_or.value();
    r.value = static_cast<void *>(holder.release());
    return r;
}

void DeleteStableHLOHolder(StableHLOHolder *holder) {
    if (holder != nullptr && holder->context) {
        if (holder->stable_hlo) {
            holder->stable_hlo.erase();
        }
        holder->context.release();
        delete holder;
    }
}

char *StableHLOToString(StableHLOHolder *holder) {
    if (holder == nullptr || !holder->context || !holder->stable_hlo) {
        return nullptr;
    }
    auto result = StableHLOToStringImplementation(holder->stable_hlo);
    return c_str(result);
}

char *StableHLOCurrentVersion() {
    string version = mlir::stablehlo::getCurrentVersion();
    return c_str(version);
}

bool SerializeStableHLO(StableHLOHolder *holder, char *version, int file_descriptor) {
    string version_str(version);
    free(version);
    llvm::raw_fd_ostream output(file_descriptor, /* shouldClose= */false);
    if (!mlir::stablehlo::serializePortableArtifact(holder->stable_hlo, version_str, output).succeeded()) {
        return false;
    }
    // Documentation does not specify if it flushes in the output, so we do it explicitly here.
    output.flush();
    return true;
}

StatusOr UnserializeStableHLO(VectorData *serialized) {
    StatusOr r{0, 0};
    std::unique_ptr<StableHLOHolder> holder(new StableHLOHolder());
    holder->context.reset(new mlir::MLIRContext());
    mlir::BaseScopedDiagnosticHandler diagnostic(holder->context.get());  // Error collector.
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::stablehlo::deserializePortableArtifact(
        llvm::StringRef(static_cast<const char *>(serialized->data), serialized->count), holder->context.get());

    free(serialized->data);
    serialized->data = nullptr;
    serialized->count = 0;
    delete serialized;

    if (!module) {
        r.status = FromStatus(xla::FromAbslStatus(diagnostic.ConsumeStatus()));
        return r;
    }
    holder->stable_hlo = module.release();
    r.value = static_cast<void *>(holder.release());
    return r;
}

// aotCompileCpuExecutable compiles for CPU, based on `openxla/xla/xla/service/xla_compile_main.cc`.
xla::StatusOr<std::string> aotCompileCpuExecutable(
    std::unique_ptr<xla::HloModule> hlo_module) {
  xla::cpu::CpuCompiler cpu_compiler;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::cpu::CpuExecutable> cpu_executable,
      cpu_compiler.CompileXlaRuntimeCpuExecutable(std::move(hlo_module)));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::AotCompilationResult> aot_result,
                      cpu_compiler.Export(cpu_executable.get()));
  TF_ASSIGN_OR_RETURN(std::string result, aot_result->SerializeAsString());

  cerr << "aotCompileCpuExecutable: Serialized to " << result.size() << " bytes." << endl;
  xla::cpu::XlaRuntimeCpuExecutableProto exec_proto;
  cerr << "aotCompileCpuExecutable: Parsing executable: " << exec_proto.ParseFromString(result) << endl;

  return result;
}

xla::StatusOr<std::string> aotCompileToSerialized2(
    stream_executor::Platform *platform, std::unique_ptr<xla::HloModule> hlo_module) {
  cerr << "Finding compiler..." << endl;
  TF_ASSIGN_OR_RETURN(xla::Compiler * compiler,
                      xla::Compiler::GetForPlatform(platform));
  cerr << "Got compiler..." << endl;
  xla::AotCompilationOptions options(platform->id());
  std::unique_ptr<xla::HloModuleGroup> hlo_module_group(new xla::HloModuleGroup(std::move(hlo_module)));
  cerr << "Got hlo_module_group ..." << endl;
  TF_ASSIGN_OR_RETURN(
    std::vector<std::unique_ptr<xla::AotCompilationResult>> aot_result,
    compiler->CompileAheadOfTime(std::move(hlo_module_group), options)
  );
  cerr << "Compilations yielded " << aot_result.size() << " results!" << endl;
//    std::unique_ptr<HloModuleGroup> module_group,
//                                const AotCompilationOptions& options)
  TF_ASSIGN_OR_RETURN(std::string result, aot_result[0]->SerializeAsString());
  cerr << "Successfully serialized!" << endl;

  return result;
}

xla::StatusOr<std::string> aotCompileToSerialized3(
    Client *client, Computation *comp, int num_params, Shape **param_shapes) {
    xla::ExecutableBuildOptions executable_build_options;
    std::vector<xla::Shape> shapes(num_params);
    for (int ii = 0; ii < num_params; ii++) {
        shapes[ii] = MakeXlaShape(param_shapes[ii]);
    }
    std::vector<xla::Shape*> shape_pointers(num_params);
    for (int ii = 0; ii < num_params; ii++) {
        shape_pointers[ii] = &shapes[ii];
    }
    cerr << "Just before LocalClient::CompileAheadOfTime() call: " << num_params << " parameters." <<  endl;
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<xla::AotCompilationResult>> aot_result,
        client->client->CompileAheadOfTime(*comp->xla_comp, shape_pointers, executable_build_options)
    );
    cerr << "Compilations yielded " << aot_result.size() << " results!" << endl;
    TF_ASSIGN_OR_RETURN(string result, aot_result[0]->SerializeAsString());
    return result;
}


StatusOr ClientAOTCompileComputation(Client *client, Computation *comp, int num_params, Shape **param_shapes)  {
    StatusOr r{0, 0};
    if (comp->xla_comp == nullptr) {
        r.status = new xla::Status(absl::StatusCode::kInvalidArgument, "Computation hasn't been compiled yet");
        return r;
    }

    if (false) {
        auto status_or = aotCompileToSerialized3(client, comp, num_params, param_shapes);
        if (!status_or.ok()) {
            r.status = FromStatus(status_or.status());
            return r;
        }
        r.value = static_cast<void *>(c_str(status_or.value()));
        return r;
    }

    // Convert to HLO Module.
    xla::HloModuleProto hlo_module_proto = comp->xla_comp->proto();
    auto shape_or = comp->xla_comp->GetProgramShape();
    if (!shape_or.ok()) {
        r.status = FromStatus(shape_or.status());
        return r;
    }
    auto shape = std::move(shape_or.value());
    xla::DebugOptions debug_options = xla::DefaultDebugOptionsIgnoringFlags();
    xla::HloModuleConfig config(shape);
    config.set_debug_options(debug_options);
    auto module_or = xla::HloModule::CreateFromProto(hlo_module_proto, config);
    if (!module_or.ok()) {
        r.status = FromStatus(module_or.status());
        return r;
    }
    std::unique_ptr<xla::HloModule> hlo_module = std::move(module_or.value());

    // Switch per client platform:
    const auto platform = client->client->platform();
    const auto platform_id = platform->id();
    if (platform_id == stream_executor::host::kHostPlatformId) {
        auto string_or = aotCompileCpuExecutable(std::move(hlo_module));
        // auto string_or = aotCompileToSerialized2(platform, std::move(hlo_module));
        if (!string_or.ok()) {
            r.status = FromStatus(string_or.status());
        }  else {
            r.value = static_cast<void *>(str_to_bytes(string_or.value()));
        }
    } else {
        r.status = new xla::Status(absl::StatusCode::kInvalidArgument,
            "Unsupported platform "+platform->Name()+", not able to AOT compile");
    }
    return r;
}
