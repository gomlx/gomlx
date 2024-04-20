package(default_visibility = ["//visibility:public"])

load("@xla//xla/tsl:tsl.bzl", "if_cuda_or_rocm", "if_with_tpu_support")

# Dependencies needed for Just-In-Time (JIT) compilation: what we use for training.
gomlx_jit_deps = [
    # "@xla//xla/mlir/utils:error_util", Temporarily replaced by:
    "//deps/xla_mlir:error_util",
    "@com_google_absl//absl/types:span",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/base:log_severity",
    "@com_google_absl//absl/log:initialize",
    "@xla//xla:comparison_util",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla:status",
    "@xla//xla:statusor",
    "@xla//xla:types",
    "@xla//xla:util",
    "@xla//xla:xla_data_proto_cc",
    "@xla//xla:xla_proto_cc",
    "@xla//xla/service:platform_util",
    "@xla//xla/client:client",
    "@xla//xla/client:client_library",
    "@xla//xla/client:xla_builder",
    "@xla//xla/client:xla_computation",
    "@xla//xla/client:executable_build_options",
    "@xla//xla/client/lib:lu_decomposition",
    "@xla//xla/client/lib:math",
    "@xla//xla/client/lib:qr",
    "@xla//xla/client/lib:svd",
    "@xla//xla/client/lib:self_adjoint_eig",
    "@xla//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
    "@xla//xla/stream_executor:stream_executor",

    # Do we need these explicitly ?
    # "@xla//xla/stream_executor:dnn",
    # @xla//xla/stream_executor:event",
    # "@xla//xla/stream_executor:scratch_allocator",

    # AOT-Compilation: StableHLO conversion,
    "@stablehlo//:stablehlo_serialization",
    "@stablehlo//:stablehlo_portable_api",

    # CPU (Host) stream executor / platform support.
    "@xla//xla/service:cpu_plugin",
    "@xla//xla/service/cpu:cpu_compiler",
    "@xla//xla/stream_executor/host:host_platform",
] + if_with_tpu_support([
    # TPU stream executor: do we need to link something form xla/service: ?
    "@xla//xla/stream_executor/tpu:tpu_executor",
    #
]) + if_cuda_or_rocm([
    # CUDA or ROCM executor
    "@xla//xla/service:gpu_plugin",
    "@xla//xla/stream_executor:cuda_platform",
    #    "@xla//xla/stream_executor:rocm_platform",
])

# Minimal set of dependencies needed to run inference in an Ahead-Of-Time (AOT) compiled model.
gomlx_aot_deps = [
    "@com_google_absl//absl/types:span",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/base:log_severity",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla:status",
    "@xla//xla:statusor",
    "@xla//xla:types",
    "@xla//xla/client:local_client",
    #    "@xla//xla/client:client",
    "@xla//xla/client:client_library",

    # CPU (Host) stream executor / platform support.
    "@xla//xla/service/cpu:cpu_runtime",
] + if_with_tpu_support([
    # TPU stream executor: do we need to link something form xla/service: ?
    #    "@xla//xla/stream_executor/tpu:tpu_executor",
    #
]) + if_cuda_or_rocm([
    # CUDA or ROCM executor
    #    "@xla//xla/service/gpu:gpu_runtime",
    #    "@xla//xla/service:gpu_plugin",
    #    "@xla//xla/stream_executor:cuda_platform",
    #    "@xla//xla/stream_executor:rocm_platform",
])

cc_library(
    name = "gomlx",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    # SKIP_ABSL_INITIALIZE_LOG: this will trigger skipping of the of the call to `absl::InitializeLog()`
    # to prevent double-calling, in case whoever is linking GoMLX also calls it. Setting it may
    # lead to spurious logging by XLA library at startup up.
    # defines = ["SKIP_ABSL_INITIALIZE_LOG"],
    linkopts = ["-shared"],
    deps = gomlx_jit_deps + [
        "@gperftools",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libgomlx_xla.so",
    linkopts = [
        "-shared",
        "-ltcmalloc",
    ],
    # Setting linkshared=1 and linkstatic=1 should try to link everything (except system libraries) statically
    # but generate a dynamically linked library -- needed to link it to other languages, Go in this case.
    # At least that's what I understood from reading the docs in
    # https://bazel.build/reference/be/c-cpp#cc_binary.linkshared
    linkshared = 1,
    linkstatic = 1,
    deps = gomlx_jit_deps + [
        ":gomlx",
    ],
)

cc_library(
    name = "gomlx_aot",
    srcs = [
        "aot_exec.cpp",
        "client.cpp",
        "literal.cpp",
        "on_device_buffer.cpp",
        "shape.cpp",
        "status.cpp",
    ],
    hdrs = [
        "aot_exec.h",
        "client.h",
        "literal.h",
        "on_device_buffer.h",
        "shape.h",
        "status.h",
    ],
    linkopts = ["-shared"],
    deps = [
        "@gperftools",
    ] + gomlx_aot_deps,
    alwayslink = True,
)

cc_binary(
    name = "libgomlx_xla_aot.so",
    linkopts = ["-shared"],
    # Setting linkshared=1 and linkstatic=1 should try to link everything (except system libraries) statically
    # but generate a dynamically linked library -- needed to link it to other languages, Go in this case.
    # At least that's what I understood from reading the docs in
    # https://bazel.build/reference/be/c-cpp#cc_binary.linkshared
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":gomlx_aot",
    ] + gomlx_aot_deps,
)

filegroup(
    name = "headers",
    srcs = glob(["*.h"]),
)
