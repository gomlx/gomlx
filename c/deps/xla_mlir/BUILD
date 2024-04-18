load("@xla//xla/tsl:tsl.bzl", "if_with_tpu_support", "if_cuda_or_rocm")

cc_library(
    name = "error_util",
    srcs = ["error_util.cc"],
    hdrs = ["error_util.h"],
    #    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:errors",
    ],
)
