load("@tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
