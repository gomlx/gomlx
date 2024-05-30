# Description:
#   Auto-imported from github.com/gomlx/gomlx

load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = [":users"],
)

license(
    name = "license",
    package_name = "gomlx",
)

licenses(["notice"])

package_group(
    name = "users",
    packages = [
        "//learning/deepmind/golang/...",
        "//third_party/golang/github_com/gomlx/gomlx/v/v0/...",
        "//third_party/gxlang/...",
    ],
)

exports_files(["LICENSE"])
