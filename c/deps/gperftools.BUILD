# Runs ./configure and ./make -- gperftools doesn't support Bazel.
# See more in https://github.com/gperftools/gperftools/issues/800
genrule(
    name = "gperftools_run_configure_make",
    srcs = glob(["**"]),
    outs = ["libtcmalloc.a"],
    cmd = """
    export OUT="$$(pwd)/$(@D)"
    cd external/gperftools
    ./configure --enable-shared
    make -j
    echo "Copying to $$OUT"
    ls ./.libs/libtcmalloc.*
    cp -fv ./.libs/libtcmalloc.* "$$OUT"
    """,
)

# This rule exposes the header files used by GoMLX.
cc_library(
    name = "gperftools",
    hdrs = [
        "src/gperftools/malloc_extension.h",
        "src/gperftools/heap-checker.h",
    ],
    includes = ["src"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
