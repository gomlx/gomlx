genrule(
    name = "run_configure_make",
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

cc_library(
    name = "gperftools",
    hdrs = [
        "src/gperftools/malloc_extension.h",
    ],
    includes = ["src"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
