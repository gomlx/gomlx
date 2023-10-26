load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//visibility:private"])

# This is the genrule used by TF install headers to correctly
# map headers into a directory structure
genrule(
    name = "gomlx_headers_include",
    srcs = ["//gomlx:headers"],
    outs = ["include"],
    cmd = """
    mkdir -p "$@"
    mkdir -p "$@/gomlx"
    echo "out=$@"
    for f in $(locations //gomlx:headers); do
      echo "$${f}"
      cp "$${f}" "$@/$${f}"
    done
    """,
)

# This genrule remaps libxla_extension.so to lib/libxla_extension.so
genrule(
    name = "gomlx_xla_lib",
    srcs = [
        "//gomlx:libgomlx_xla.so",
        "@gperftools//:libtcmalloc.a",
        # "//gomlx/aot:libgomlx_xla.so",
    ],
    outs = ["lib"],
    cmd = """
    mkdir "$@"
    ln $(location //gomlx:libgomlx_xla.so) "$@"

    mkdir "$@/gomlx"
    GPERFTOOLS_LIB="$$(pwd)/$$(dirname $(location @gperftools//:libtcmalloc.a))"
    if [ -d "$$GPERFTOOLS_LIB"/lib ]; then
      GPERFTOOLS_LIB="$$GPERFTOOLS_LIB"/lib
    fi
    ln "$$GPERFTOOLS_LIB"/lib* "$@/gomlx"
    """,
)

# Tar packaging: notice we force date/time and ordering to be deterministic, to generate the
# always same archive, if the contents are the same. See https://reproducible-builds.org/docs/archives/.
genrule(
    name = "gomlx_xla",
    srcs = [
        ":gomlx_headers_include",
        ":gomlx_xla_lib",
    ],
    outs = ["gomlx_xla.tar.gz"],
    cmd_bash = """
    tt="$$(mktemp -d)"
    files=""
    for ii in $(SRCS) ; do
        ln -sf "$$(pwd)/$${ii}" "$${tt}"
        files="$${files} $$(basename $${ii})"
    done
    echo "$${files}"
    TAR=tar
    if [[ "$$OSTYPE" == "darwin"* ]]; then
        if ! command -v gtar &> /dev/null; then
            echo -e "\\033[0;31mgtar (gnu-tar) cannot not be found. gnu-tar can be installed using:\nbrew install gnu-tar"
        exit
        fi
        TAR=gtar
    fi
    $$TAR --create --dereference --directory="$${tt}" \
        --sort=name --owner=0 --group=0 --numeric-owner --format=gnu \
        --gzip --file "$@" $${files}
    rm -rf "$${tt}"
    """,
)

genrule(
    name = "gomlx_xla_quick",
    srcs = [
        ":gomlx_headers_include",
        ":gomlx_xla_lib",
    ],
    outs = ["gomlx_xla_quick.txt"],
    cmd_bash = """
    touch "$@"
    """,
)
