#!/bin/bash
#
# The following environment variables and flags can be defined:
#
# * STARTUP_FLAGS and BUILD_FLAGS: passed as `bazel ${STARTUP_FLAGS} build <build_target> ${BUILD_FLAGS}.
# * --tpu: Also compile for TPUs.
# * --gpu: Also compile for GPUs.
# * --output <dir>: Directory to use for build.
# * <build_target>: Default is ":gomlx_xla". Another common option is ":gomlx_xla_lib".

#BUILD_TARGET=":gomlx_xla_lib"
BUILD_TARGET=":gomlx_xla"

export USE_BAZEL_VERSION=6.1.1  # Latest as of this writing.

USE_GPU=0
USE_TPU=0
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      USE_GPU=1
      shift # past argument
      ;;
    --tpu)
      USE_TPU=1
      echo "TPU support not tested/tried yet ... this is experimental."
      shift # past argument
      ;;
    --output)
      shift
      echo "Output directory set to $1"
      OUTPUT_DIR="--output_base=$1"
      shift
      ;;
    -*|--*)
      echo "Unknown flag $1"
      exit 1
      ;;
    *)
      BUILD_TARGET="$1"
      shift
  esac
done


set -vex

BAZEL=${BAZEL:-bazel}  # Bazel version > 5.
PYTHON=${PYTHON:-python}  # Python, version should be > 3.7.

# Check the OpenXLA version (commit hash) changed, and if so, download an
# updated `openxla_xla_bazelrc` file from github.
# TODO: include a sha256 verification of the file as well.
if ! grep -q "OPENXLA_XLA_COMMIT_HASH" WORKSPACE ; then
  echo "Did not find OPENXLA_XLA_COMMIT_HASH in WORKSPACE file!?"
  exit 1
fi
OPENXLA_XLA_COMMIT_HASH="$(
  grep -E "^OPENXLA_XLA_COMMIT_HASH[[:space:]]*=" WORKSPACE |\
    sed -n 's/^[^"]*"\([^"]*\)".*/\1/p'
)"
OPENXLA_BAZELRC="openxla_xla_bazelrc"
if [[ ! -e "${OPENXLA_BAZELRC}" || ! -e "${OPENXLA_BAZELRC}.version" \
  || "$(< "${OPENXLA_BAZELRC}.version")" != "${OPENXLA_XLA_COMMIT_HASH}" ]] ; then
    echo "Fetching ${OPENXLA_BAZELRC} at version \"${OPENXLA_XLA_COMMIT_HASH}\""
    curl "https://raw.githubusercontent.com/openxla/xla/${OPENXLA_XLA_COMMIT_HASH}/.bazelrc" -o ${OPENXLA_BAZELRC}
    echo "${OPENXLA_XLA_COMMIT_HASH}" > "${OPENXLA_BAZELRC}.version"
else
    echo "File ${OPENXLA_BAZELRC} at version \"${OPENXLA_XLA_COMMIT_HASH}\" already exists, not fetching."
fi

STARTUP_FLAGS="${STARTUP_FLAGS} ${OUTPUT_DIR}"
STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=${OPENXLA_BAZELRC}"

# bazel build flags
BUILD_FLAGS="${BUILD_FLAGS:---keep_going --verbose_failures --sandbox_debug}"
BUILD_FLAGS="${BUILD_FLAGS} --config=linux"  # Linux only for now.
if ((USE_GPU)) ; then
  BUILD_FLAGS="${BUILD_FLAGS} --config=cuda"  # Include CUDA.
fi
if ((USE_TPU)) ; then
  BUILD_FLAGS="${BUILD_FLAGS} --config=tpu"  # Include CUDA.
fi

# OpenXLA sets this to true for now to link with TF. But we need this enabled:
BUILD_FLAGS="${BUILD_FLAGS} --define tsl_protobuf_header_only=false"

# We need the dependencies to be linked statically -- they won't come from some external .so:
BUILD_FLAGS="${BUILD_FLAGS} --define framework_shared_object=false"


# XLA rules weren't meant to be exported, so we overrule their visibility
# constraints.
BUILD_FLAGS="${BUILD_FLAGS} --check_visibility=false"

# Required from some `third_party/tsl` package:
BUILD_FLAGS="${BUILD_FLAGS} --experimental_repo_remote_exec"

# Invoke bazel build
time "${BAZEL}" ${STARTUP_FLAGS} build ${BUILD_TARGET} ${BUILD_FLAGS} --build_tag_filters=-tfdistributed