# Setup CGO flags so that it uses local copies of the C/C++ header files and libraries
# during development.
#
# During normal usage (not developing C/C++ wrapper) the better way is just to install
# the GoMLX library in a standard location, like /usr/local/include and /usr/local/lib,
# and not use this script.
(
  [[ -n $ZSH_VERSION && $ZSH_EVAL_CONTEXT =~ :file$ ]] ||
  [[ -n $KSH_VERSION && "$(cd -- "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")" != "$(cd -- "$(dirname -- "${.sh.file}")" && pwd -P)/$(basename -- "${.sh.file}")" ]] ||
  [[ -n $BASH_VERSION ]] && (return 0 2>/dev/null)
) && sourced=1 || sourced=0
if ((sourced == 0)) ; then
  echo "$0 must be sourced ($ source $0) to it exports the necessary env variables."
  exit 1
fi

if [[ "${GOMLX_DIR}" == "" ]] ; then
  GOMLX_DIR="$(git rev-parse --show-toplevel)"
fi
if [[ "${GOMLX_DIR}" == "/" || "${GOMLX_DIR}" == "." || "${GOMLX_DIR}" == "" ]] ; then
  echo "Failed to find gomlx root path." 1>&2
  return
fi

echo "Root GoMLX path is ${GOMLX_DIR}"

# CGO flags.
export CGO_CFLAGS="-I${GOMLX_DIR}/c/bazel-bin/include"
export CGO_CPPFLAGS="-I${GOMLX_DIR}/c/bazel-bin/include"
export CGO_CXXFLAGS="-I${GOMLX_DIR}/c/bazel-bin/include"
export CGO_LDFLAGS="-L${GOMLX_DIR}/c/bazel-bin/gomlx"

# Makes sure the developer library is in the path for dynamic linking
# when the program is run.
export LD_LIBRARY_PATH="${GOMLX_DIR}/c/bazel-bin/gomlx:${LD_LIBRARY_PATH}"

# Disable verbose, often useless and generally annoying TF info logs
#export TF_CPP_MIN_LOG_LEVEL=2
