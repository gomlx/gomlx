FROM --platform=linux/amd64 ubuntu:noble AS gomlx_jupyterlab_base
LABEL maintainer="pfeifer@gmail.com"

#######################################################################################################
# User, home directory and basics.
#######################################################################################################
# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Update apt and install basic utils.
# C compiler is included because some of the Go dependencies require C.
RUN apt-get update --yes \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
      sudo wget vim micro unzip hdf5-tools libc6-dev curl gcc g++ git \
      openssh-client

# Create NB_USER (jovyan) and its home directory.
ARG NB_USER="jupyter"
ARG NB_UID=1001
ARG NB_GID=100
RUN useradd --create-home --uid ${NB_UID} --gid ${NB_GID} --shell /bin/bash ${NB_USER}
ENV HOME /home/${NB_USER}

#######################################################################################################
# Jupyter Lab
#######################################################################################################
USER root
RUN apt install --yes --no-install-recommends python3 python3-venv
ENV PATH=${PATH}:${HOME}/.local/bin

USER ${NB_USER}
WORKDIR ${HOME}
RUN <<EOF
set -e
mkdir -p .venv/jupyter
python3 -m venv .venv/jupyter
source .venv/jupyter/bin/activate
pip install jupyterlab
EOF

#######################################################################################################
# Go, GoNB and gopjrt
#######################################################################################################
ARG GO_VERSION="1.24.4"
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# Download Go in users home directory.
USER ${NB_USER}
ENV GOPATH /home/${NB_USER}/go-data
ENV PATH ${HOME}/go/bin:${GOPATH}/bin:${PATH}
RUN mkdir ${HOME}/work

WORKDIR ${HOME}
RUN wget --quiet --output-document=- "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -xz \
    && go version

# Download GoNB and its dependencies.
RUN go install github.com/janpfeifer/gonb@latest \
    && go install golang.org/x/tools/cmd/goimports@latest \
    && go install golang.org/x/tools/gopls@latest \
    && gonb --install --alsologtostderr

# Install dlv for debugging.
RUN go install github.com/go-delve/delve/cmd/dlv@latest

# Clone and install gopjrt.
USER ${NB_USER}
WORKDIR ${HOME}
RUN <<EOF
mkdir Projects
cd Projects
git clone https://github.com/gomlx/gopjrt.git
EOF

# Install gopjrt libraries.
USER root
WORKDIR ${HOME}/Projects/gopjrt
RUN ./cmd/install_linux_amd64.sh && ./cmd/install_cuda.sh

# Test gopjrt
USER ${NB_USER}
WORKDIR ${HOME}/Projects/gopjrt
RUN go test ./...


#######################################################################################################
# Install GOMLX libraries and code
#
# As a separate docker target so it can be easily cache busted.
#######################################################################################################
FROM gomlx_jupyterlab_base AS gomlx_jupyterlab

# Clone and test GoMLX
arg GOMLX_REPOSITORY_BRANCH="main"
USER ${NB_USER}
WORKDIR ${HOME}/Projects
RUN <<EOF
git clone -b ${GOMLX_REPOSITORY_BRANCH} https://github.com/gomlx/gomlx.git
ln "./gomlx/examples/tutorial/tutorial.ipynb" "${HOME}/"
ln "./gomlx/examples/tutorial/"*.jpg "${HOME}/"
cd gomlx
go test ./types/... ./graph/... ./ml/... ./backends/...
go build ./examples/... ./models/...

# Use local modules, so they can be edited by the user, if they want.
go work init
go work use . ../gopjrt ./cmd/gomlx_checkpoints

# Install gomlx_checkpoints tool.
go install ./cmd/gomlx_checkpoints && gomlx_checkpoints -help

EOF

#######################################################################################################
# Tests: run at the end, so docker build cache is not busted.
#######################################################################################################

# I have no idea why, but using CUDA/CuDNN with XLA the very first time is very slow. It takes a couple of minutes
# to run. So we run a trivial model once upfront.
#
# The problem is that `docker build` more recently is not run with access to GPU. See issue here:
# https://forums.developer.nvidia.com/t/nvidia-driver-is-not-available-on-latest-docker/246265/2
# https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime/61737404#61737404
#
# So for now, the alternative if to run this after building the image, and then we commit the container.
#RUN go run ./examples/linear

#######################################################################################################
# Finishing touches.
#######################################################################################################

# Clean up space used by apt.
USER root
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Back to home directory.
USER ${NB_USER}
WORKDIR ${HOME}
EXPOSE 8888/tcp
CMD ["bash", "-c", "source .venv/jupyter/bin/activate ; jupyter-lab --no-browser --ip=0.0.0.0"]
ENTRYPOINT []
