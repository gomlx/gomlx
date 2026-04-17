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
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends sudo wget vim micro unzip hdf5-tools libc6-dev curl gcc g++ git openssh-client

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
# Go, GoNB and go-xla
#######################################################################################################
ARG GO_VERSION="1.26.2"
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
    && gonb --install --alsologtostderr -go_work "${HOME}/Projects/gomlx,${HOME}/Projects/go-huggingface,${HOME}/Projects/onnx-gomlx,${HOME}/Projects/go-xla"

# Install dlv for debugging.
RUN go install github.com/go-delve/delve/cmd/dlv@latest

#######################################################################################################
# Install go-xla: this include NVidia libraries, and hence in a separate sub-docker.
#######################################################################################################
FROM gomlx_jupyterlab_base AS gomlx_jupyterlab_goxla

# Clone and install gopjrt_installer.
USER ${NB_USER}
WORKDIR ${HOME}
RUN <<EOF
mkdir Projects
cd Projects
git clone https://github.com/gomlx/go-xla.git
cd go-xla/cmd/pjrt_installer
go install .
EOF

# Install cpu and cuda (13) PJRT plugins in /usr/local (requires root).
RUN <<EOF
$(go env GOPATH)/bin/pjrt_installer -plugin linux -version latest -path ${HOME}/.local/lib/go-xla
$(go env GOPATH)/bin/pjrt_installer -plugin cuda13 -version latest -path ${HOME}/.local/lib/go-xla
EOF

# Test go-xla
RUN <<EOF
cd Projects/go-xla
go test
go test ./...
EOF

#######################################################################################################
# Install GOMLX libraries and code
#
# As a separate docker target so it can be easily cache busted.
#######################################################################################################
FROM gomlx_jupyterlab_goxla AS gomlx_jupyterlab

# Clone and test GoMLX
ARG GOMLX_REPOSITORY_BRANCH="main"
USER ${NB_USER}
WORKDIR ${HOME}
RUN <<EOF
cd Projects
git clone -b ${GOMLX_REPOSITORY_BRANCH} https://github.com/gomlx/gomlx.git
git clone https://github.com/gomlx/onnx-gomlx.git
git clone https://github.com/gomlx/go-huggingface.git

# GoMLX setup: use local modules, so they can be edited by the user, if they want.
ln "./gomlx/examples/tutorial/tutorial.ipynb" "${HOME}/"
ln "./gomlx/examples/tutorial/"*.jpg "${HOME}/"
cd gomlx
go work init && go work use . ./examples ./cmd/gomlx_checkpoints ../onnx-gomlx ../go-huggingface ../go-xla
go install ./cmd/gomlx_checkpoints && gomlx_checkpoints -help
go test ./types/... ./graph/... ./ml/... ./backends/...
go build ./examples/... ./models/...

# ONNX-GoMLX and go-huggingface setup.
cd ../go-huggingface
go work init && go work use . ../gomlx

cd ../onnx-gomlx
go work init && go work use . ../gomlx ../go-huggingface

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
RUN cd Projects/gomlx && go run ./examples/linear

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
