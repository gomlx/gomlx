FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="pfeifer@gmail.com"
ARG GOMLX_VERSION=v0.1.0
ARG GO_VERSION=1.20.3


#######################################################################################################
# User, home directory and basics.
#######################################################################################################

ARG NB_USER="jovyan"
ARG NB_UID=1000
ARG NB_GID=100

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Update apt and install basic utils
RUN apt-get update --yes \
    && apt-get install -y --no-install-recommends wget vim

# Create NB_USER (jovyan) and its home directory.
RUN useradd --create-home --uid ${NB_UID} --gid ${NB_GID} --shell /bin/bash ${NB_USER}
ENV HOME /home/${NB_USER}



#######################################################################################################
# NVidia/CUDA/CuDNN hacks, that the base package didn't install.
#######################################################################################################
USER root

# This creates the missing symlink to /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so.
RUN apt-get install -y --no-install-recommends cuda-nvrtc-dev-11-8

# NVCC: required by XLA to just-in-time compile I suppose.
RUN apt-get install -y --no-install-recommends cuda-nvcc-11-8



#######################################################################################################
# Go and GoMLX Libraries and Code
#######################################################################################################

# Download Go in users home directory.
USER ${NB_USER}
ENV PATH ${HOME}/go/bin:${PATH}
WORKDIR ${HOME}
RUN wget --quiet --output-document=- "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -xz \
    && go version

# Download GoNB and its dependencies.
RUN go install github.com/janpfeifer/gonb@latest \
    && go install golang.org/x/tools/cmd/goimports@latest \
    && go install golang.org/x/tools/gopls@latest \
    && gonb --install --alsologtostderr

# Download and install GoMLX C++ library and its dependencies (tcmalloc, libunwind8 and liblzma5).
USER root

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends gcc libunwind8 liblzma5 libgoogle-perftools-dev

WORKDIR "/usr/local"
RUN wget --quiet --output-document=- "https://github.com/gomlx/gomlx/releases/download/v0.0.1/gomlx_xla-v0.0.1-linux-amd64.tar.gz" | tar -xz
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}



#######################################################################################################
# Jupyter Lab
#######################################################################################################
USER root
RUN apt install --yes --no-install-recommends python3 python3-pip
ENV PATH=${PATH}:${HOME}/.local/bin

USER ${NB_USER}
WORKDIR ${HOME}
RUN pip3 install jupyterlab



#######################################################################################################
# Tests: run at the end, so docker build cache is not busted.
#######################################################################################################

# Copy the GoMLX source code, which includes the examples, that can be interesting to
# look at. Also make a copy of the tutorial at the root directory for quick access.
USER ${NB_UID}
COPY --link --chown=${NB_UID}:${NB_GID} . "${HOME}/Projects/gomlx"

# Run tests -- only on "Host" since GPU in docker build is not working.
WORKDIR "${HOME}/Projects/gomlx"
RUN go test ./...

# I have no idea why, but using CUDA/CuDNN with XLA the very first time is very slow. It takes a couple of minutes
# to run. So we run a trivial model once upfront.
#
# The problem is that `docker build` more recently is not run with access to GPU. See issue here:
# https://forums.developer.nvidia.com/t/nvidia-driver-is-not-available-on-latest-docker/246265/2
# https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime/61737404#61737404
#
# So for now, the alternative if to run this after building the image, and then we commit the container.



#######################################################################################################
# Finishing touches.
#######################################################################################################

# Clean up space used by apt.
USER root
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Back to home directory.
USER ${NB_USER}
WORKDIR "${HOME}"
EXPOSE 8888/tcp
CMD ["jupyter-lab", "--no-browser", "--ip=0.0.0.0"]
ENTRYPOINT []

