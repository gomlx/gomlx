# See accompanying README.md for instructions on how to use it.
FROM tensorflow/build:latest-python3.12
# FROM tensorflow/tensorflow:devel-gpu

# Define work user. DEVEL_HOME should have the same value as TEST_TMPDIR used when running the Docker (see README.md)
# DEVEL_USER_ID will be the owner of the
ARG DEVEL_USER=xla
ARG DEVEL_HOME=/opt/bazel-cache
ARG DEVEL_USER_ID=1000

LABEL maintainer="pfeifer@gmail.com"
ARG GO_VERSION=1.22.2

# ARG CUDA_VERSION=11.8
ARG CUDA_VERSION=12.3

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# We will run everything as root.
USER root
ENV CUDA_VERSION=$CUDA_VERSION

# Basic utilities and GoMLX dependencies.
# The base tensorflow:devel-gpu docker is out-of-sync with the recent requirements of XLA,
# so we need to do `apt-get upgrade` -- in particular for NVIDIA GPU dependencies.
#
# Also, it's not documented, but XLA needs CUDA > 11.6. Here we replace all cuda 11.2 by
# cuda 11.8, the latest available in a normal Ubuntu desktop (more common target).
# The code is very brittle (s/11-2/11-8), but works currently.
#
# We are not using CUDA 12.X yet, because they are not easily available on folks system
# (Ubuntu 23.04 offers CUDA 11.8 only), and we don't want to force folks to go to NVIDIA
# and figure out painstakingly how to install these things.
RUN <<EOT
apt-get update --yes \
    && apt-get upgrade --yes \
    && apt-get install -y --no-install-recommends \
        clang libnvidia-ml-dev:amd64 \
        wget vim \
        libunwind8 liblzma5 google-perftools libgoogle-perftools-dev:amd64 \
EOT


# Install Go at root and Bazelisk.
ENV GOBIN=/root/go/bin
ENV PATH=${PATH}:${GOBIN}
WORKDIR "/root"
RUN wget --quiet --output-document=- "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"  \
    | tar -xz \
    && go version \
    && go install github.com/bazelbuild/bazelisk@latest

# Convenient terminal editor.
RUN apt-get install --yes micro

# Needed because they include NVidia TensorRT.
RUN apt-get install --yes libnvinfer8 libnvinfer-dev

#######################################################################################################
# Finishing touches.
#######################################################################################################

# Clean up space used by apt.
USER root
RUN <<EOT
apt-get clean && rm -rf /var/lib/apt/lists/*
EOT

# Setup user to use to build GoMLX+XLA libraries.
USER root
ENV DEVEL_USER=$DEVEL_USER
ENV DEVEL_HOME=$DEVEL_HOME
ENV DEVEL_USER_ID=$DEVEL_USER_ID

RUN <<EOT
groupadd -g "${DEVEL_USER_ID}" "${DEVEL_USER}"
useradd -u "${DEVEL_USER_ID}" -g "${DEVEL_USER}" -m -d "${DEVEL_HOME}" -s /bin/bash xla
echo "${DEVEL_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
echo "done"
EOT

# Default is to start a shell as root.
WORKDIR "/root"
CMD ["bash"]