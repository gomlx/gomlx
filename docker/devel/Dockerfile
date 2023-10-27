FROM tensorflow/tensorflow:devel-gpu

LABEL maintainer="pfeifer@gmail.com"
ARG GO_VERSION=1.21.3
ARG CUDA_VERSION=11.8
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

# For some unfathomable reason CUDA/XLA wants nvml.h somewhere else.
cuda_version_pkg=${CUDA_VERSION/./-}
dpkg --get-selections | grep -v deinstall | grep 11-2 | grep -v cuda-memcheck \
       | cut -f 1 | sed -e "s/11-2/${cuda_version_pkg}/g;" | xargs apt install --yes

# File required to be present in CUDA directory for some reason.
cp /usr/include/nvml.h /usr/local/cuda-${CUDA_VERSION}/include/
EOT


# Install Go at root and Bazelisk.
ENV GOBIN=/root/go/bin
ENV PATH=${PATH}:${GOBIN}
WORKDIR "/root"
RUN wget --quiet --output-document=- "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"  \
    | tar -xz \
    && go version \
    && go install github.com/bazelbuild/bazelisk@latest

# Fetch XLA source: probably better done manually when updating XLA.
#RUN <<EOT
#  git clone https://github.com/openxla/xla.git
#EOT

# It is a downgrade, but as of 10/2023, this is the cudnn version that seems to compile.
RUN apt-get install --yes --allow-downgrades \
    libcudnn8-dev=8.5.0.96-1+cuda11.7 libcudnn8=8.5.0.96-1+cuda11.7

# Convenient terminal editor.
RUN apt-get install --yes micro

# Needed because they include NVidia TensorRT, even though they are for CUDA 12 ...
RUN apt-get install --yes libnvinfer8 libnvinfer-dev

#######################################################################################################
# Finishing touches.
#######################################################################################################

# Clean up space used by apt.
USER root
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Default is to start a shell as root.
WORKDIR "/root"
CMD ["bash"]