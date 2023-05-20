FROM tensorflow/tensorflow:devel-gpu

LABEL maintainer="pfeifer@gmail.com"
ARG GOMLX_VERSION=v0.2.1
ARG GO_VERSION=1.20.4

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Basic utilities and GoMLX dependencies.
RUN apt-get update --yes \
    && apt-get install -y --no-install-recommends \
        wget vim \
        libunwind8 liblzma5 google-perftools libgoogle-perftools-dev:amd64

# Install Go at root and Bazelisk
ENV GOBIN=/root/go/bin
ENV PATH=${PATH}:${GOBIN}
WORKDIR "/root"
RUN wget --quiet --output-document=- "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"  \
    | tar -xz \
    && go version \
    && go install github.com/bazelbuild/bazelisk@latest

WORKDIR "/"
CMD ["bash"]