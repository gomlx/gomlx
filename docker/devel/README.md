## Devel Docker

This docker is used to compile the C++-Go bridge `libgomlx_xla` library, whose files are 
under `c/` subdirectory.

See details how to use it in [docs/building.md](https://github.com/gomlx/gomlx/blob/main/docs/building.md).

The docker is derived from `tensorflow/tensorflow:devel-gpu` docker, which is used to build TensorFlow -- and 
it has been working with XLA (XLA used to be included inside TensorFlow).
It does require the extra package `libnvidia-ml-dev:amd64` to be installed though -- maybe `tensorflow:devel_gpu`
is getting abandoned (it hasn't been updated in 10 months as of this writing) ?

### Running it

You probably want to run it from the `.../gomlx/c` directory, where `.../gomlx` is the directory where you
cloned the GoMLX repository.

The building process used by OpenXLA requires different temporary directories: one for `bazel` another for `bazelisk`.
But since we want to share it with the Docker container, we recommend creating on a separate directory
(e.g: `/opt/bazel-cache/`), mount in the docker container, and configure it in the environment variables
`TEST_TMPDIR` and `HOME` (will become the home directory for the user inside the docker).

If you have GPU(s) and want to make them accessible you need the
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed to add GPU support for _docker_.

```bash
GOMLX_DOCKER=janpfeifer/gomlx_devel:latest
docker pull ${GOMLX_DOCKER}
export TEST_TMPDIR="/opt/bazel-cache"   # Or set your own
docker run -it --gpus all -w /mnt \
  -v "${PWD}":/mnt \
  -v "${TEST_TMPDIR}:${TEST_TMPDIR}" \
  -v "${TEST_TMPDIR}:/.cache" \
  -e "HOME=${TEST_TMPDIR}" \
  -e "TEST_TMPDIR=${TEST_TMPDIR}" \
  -e "USER=${USER}" \
  -u "$(id -u):$(id -g)" \
  ${GOMLX_DOCKER}
```

> [!NOTE] Remove the `--gpus all` if you don't have a GPU available

```bash
docker run -it --gpus all -w /mnt -v "${PWD}":/mnt -e HOST_PERMS="$(id -u):$(id -g)" janpfeifer/gomlx_devel:latest
```

### Building the Docker

```bash
docker build -t janpfeifer/gomlx_devel:latest -f docker/devel/Dockerfile .
```

### TODOs

- Create a version without CUDA, to save space for those not using it.
- Create an organization named `gomlx` in [Docker Hub](https://hub.docker.com/) (it costs $9 per month as of 4/2023),
  and use that to store the docker -- it shouldn't be located on a personal namespace.
