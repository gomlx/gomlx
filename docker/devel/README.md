## Devel Docker

This docker is used to compile the C++-Go bridge `libgomlx_xla` library, whose files are 
under `c/` subdirectory.

See details how to use it in [docs/building.md](https://github.com/gomlx/gomlx/blob/main/docs/building.md).

The docker is derived from `tensorflow/tensorflow:devel-gpu` docker, which is used to build TensorFlow -- and 
it has been working with XLA (XLA used to be included inside TensorFlow).

### Running it

You probably want to run it from the `gomlx/c` directory, where `gomlx` is the directory where you
cloned the GoMLX repository.

```bash
docker pull janpfeifer/gomlx_devel:latest
docker run -it -w /mnt -v "${PWD}":/mnt -e HOST_PERMS="$(id -u):$(id -g)" janpfeifer/gomlx_devel:latest
```

If you have GPU(s) and want to make them accessible (you need the 
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed to add GPU support for _docker_) use instead:

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

