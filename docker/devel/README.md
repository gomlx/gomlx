## Devel Docker

This docker is used to compile the C++-Go bridge `libgomlx_xla` library, whose files are 
under `c/` subdirectory.

See details how to use it in [docs/building.md](https://github.com/gomlx/gomlx/blob/main/docs/building.md).

The docker is derived from `tensorflow/tensorflow:build` docker, which is used to build TensorFlow -- and 
it has been working with XLA (XLA used to be included inside TensorFlow). It is also recommended in 
[XLA building guidelines](https://github.com/openxla/xla/blob/main/docs/developer_guide.md).

### Running it

Follow the instructions in [docs/building.md](https://github.com/gomlx/gomlx/blob/main/docs/building.md).


### Building the Docker

It allows customizing the arguments `DEVEL_USER`, `DEVEL_HOME` and `DEVEL_USER_ID` should be set to the user compiling 
GoMLX inside the docker. The recommendation is to set it to match the user used outside. So:

* `DEVEL_USER` can be set to ${USER} (outside Docker).
* `DEVEL_HOME` can be set to the same value of `TEST_TMPDIR` -- here I use `/opt/bazel-cache` -- and one can copy
  over files like `.bashrc` and `.inputrc` for user command line customization.
* `DEVEL_USER_ID` can be set to `id -u`, the id of the current user. This way the owners of the generated files remain
  the same as the user outside.

```bash
docker build -t janpfeifer/gomlx_devel:latest -f docker/devel/Dockerfile .
```

### TODOs

- Create a version without CUDA, to save space for those not using it.
- Create an organization named `gomlx` in [Docker Hub](https://hub.docker.com/) (it costs $9 per month as of 4/2023),
  and use that to store the docker.
