# Building

## **gomlx_xla** C library building

The **gomlx_xla** library includes [OpenXLA/XLA](https://github.com/openxla/xla/) and some glue Go-C library.

XLA includes lots of things, including the whole LLVM, and it can take quite a while (~30 minutes) to compile.

Compiling, in particular with GPU support has lots dependencies, so to faciliated we created docker with the
tools needed, in `janpfeifer/gomlx_devel`, which by default include GPU libraries (but building to support
GPUs is optional). See details in [docker/devel](https://gihub.com/gomlx/gomlx/docker/devel).

```bash
$  docker pull janpfeifer/gomlx_devel:latest
```

Then go to the `c/` subdirectory, which will be mapped to `/mnt` in the docker, and run:
(remove the `--gpus all` if you don't have a GPU available)

```bash
docker run --gpus all -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" janpfeifer/gomlx_devel:devel bash
```

Finally, to build use the commands bellow. It will create a local `.cache` directory, which will be
visible from outside the docker and set it to be used for compilation (final built files will be there).

```bash
mkdir .cache
./bazel.sh --output .cache --gpu --tpu 
```

> **Note**:
> - Remove `--gpu` or `--tpu` if you don't want support for those. TPU support compiles but hasn't been tested yet.

The resulting package file will be inside the docker in `/mnt/bazel-bin/gomlx_xla.tar.gz`, and the library,
if one wants to use it directly (during new ops development), will be in `/mnt/bazel-bin/gomlx/libgomlx_xla.so`
inside the docker.

If you used the `--output .cache` option to `bazel.sh`, these files will be also accessible from outside the
docker inside the `gomlx/c/.cache` directory. You'll have to manually translate the `bazel-bin` link
to the access the library from outside the docker.

## Updating `OpenXLA/XLA` version

There is no official release yet in [github.com/openxla/xla](https://github.com/openxla/xla), so instead we 
build on specific commit hash points. Go to the github repository, choose a commit hash, and paste it into
the WORKSPACE file.

Second you will need to manually download the zip file for the version (see `WORKSPACE` file for the URL), and
manually run `sha256` on the zip file to verify its hash code, which should also be pasted into `WORKSPACE`. With
luck the XLA build is not broken and just run the docker and the build command above.
