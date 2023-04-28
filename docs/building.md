# Building

### **gomlx_xla** C library building

The **gomlx_xla** library includes [OpenXLA/XLA](https://github.com/openxla/xla/) and some glue Go-C library.

XLA includes lots of things, including the whole LLVM, and it can take quite a while (~30 minutes) to compile.

Compiling, in particular with GPU support has lots dependencies, and the easiest
way is to use TensorFlow devel docker (see
[TensorFlow instructions](https://www.tensorflow.org/install/source#docker_linux_builds))
image `tensorflow/tensorflow:devel` to build it.

To compile-in GPU support use `tensorflow/tensorflow:devel-gpu` instead, and install the
[Nvidia Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).
Probably the most up-to-date instructions will be in
[TensorFlow Docker Build with GPU section](https://www.tensorflow.org/install/source#gpu_support_3).

```bash
# For GPU support use tensorflow/tensorflow:devel-gpu
$  docker pull tensorflow/tensorflow:devel
```

Now we need a few more things in the docker go to the directory you cloned **gomlx**, go to the `c/` subdirectory
(with Bazel's BUILD file) and get a shell in the docker, with the current directory mounted under `/mnt` within
the docker:

```bash
$ cd c/
$ docker run -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:devel bash
```

For GPUs instead:

```bash
$ cd c/
$ docker run --gpus all -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:devel-gpu bash
```

Once inside the docker, we still need to install one extra required library, `tcmalloc`, part of `google-perftools`:

```bash
$ apt install google-perftools libgoogle-perftools-dev:amd64
```

**It also requires the installation [Bazelisk](https://github.com/bazelbuild/bazelisk)**, so it can use
the newest version of [Bazel](https://bazel.build/).

Finally, to train use the sequence bellow. It will create a local `.cache` directory, which will be
visible from outside the docker and set it to be used for compilation (final built files will be there).

```bash
$ mkdir .cache
$ ./bazel.sh --output .cache  # `--gpu --tpu` to add support for GPUs and TPUs. 
```

Experimentally you can set the extra --tpu for TPU compilation. It compiles, but it hasn't been actually
tested yet.

The resulting package file will be inside the docker in `/mnt/bazel-bin/gomlx_xla.tar.gz`, and the library,
if one wants to use it directly (during new ops development), will be in `/mnt/bazel-bin/gomlx/libgomlx_xla.so`.
If you used the `--output .cache` option to `bazel.sh`, these files will be accessible from outside the
docker inside the `gomlx/c/.cache` directory, you'll have to manually translate the `blaze-bin` link
to translate it.

If you are developing on the C++ side, you may want to 
[commit the docker](https://docs.docker.com/engine/reference/commandline/commit/),
to a new image, so you don't need to run the `apt` commands every time you start the docker again.

TODO: create the `docker/devel` Dockerfile with everything needed instead.
