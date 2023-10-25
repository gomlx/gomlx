# Building

## **gomlx_xla** C library building

The **gomlx_xla** library includes [OpenXLA/XLA](https://github.com/openxla/xla/) and some glue Go-C library.

XLA includes lots of things, including the whole LLVM, and it can take quite a while (~30 minutes) to compile.

Compiling, in particular with GPU support has lots dependencies, so to facilitated we created docker with the
tools needed, in `janpfeifer/gomlx_devel`, which by default include GPU libraries (but building to support
GPUs is optional). See details in [docker/devel](https://gihub.com/gomlx/gomlx/docker/devel).

```bash
$  docker pull janpfeifer/gomlx_devel:latest
```

You probably want to run it from the `.../gomlx/c` directory, where `.../gomlx` is the directory where you
cloned the GoMLX repository.

The building process used by OpenXLA requires different temporary directories: one for `bazel` another for `bazelisk`.
But since we want to share it with the Docker container, we recommend creating on a separate directory
(e.g: `/opt/bazel-cache/`), mount in the docker container, and configure it in the environment variables 
`TEST_TMPDIR` and `HOME` (will become the home directory for the user inside the docker).

```bash
GOMLX_DOCKER=janpfeifer/gomlx_devel:latest
docker pull ${GOMLX_DOCKER}
docker run -it --gpus all -w /mnt \
  -v "${PWD}":/mnt \
  -v "${TEST_TMPDIR}:${TEST_TMPDIR}" \
  -v "${TEST_TMPDIR}:/.cache" \
  -e "HOME=${TEST_TMPDIR}" \
  -e "TEST_TMPDIR=${TEST_TMPDIR}" \
  -e "USER=${USER}" \
  -u "$(id -u):$(id -g)" \
```

If you have GPU(s) and want to make them accessible you need the
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed to add GPU support for _docker_.

> [!NOTE] Remove the `--gpus all` and `-e "TF_NEED_CUDA=1"` if you don't want GPU support.

Then run **bazel**:

```bash
./bazel.sh --gpu --tpu 
```

The output will be in `bazel-bin/` subdirectory, which in turn is a link inside `$TEST_TMPDIR` that you configured above.

> [!NOTE] Remove `--gpu` or `--tpu` if you don't want support for those. TPU support compiles but hasn't been tested yet.

The resulting package file will be inside the docker in `/mnt/bazel-bin/gomlx_xla.tar.gz`, and the library,
if one wants to use it directly (during new ops development), will be in `/mnt/bazel-bin/gomlx/libgomlx_xla.so`
inside the docker.

Since this will be linked to the value of `$TEST_TMPDIR` you configured above, the symbolic link created in
`.../gomlx/c/bazel-bin` should work outside the docker, and you will be able to see output release file
in `.../gomlx/c/bazel-bin/gomlx_xla.tar.gz`.

## Updating `OpenXLA/XLA` version

There is no official release yet in [github.com/openxla/xla](https://github.com/openxla/xla), so instead we 
build on specific commit hash points. 
Go to the github repository, choose a commit hash, and paste it into
the WORKSPACE file.

Second you will need to manually download the zip file for the version (see `WORKSPACE` file for the URL), and
manually run `sha256` on the zip file to verify its hash code, which should also be pasted into `WORKSPACE`. With
luck the XLA build is not broken and just run the docker and the build command above.

Also, sadly my experience with XLA never builds out-of-the-box after an update. There are always many issues that take days to
fix :(

The steps I followed recently are:

### Creating / Updating the Devel Docker

Instructions in `.../gomlx/docker/devel/README.md`.
It's based on `tensorflow/devel-gpu` (to include GPU support), per XLA suggestion, but requires lots of updating.
The `Dockerfile` is always updated with what worked for the last release. 
Usually, it requires changes every time I update XLA. 

### Generating new `.../gomlx/c/tf_configure.bazelrc` file

I start by following their [building guidelines](https://github.com/openxla/xla/blob/main/docs/developer_guide.md),
then I run from the XLA root the `./configure` and copy over the generated  `.tf_configure.bazelrc` file
to the `.../gomlx/c/tf_configure.bazelrc` file. 

For the XLA configuration I used the following values (env variables used by the
`./configure` script): 

```
# Configuration:
# . See capabilities in 
export TF_NEED_CUDA=1
export CUDA_VERSION=11.8
export TF_CUDA_VERSION=${CUDA_VERSION}
export USE_DEFAULT_PYTHON_LIB_PATH=1
export PYTHON_BIN_PATH=/usr/bin/python3
export TF_NEED_ROCM=0
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_CUDA_COMPUTE_CAPABILITIES="6.1,9.0"
export CC_OPT_FLAGS=-Wno-sign-compare
```

> [!NOTE] Of notice:
> * clang was not working last time I tried -- something related to Bazel requiring BUILD rules.
> * ROCM not compatible with CUDA, according to some error messages.
> * "Compute Capabilities" maps to the supported NVidia hardware: see https://developer.nvidia.com/cuda-gpus

Careful, their instructions have been for long broken and it requires several updates 
to the `tensorflow:devel-gpu` docker to work.
These updates are incorporated into the `.../gomlx/docker/devel/Dockerfile`.

I try to build XLA while I'm at it, just to see if things are compiling.
Even if the full build of XLA fails, it doesn't mean that only the subset needed by GoMLX will fail
(this was the case on the 10/2023 release), but it's a good test.

Notice that for this step I used the `root` user in the docker -- as opposed to the current user used 
to build GoMLX C bindings.

### Building `gomlx.tar.gz`

Follow the building instructions above, on the same docker image you updated above.

Notice the change in users: to build XLA I used the `root` user (suggested by XLA documentation)
in the docker, while to build GoMLX C bindings we use the current user.



