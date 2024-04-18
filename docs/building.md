# Building

## **gomlx_xla** C library building

The **gomlx_xla** library includes [OpenXLA/XLA](https://github.com/openxla/xla/) and some glue Go-C library.

XLA includes lots of things, including the whole LLVM, and it can take quite a while (~30 minutes) to compile.

Compiling, in particular with GPU support has lots dependencies, so to facilitated we created docker with the
tools needed, in `janpfeifer/gomlx_devel`, which by default include GPU libraries (but building to support
GPUs is optional). 

You can download the pre-built `gomlx_devel` docker with `docker pull janpfeifer/gomlx_devel:latest`
or optionally built it your self and configure the user to be used. There are instructions on
[docker/devel/README.md](https://github.com/gomlx/gomlx/blob/main/docker/devel/README.md) 
on how to build the docker, if preferred.

Since we want the generated files (we use `bazel` build system) among the host and the _docker_ container, we recommend
creating on a separate directory (default to `/opt/bazel-cache/`), had it mounted in the docker container, and
configure it in the environment variables `TEST_TMPDIR` and `HOME` (will become the home directory for the user 
inside the docker).

Notice ou probably want to run the docker from the `.../gomlx/c` directory, where `.../gomlx` is the directory where you
cloned the GoMLX repository.

Running the docker:

```bash
export TEST_TMPDIR="/opt/bazel-cache"   # Or set your own, but it should match the DEVEL_HOME in the `Dockerfile`..
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

If you have GPU(s) and want to make them accessible you need the
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed to add GPU support for _docker_.

> [!NOTE] Remove the `--gpus all` if you don't want GPU support.

Then run **bazel**:

```bash
./bazel.sh --gpu
```

> [!WARNING] Currently, there is a bug that require a manual hack to get NVidia's library NCCL to be linked
>   statically. See: https://github.com/openxla/xla/issues/11604. The short-term solution
>   is manually editing XLA code (after it is pulled by bazel), and edit the 2 lines in the rule as pointed out in the
>   bug. Hopefully, it will be fixed later.

The output will be in `bazel-bin/` subdirectory, which in turn is a link inside `$TEST_TMPDIR` that you configured above.

> [!NOTE] Remove `--gpu` and regenerated `xla_configure.bazelrc` (see below)  if you don't want support for GPU.

> [!NOTE] TPUs are not supported, since they are no longer is OSS :(. See https://github.com/openxla/xla/issues/11599

The resulting package file will be inside the docker in `.../bazel-bin/gomlx_xla.tar.gz`, and the library,
if one wants to use it directly (during new ops development), will be in `.../bazel-bin/gomlx/libgomlx_xla.so`.
If you used the same running instructions (the linked `$TEST_TMPDIR`) it will be visible both inside the
docker under `/mnt` and in the host under `.../gomlx/c`.

By default, the library is built statically including its dependencies, so it will work both in environments with
a GPU and in environments without a GPU.

## Updating `OpenXLA/XLA` version

There is no official release yet in [github.com/openxla/xla](https://github.com/openxla/xla), so instead we 
build on specific commit hash points. 
Go to the github repository, choose a commit hash, and paste it into
the WORKSPACE file.

Second you will need to manually download the zip file for the version (see `WORKSPACE` file for the URL), and
manually run `sha256` on the zip file to verify its hash code, which should also be pasted into `WORKSPACE`. With
luck the XLA build is not broken and just run the docker and the build command above.

Also, sadly my experience with XLA never builds out-of-the-box after an update. There are always issues that take days
to fix :(

The steps I followed recently are:

### Creating / Updating the Devel Docker

Instructions in `.../gomlx/docker/devel/README.md`.
It's based on `tensorflow/build` (to include GPU support), per XLA suggestion, but requires lots of updating.
The `Dockerfile` is always updated with what worked for the last release. 
Usually, it requires changes every time I update XLA. 

### Generating new `.../gomlx/c/xla_configure.bazelrc` file

This is needed if one is updating the XLA version.

First I clone the XLA repository **inside the devel docker**. 

Then following XLA's [building guidelines](https://github.com/openxla/xla/blob/main/docs/developer_guide.md),
then from XLA's root directory I run `python configure.py` and copy over the generated `xla_configure.bazelrc` file
to the `.../gomlx/c/xla_configure.bazelrc` file. 

For the XLA configuration I used the following values (env variables used by the
`./configure` script): 

```
# Configuration:
python configure.py --backend=CUDA --cuda_compute_capabilities="sm_60,sm_70,sm_80,sm_89,compute_90" --nccl
```

> [!NOTE] Of notice:
> * "Compute Capabilities" maps to the supported NVidia hardware, we default to supporting Tesla P100 to H100 (inc. Quadro P6000+ and GeForce GTX1080+ but not GTX TITAN).
>   See https://developer.nvidia.com/cuda-gpus#compute. 
>   * `compute_XY` enables PTX embedding in addition to SASS. PTX is forward compatible beyond the current compute  
>     capability major release while SASS is only forward compatible inside the current major release. Example: 
>     sm_80 kernels can run on sm_89 GPUs but not on sm_90 GPUs. compute_80 kernels though can also run on sm_90 GPUs.
> * NCCL (library to work with multi-GPUs) is needed, see https://github.com/openxla/xla/issues/11596
> * NCCL needs to be compiled statically, see: https://github.com/openxla/xla/issues/11604. The short-term solution
>   is manually editing XLA code (after it is pulled by bazel), and edit the 2 lines in the rule as pointed out in the
>   bug. Hopefully, it will be fixed.
> * ROCM not compatible with CUDA, according to some error messages.

I try to build XLA while I'm at it, just to see if things are compiling.
Even if the full build of XLA fails, it doesn't mean that only the subset needed by GoMLX will fail
(this was the case on the 10/2023 release), but it's a good test.


