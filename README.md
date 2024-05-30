# GoMLX, an Accelerated ML and Math Framework (PyTorch/Jax/Tensorflow-like for Go)


[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gomlx?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gomlx)](https://github.com/Kwynto/gosession/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gomlx)](https://goreportcard.com/report/github.com/gomlx/gomlx)
[![TestStatus](https://github.com/gomlx/gomlx/actions/workflows/go.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/go.yaml)
![Coverage](https://img.shields.io/badge/Coverage-75.2%25-yellow)

GoMLX is a fast and easy-to-use set of Machine Learning and generic math libraries and tools. 
It can be seen as a TensorFlow/Jax/PyTorch for Go.

It uses just-in-time compilation to CPU and GPU (hopefully soon TPUs also) and is built on
top of [OpenXLA](https://github.com/openxla/xla), wich itself uses LLVM to JIT-compile code.
It's the same engine that powers Google's [Jax](https://github.com/google/jax) and 
[TensorFlow](https://tensorflow.org/), and it has the same speed in many cases.

## Quick Start: see our [tutorial](examples/tutorial/tutorial.ipynb), or a [guided example for Kaggle Dogs Vs Cats](examples/dogsvscats/dogsvscats.ipynb).

<div>
<img src="docs/gomlx_gopher.jpg" align="left" alt="GoMLX Gopher" width="160" height="160"/>
<p>It was developed to be full-featured ML platform for Go, and to easily experiment with ML ideas -- see Long-Term Goals below.</p>

It strives to be **simple to read and reason about**, leading the user to a correct and transparent mental model 
of what is going on (no surprises) -- aligned with Go philosophy.
At the cost of more typing (more verbose) at times.

It is also incredibly flexible, and easy to extend and try non-conventional things: use it to experiment with new optimizer ideas, complex regularizers, funky multi-tasking, etc.

Documentation is kept up-to-date (if it is not well documented, it is as if the code is not there)
and error messages are useful and try to make it easy to solve issues.
</div>

**GoMLX is still under development, and should be considered experimental.**

## Overview

**GoMLX** has many important components of an ML framework in place, 
from the bottom to the top of the stack. But it is still only a slice of what a major ML library/framework should provide 
(like TensorFlow, Jax or PyTorch).

It includes:

* Examples: Synthetic linear model; Adult/Census model; Cifar-10 demo; Dogs & Cats classifier demo; IMDB Movie Review 
  demo; Diffusion model for Oxford Flowers 102 dataset (generates random flowers).
* Pre-Trained models to use: InceptionV3 (image model) -- more to come.
* Docker with integrated JupyterLab and GoNB (a Go kernel)
* Just-In-Time (JIT) compilation using [OpenXLA](https://github.com/openxla/xla)] for CPUs and GPUs -- hopefully soon TPUs.
* Autograd: automatic differentiation -- only gradients for now, no jacobian.
* Context: automatic variable management for ML models.
* ML layers library with some of the most popular machine learning "layers": dense (simple FFN layer),  
  activation functions, Layer Normalization, Batch Normalization, Convolutions, Pooling, Dropout, Multi-Head-Attention
  (for transformer layers), PiecewiseLinear (for calibration and normalization).
* Training library, with some pretty-printing. Including plots for Jupyter notebook, using [GoNB, a Go Kernel](https://github.com/janpfeifer/gonb).
  * Also, various debugging tools: collecting values for particular nodes for plotting, simply logging  the value
    of nodes during training, stack-trace of the code where nodes are created (TODO: automatic printing stack-trace
    when a first NaN appears during training).
* SGD and Adam (AdamW and Adamax) optimizers.
* Various losses and metrics.

## Installation

Releases for Linux only, but it's been succesfully compiled in MacOS. 
It does work well also in WSL (Windows Subsystem for Linux) in Windows or using Docker. 

### [Pre-built Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)

The easiest to start playing with it, it's just [pulling the docker image](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)
that includes **GoMLX** + [JupyterLab](https://jupyterlab.readthedocs.io/) + [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter) and 
[Nvidia's CUDA runtime](https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-runtime-ubuntu22.04/images/sha256-08aed54a213b52e9cb658760b6d985db2f4c5f7e8f11ac45ec66b5c746237823?context=explore)
(for optional support of GPU) pre-installed -- it is ~5Gb to download.

From a directory you want to make visible in Jupyter, do:
> For GPU support add the flag `--gpus all` to the `docker run` command bellow.

```bash
docker pull janpfeifer/gomlx_jupyterlab:latest
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jupyter/work janpfeifer/gomlx_jupyterlab:latest
```

It will display a URL starting with `127.0.0.1:8888` in the terminal (it will include a secret token needed) that you can open in your browser.

You can open and interact with the tutorial from there, it is included in the docker under the directory `Projects/gomlx/examples/tutorial`.

More details on the [docker here](docker/jupyterlab/README.md).

### Linux

The library depends on the following libraries to compile and run:

* `libunwind8`: usually available in most Linux systems.
* `liblzma5`: compression library, also usually available.
* TC Malloc, usually packaged as `libgoogle-perftools-dev`: fast malloc version, and memory debugging tools.
  The prebuilt libray shared in the releases `gomlx_xla.tar.gz` also include a build of `libtcmalloc.so`, but
  leaves it in a separate sub-directory under `/usr/local/lib/gomlx/` to avoid conflict. Feel free to use it
  if you want (by copying it to `/usr/local/lib` or adding the directory to `$LD_LIBRARY_PATH`).
* `hdf5-tools`: access to `.h5` file format, used by hold pre-trained weights for some some models. 

Typically, this can be installed with:

```bash
sudo apt-get install libunwind8 libgoogle-perftools-dev liblzma5 hdf5-tools
```

Second you need the pre-compiled GoMLX+XLA C library, included in each release. The library is pretty large,
~500Mb (with GPU and TPU, it statically links most of what it needs) -- for Just-In-Time (JIT)
compilation it includes the whole [LLVM compiler](http://llvm.org). 

Latest version in [github.com/gomlx/gomlx/releases/latest/download/gomlx_xla-linux-amd64.tar.gz](https://github.com/gomlx/gomlx/releases/latest/download/gomlx_xla-linux-amd64.tar.gz).

The contents are a `libgomlx_xla.so` file and a few `.h` files
needed for the compilation of GoMLX. They are separated on two top level directories `/lib` and `/include`, and for
now the recommended way is to just *untar* them in `/usr/local`, which is usually in the default
path for inclusion and dynamic library loading. So you can do:

```bash
cd /usr/local
tar xzvf .../path/to/gomlx_xla-linux-amd64.tar.gz
```

This should be enough for most installations. If [CGO](https://pkg.go.dev/cmd/cgo) is not finding the library,
you may need to configure some environment variables (`LD_LIBRARY_PATH`, `CGO_CPPFLAGS`, `CGO_LDFLAGS`) to include
the corresponding directories under `/usr/local` (most linux distributions won't need this).

After that, just import it as with any Go library.

More on building the C library, see [docs/building.md](docs/building.md).

#### GPU Support (NVidia)

Typically one needs the same NVidia libraries as TensorFlow/Jax. On a fresh 23.04 Ubuntu install, all it took was 
having the commercial NVidia drivers installed (not the Nouveau drivers), and additionally installing:

```bash
apt install nvidia-cudnn
```

After that, another needed step — some misconfiguration among NVidia's CuDNN library, Ubuntu package maintainer and XLA 
code, I'm not sure — is to create the following directory and symbolic link:

```bash
sudo mkdir /usr/lib/nvidia-cuda-toolkit/nvvm
sudo ln -s /usr/lib/nvidia-cuda-toolkit/libdevice /usr/lib/nvidia-cuda-toolkit/nvvm/
```

Without this you'll see errors complaining about not finding `libdevice.10.bc`.

### MacOS

See #23: the required C++ library (`libgomlx_xla.so`) is reported to successfully compile in MacOS. 
It compiles along Google's `libtcmalloc.so`, and one just need to move it to a standard library directory.
Unfortunately, I don't have a mac to build and include these in the releases.

## Tutorial

See the [tutorial here](examples/tutorial/tutorial.ipynb). It covers a bit of everything. 

After that look at the demos in the [examples/](https://github.com/gomlx/gomlx/tree/main/examples) directory.

The library itself is well documented (pls open issues if something is missing), and the code is
not too hard to read (except the bindings to C/XLA, which are tricky). 
Godoc available in [pkg.go.dev](https://pkg.go.dev/github.com/gomlx/gomlx).

Finally, feel free to ask questions: time allowing (when not in work) I'm always happy to help -- I created [groups.google.com/g/gomlx-discuss](https://groups.google.com/g/gomlx-discuss), or use [GitHub discussions page](https://github.com/gomlx/gomlx/discussions).

## Long-term Goals

1. Building and training models in Go -- as opposed to Python (or some other language) -- with focus on:
   - Being simple to read and reason about, leading the user to a correct and transparent mental
     model of what is going on. Even if that means being more verbose when writing.
   - Clean, separable APIs: individual APIs should be self-contained and decoupled where possible.
   - Composability: Any component should be replaceable, so they can be customized and experimented.
     That means sometimes more coding (there is not one magic train object that does everything),
     but it makes it clear what is happening, and it's easy to replace parts with a third party
     versions or something custom.
   - Up-to-date documentation: if the documentation is not there or if it's badly written, it's as 
     if the code was not there either.
   - Clear and actionable error reporting
2. To be a productive research and educational platform to experiment with new ML ideas and learn.
   - Support mirrored training on multiple devices and various forms of distributed training (model and/or data
     parallelism) in particular to support for large language models and similarly large model training.
3. To be a robust and reliable platform for production. Some sub-goals:
   - Support modern accelerator hardware like TPUs and GPUs.
   - Save models to industry tools like [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).
   - Import pre-trained models from
     [Hugging Face Hub](https://huggingface.co/models) and [TensorFlow Hub](https://www.tensorflow.org/hub) where possible.
   - Compile models to binary as in C-libraries and/or WebAssembly, to be linked and consumed (inference) anywhere
     (any language).

## Collaborating

The project is looking forward contributions for anyone interested. Many parts are not yet set 
in stone, so there is plenty of space for improvements and re-designs for those interested
and with good experience in Go, Machine Learning and APIs in general. See the [TODO file](docs/TODO.md)
for inspiration.

No governance guidelines have been established yet.

## Advanced Topics

* [CHANGELOG](docs/CHANGELOG.md)
* [TODO](docs/TODO.md)
* [Error Handling](docs/error_handling.md)
* [Building C/XLA Bindings](docs/building.md)
* [Bare-bones Graph Execution](docs/barebones.md)
* [Developing](docs/developing.md)
