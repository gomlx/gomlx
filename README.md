# GoMLX -- Accelerated ML Libraries for Go

GoMLX is a fast and easy-to-use set of ML libraries built on top of [OpenXLA](https://github.com/openxla/xla),
a just-in-time compiler of numeric computations to CPU, GPU and TPU.

## Quick Start: see our [tutorial](examples/tutorial/tutorial.ipynb), or a [guided example for Kaggle Dogs Vs Cats using GoMLX](examples/dogsvscats/dogsvscats.ipynb).

**This under development, and should be considered experimental for now.**

It was developed primarily as a platform to easily experiment with ML ideas, and to
allow one to use Go for ML. Hopefully it can grow beyond that -- see Long-Term Goals below.

It strives to be **simple to read and reason about**, leading the 
user to correct and transparent mental model of what is going on (no surprises). At the cost of
more typing (more verbose) at times.

Documentation is kept up-to-date (if it is not well documented, it is as if the code is not there)
and error messages are useful and try to make it easy to solve issues (a good guideline is described in
Section 2 of [Yggdrasil Decision Forests paper](https://arxiv.org/pdf/2212.02934.pdf)). 

![GoMLX Gopher](docs/gomlx_gopher.jpg)

## Overview

**GoMLX** is a working proof-of-concept, with many important components of an ML framework in place, 
from the bottom to the top of the stack. But only still a narrow slice of what a major ML library/framework should provide 
(like TensorFlow, Jax or PyTorch).

It includes:

* XLA integration for model training and evaluation -- including GPU (and presumably TPU, but never tested so likely 
  not working).
* Autograd: automatic differentiation -- only gradients for now, no Jacobians.
* Context: automatic variable management for ML models.
* ML layers library with some of the most popular machine learning "layers": dense (simple FFN layer),  
  activation functions, Layer Normalization, Batch Normalization, Convolutions, Pooling, Dropout, Multi-Head-Attention
  (for transformer layers), PiecewiseLinear (for calibration and normalization).
* Training library, with some pretty-printing. Including plots for Jupyter notebook, using [Bash Kernel](https://github.com/takluyver/bash_kernel).
  * Also, various debugging tools: collecting values for particular nodes for plotting, simply logging  the value
    of nodes during training, stack-trace of the code where nodes are created (TODO: automatic printing stack-trace
    when a first NaN appears during training).
* SGD and Adam optimizers.
* Various losses and metrics.
* Examples: Synthetic linear model; Adult/Census model; Cifar-10 demo; Dogs & Cats classifier demo; IMDB Movie Review 
  demo. 
* Docker with integrated JupyterLab and GoNB (a Go kernel)

## Installation

For now Linux only. It does work well also in WSL (Windows Subsystem for Linux) in Windows or using Docker. 

Likely it would work in Macs with some work --> contributions are very welcome, I don't have a Mac. It will likely work in Docker, but not natively supporting M1/M2.

### Pre-built docker

The easiest to start playing with it, it's just pulling the docker image that has GoMLX + [JupyterLab](https://jupyterlab.readthedocs.io/) + [GoNB](https://github.com/janpfeifer/gonb)(a Go kernel) pre-installed.

From a directory you want to make visible in Jupyter, do:
> For GPU support add the flag `--gpus all` to the `docker run` command bellow.

```bash
docker pull janpfeifer/gomlx_jupyterlab
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/gomlx/work janpfeifer/gomlx_jupyterlab:latest
```

And then open your browser in the `localhost:8888` link it will display in the termianl -- it will include a secret token needed.

You can open and interact with the tutorial from there, it is included in the docker under the directory `Projects/gomlx/examples/tutorial`.

More details on the [docker here](docker/jupyterlab/README.md).

### Linux

The library depends on the following libraries to compile and run:

* `libunwind8`: usually available in most Linux systems.
* `liblzma5`: compression library, also usually available.
* TC Malloc, usually packaged as `libgoogle-perftools-dev`: fast malloc version, and memory debugging tools.

Typically, this can be installed with:

```bash
sudo apt-get install libunwind8 libgoogle-perftools-dev liblzma5
```

Second you need the pre-compiled GoMLX+XLA C library, included in each release. The library is pretty large,
~500Mb (with GPU and TPU, it statically links most of what it needs) -- for Just-In-Time (JIT)
compilation it includes the whole [LLVM compiler](http://llvm.org). 

The contents are a `libgomlx_xla.so` file and a few `.h` files
needed for the compilation of GoMLX. They are separated on two top level directories `/lib` and `/include`, and for
now the recommended way is to just *untar* them in `/usr/local`, which is usually in the default
path for inclusion and dynamic library loading. For instance:

```bash
cd /usr/local
tar xzvf .../path/to/gomlx_xla-v0.0.1.tar.gz
```

Latest version in [github.com/gomlx/gomlx/releases/download/v0.0.1/gomlx_xla-v0.0.1-linux-amd64.tar.gz](https://github.com/gomlx/gomlx/releases/download/v0.0.1/gomlx_xla-v0.0.1-linux-amd64.tar.gz).

This should be enough for most installations. If [CGO](https://pkg.go.dev/cmd/cgo) is not finding the library,
you may need to configure some environment variables (`LD_LIBRARY_PATH`, `CGO_CPPFLAGS`, `CGO_LDFLAGS`) to include
the corresponding directories under `/usr/local` (most linux distributions won't need this).

After that, just import it as with any Go library.

More on building the C library, see [docs/building.md](docs/building.md).

## Tutorial

See the [tutorial here](examples/tutorial/tutorial.ipynb). It covers a bit of everything. 

After that look at the demos in the [examples/](examples/) directory.

The library itself is well documented (pls open issues if something is missing), and the code is
not too hard to read (except the bindings to C/XLA, which were done very adhoc). Godoc available in [pkg.go.dev](https://pkg.go.dev/github.com/gomlx/gomlx).

Finally, feel free to ask questions: time allowing (when not in work) I'm always happy to help -- I created [groups.google.com/g/gomlx-discuss](https://groups.google.com/g/gomlx-discuss).

## Long-term Goals_

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
   - Clear and actuable error reporting: 
1. To be a robust and reliable platform for production. Some sub-goals:
   - Support modern accelerator hardware like TPUs and GPUs.
   - Save models to industry tools like [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).
   - Import pre-trained models from
     [Hugging Face Hub](https://huggingface.co/models) and [TensorFlow Hub](https://www.tensorflow.org/hub).
   - Compile models to binary as in C-libraries and/or WebAssembly, to be linked and consumed (inference) anywhere
     (any language).
1. To be a productive research and educational platform to learn and try new ML ideas:
   - As composable and decoupled as possible, to allow anything to be tweaked and replaced without much hassle.
   - Simple to read and well documented for anyone wanting to see how things are done. 
   - Support mirrored training on multiple devices and various forms of distributed training (model and/or data
     parallelism) in particular to support for large language models and similarly large model training.

## Collaborating

The project is looking forward contributions for anyone interested. Many parts are not yet set 
in stone, so there is plenty of space for improvements and re-designs for those interested
and with good experience in Go, Machine Learning and APIs in general.

No governance guidelines have been established yet, this also needs work.

## Advanced Topics

* [CHANGELOG](docs/CHANGELOG.md)
* [TODO](docs/TODO.md)
* [Error Handling](docs/error_handling.md)
* [Building C/XLA Bindings](docs/building.md)
* [Barebones Graph Execution](docs/barebones.md)
* [Developing](docs/developing.md)
