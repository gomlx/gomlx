# **_GoMLX_**, an Accelerated ML and Math Framework

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gomlx?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gomlx)](https://github.com/Kwynto/gosession/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gomlx)](https://goreportcard.com/report/github.com/gomlx/gomlx)
[![TestStatus](https://github.com/gomlx/gomlx/actions/workflows/go.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/go.yaml)
![Coverage](https://img.shields.io/badge/Coverage-69.4%25-yellow)


## 📖 About **_GoMLX_**
<img align="right" src="docs/gomlx_gopher.jpg" alt="GoMLX Gopher" width="220px"/>

**GoMLX** is a fast and easy-to-use set of Machine Learning and generic math libraries and tools. 
It can be seen as a **PyTorch/Jax/TensorFlow for Go**.

It can be used to train, fine-tune, modify and combine machine learning models. It provides all
the tools to make that work easy: from a complete set of differentiable operators, all the way to UI
tools to plot metrics while training in a notebook.

It's main backend engine based on [OpenXLA/PJRT](https://github.com/openxla/xla) uses just-in-time
compilation to CPU and GPU (optionally TPUs also). 
It's the same engine that powers Google's [Jax](https://github.com/google/jax) and 
[TensorFlow](https://tensorflow.org/), and it has the same speed in many cases.

> [!Tip]
> 🎓 Quick Start:
> * See our [**tutorial**](https://gomlx.github.io/gomlx/notebooks/tutorial.html)
> * See _Eli Bendersky_'s blog post ["GoMLX: ML in Go without Python"](https://eli.thegreenplace.net/2024/gomlx-ml-in-go-without-python/)
> * A [guided example for Kaggle Dogs Vs Cats](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html).
> * [Installation here](#installation).

<div>
<p>It was developed to be full-featured ML platform for Go, and to easily experiment with ML ideas -- see Long-Term Goals below.</p>

It strives to be **simple to read and reason about**, leading the user to a correct and transparent mental model 
of what is going on (no surprises) -- aligned with Go philosophy.
At the cost of more typing (more verbose) at times.

It is also incredibly flexible, and easy to extend and try non-conventional ideas: use it to experiment with new optimizer ideas, complex regularizers, funky multi-tasking, etc.

Documentation is kept up-to-date (if it is not well documented, it is as if the code is not there)
and error messages are useful (always with a stack-trace) and try to make it easy to solve issues.
</div>

**GoMLX is still evolving, and should be considered experimental.** There are occasional changes, but mostly it has just been
adding new functionality.

## 🗺️ Overview

**GoMLX** has many important components of an ML framework in place, 
from the bottom to the top of the stack. But it is still only a slice of what a major ML library/framework should provide 
(like TensorFlow, Jax or PyTorch).


**Examples:**

  * [Adult/Census model](https://gomlx.github.io/gomlx/notebooks/uci-adult.html);
  * [How KANs learn ?](https://gomlx.github.io/gomlx/notebooks/kan_shapes.html); 
  * [Cifar-10 demo](https://gomlx.github.io/gomlx/notebooks/cifar.html); 
  * [Dogs & Cats classifier demo](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html); 
  * [IMDB Movie Review demo](https://gomlx.github.io/gomlx/notebooks/imdb.html); 
  * [Diffusion model for Oxford Flowers 102 dataset (generates random flowers)](examples/oxfordflowers102/OxfordFlowers102_Diffusion.ipynb);
    * 🎉**NEW**🎉 [Flow Matching Study Notebook](https://github.com/gomlx/gomlx/blob/main/examples/FlowMatching/flow_matching.ipynb) based on Meta's ["Flow Matching Guide and Code"](https://ai.meta.com/research/publications/flow-matching-guide-and-code/).
  * [GoMLX/Gemma](https://github.com/gomlx/gemma), a **GoMLX** implementation of 
    [Google DeepMind's Gemma v2 model](https://github.com/google-deepmind/gemma) ([blog post](https://ai.google.dev/gemma))
  * [GNN model for OGBN-MAG (experimental)](examples/ogbnmag/ogbn-mag.ipynb).
  * Last, a trivial [synthetic linear model](https://github.com/gomlx/gomlx/blob/main/examples/linear/linear.go), for those curious to see a barebones simple model.

> **🚀 NEW 🚀**: 
>    - Converting ONNX models to GoMLX with [onnx-gomlx](https://github.com/gomlx/onnx-gomlx): both as an alternative for `onnxruntime` (leveraging XLA), but also to further fine-tune models. See also [go-huggingface](https://github.com/gomlx/go-huggingface) to easily download ONNX model files from HuggingFace.
>    - Support static linking of PJRT: slower to build the Go program, but deploying it doesn't require installing a PJRT plugin in the machine you are deploying it. 
>      Use `go build --tags=pjrt_cpu_static` or include `import _ "github.com/gomlx/gomlx/backends/xla/cpu/static"`. 
>    - Experimental 🚧🛠 support for MacOS (both Arm64 and the older x86_64) for CPU: only with static linking so far.

**Highlights:**

* Pre-Trained models to use: InceptionV3 (image model) -- more to come.
* [Docker "gomlx_jupyterlab"](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab) with integrated JupyterLab and [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter notebooks)
* Just-In-Time (JIT) compilation using [OpenXLA](https://github.com/openxla/xla) for CPUs and GPUs -- hopefully soon TPUs.
* Autograd: automatic differentiation -- only gradients for now, no jacobian.
* Context: automatic variable management for ML models.
* ML layers library with some of the most popular machine learning "layers": FFN layers,  
  activation functions, layer and batch normalization, convolutions, pooling, dropout, Multi-Head-Attention
  (for transformer layers), LSTM, KAN (B-Splines, [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594), Discrete-KAN, PiecewiseLinear KAN),
  PiecewiseLinear (for calibration and normalization), various regularizations,
  FFT (reverse/differentiable), learnable rational functions (both for activations and [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594)) etc. 
* Training library, with some pretty-printing. Including plots for Jupyter notebook, using [GoNB, a Go Kernel](https://github.com/janpfeifer/gonb).
  * Also, various debugging tools: collecting values for particular nodes for plotting, simply logging  the value
    of nodes during training, stack-trace of the code where nodes are created.
* SGD and Adam (AdamW and Adamax) optimizers.
* Various losses and metrics.

## 👥 Support

* [Q&A and discussions](https://github.com/gomlx/gomlx/discussions/categories/q-a)
* [Issues](https://github.com/gomlx/gomlx/issues)
* Random brainstorming on projects: just start a Q&A and I'm happy to meet in discord somewhere or VC.

## <a id="installation"></a>🛠️ + ⚙️ Installation

**TLDR;**: Two simple options:

(1) [Use the Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab);

(2) Use pre-built binaries (C/C++ libraries) for Linux or MacOS (Darwin). See commands below, or more more details see  
[**gopjrt** installation instructions](https://github.com/gomlx/gopjrt?#installing).
 

### **Linux/amd64**, run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_linux_amd64.sh)):

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash
```

In addition, for **Linux+CUDA (NVidia GPU)** support, run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_cuda.sh)):

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash
```

Depending on what data formats you use, you may want to install `hdf5-tools` programs (`sudo apt install hdf5-tools` in Linux).

### **Darwin/arm64 and Darwin/amd64**

> [!Note]
> Currently, Darwin (MacOS) 🚧🛠 it only works with statically linked PJR CPU plugin 🛠🚧️ so that is the default (see issue in [XLA's issue #19152](https://github.com/openxla/xla/issues/19152) and on XLA's discord channels).
**Experimental**.

For **Arm64 (M1, M2, ... CPUs)**, run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_arm64.sh))

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin_arm64.sh | bash
```

For **Amd64 (i86_64)**, run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_amd64.sh))

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin_amd64.sh | bash
```

### 🐳  [Pre-built Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)

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

## 🧭 Tutorial

See the [tutorial here](examples/tutorial/tutorial.ipynb). It covers a bit of everything. 

After that look at the demos in the [examples/](https://github.com/gomlx/gomlx/tree/main/examples) directory.

The library itself is well documented (pls open issues if something is missing), and the code is
not too hard to read. 
Godoc available in [pkg.go.dev](https://pkg.go.dev/github.com/gomlx/gomlx).

Finally, feel free to ask questions: time allowing (when not in work) I'm always happy to help -- I created [groups.google.com/g/gomlx-discuss](https://groups.google.com/g/gomlx-discuss), or use [GitHub discussions page](https://github.com/gomlx/gomlx/discussions).

### Inference

Inference or serving a model is done currently by using the Go code used to create the model along with the checkpoint
with the trained weights and hyperparameters used to train the model. In other words, it uses the same tools used
for training.

For a simple example of how to do this and export a model inference as a library, see 
[`.../examples/cifar/classifer`](https://github.com/gomlx/gomlx/blob/main/examples/cifar/classifier/classifier.go), 
and its use in the last cells of the [Cifar-10 demo](https://gomlx.github.io/gomlx/notebooks/cifar.html).

In the future we plan to also export models to ONNX or StableHLO and one could use tools that serve those.

## 🎯 Long-term Goals

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
   - Multiple backends beyond XLA, e.g:  llamacpp, WebNN (with Wasm), pure Go version, etc.
   - Import pre-trained models from [Hugging Face Hub](https://huggingface.co/models) and allow fine-tuning -- ONNX versions already working for many models in [onnx-gomlx](https://github.com/gomlx/onnx-gomlx).  
   - Compile models to binary as in C-libraries and/or WebAssembly, to be linked and consumed (inference) anywhere
     (any language).

## FAQ

- **What are the environment variables are used by GoMLX ?** 
  - `GOMLX_BACKEND`: defines the backend engine to use (if using `backend.New()`). The value is formated as "<backend_name>:<backend_config>".
    Examples for XLA, the default engine: `GOMLX_BACKEND="xla:cpu"` (for CPU), `GOMLX_BACKEND="xla:cuda"` (for Nvidia CUDA) or
    `GOMLX=BACKEND="xla:/path/to/my/pjrt_plugin.so"` for some custom PJRT plugin.
  - `PJRT_PLUGIN_LIBRARY_PATH`: the underlying XLA backend uses this variable as an extra directory to search for plugin locations.
    It searches for the systems library paths (`$LD_LIBRARY_PATH`, `/etc/ld.so.conf`), the default `/usr/local/lib/gomlx/pjrt` and
    `$PJRT_PLUGIN_LIBRARY_PATH` if set.

## 🤝 Collaborating

The project is looking forward contributions for anyone interested. Many parts are not yet set 
in stone, so there is plenty of space for improvements and re-designs for those interested
and with good experience in Go, Machine Learning and APIs in general. See the [TODO file](docs/TODO.md)
for inspiration.

No governance guidelines have been established yet.

## 🚀 Advanced Topics

* [CHANGELOG](docs/CHANGELOG.md)
* [TODO](docs/TODO.md)
* [Error Handling](docs/error_handling.md)
* [Developing](docs/developing.md)

## ⚖️ License 

> Copyright 2024 Jan Pfeifer

**GoMLX** is distributed under the terms of the [Apache License Version 2.0](https://github.com/gomlx/gomlx/blob/main/LICENSE).
Unless it is explicitly stated otherwise, any contribution intentionally submitted for inclusion in this project shall be licensed under [Apache License Version 2.0](https://github.com/gomlx/gomlx/blob/main/LICENSE)
without any additional terms or conditions.
