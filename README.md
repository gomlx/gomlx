# **_GoMLX_**, an Accelerated ML and Math Framework

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gomlx?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gomlx)](https://github.com/gomlx/gomlx/blob/main/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gomlx)](https://goreportcard.com/report/github.com/gomlx/gomlx)
[![Linux Tests](https://github.com/gomlx/gomlx/actions/workflows/linux_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/linux_tests.yaml)
[![Darwin Tests](https://github.com/gomlx/gomlx/actions/workflows/darwin_tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/darwin_tests.yaml)
![Coverage](https://img.shields.io/badge/Coverage-72.5%25-yellow)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)


## 📖 About **_GoMLX_**
<img align="right" src="docs/gomlx_gopher2.png" alt="GoMLX Gopher" width="220px"/>

**GoMLX** is an easy-to-use set of Machine Learning and generic math libraries and tools. 
It can be seen as a **PyTorch/Jax/TensorFlow for Go**.

It can be used to train, fine-tune, modify, and combine machine learning models. 
It provides all the tools to make that work easy: from a complete set of differentiable operators, 
all the way to UItools to plot metrics while training in a notebook.

It runs almost everywhere Go runs, using a pure Go backend. 
It runs even in the browser with WASM ([see demo created with GoMLX](https://janpfeifer.github.io/hiveGo/www/hive/)). 
Likely, it will work in embedded devices as well (see [Tamago](https://github.com/usbarmory/tamago)).

It also supports a very optimized backend engine based on [OpenXLA](https://github.com/openxla/xla) 
that uses just-in-time compilation to CPU, GPUs (Nvidia, and likely AMD ROCm, Intel, Macs) and Google's TPUs.
It also supports modern distributed execution (**new, still being actively improved**) for multi-TPU or multi-GPU
using XLA Shardy, an evolution of the [GSPMD distribution](https://arxiv.org/abs/2105.04663)).

It's the same engine that powers Google's [Jax](https://github.com/google/jax), 
[TensorFlow](https://tensorflow.org/) and [Pytorch/XLA](https://docs.pytorch.org/xla/master/learn/xla-overview.html),
and it has the same speed in many cases. 
Use this backend to train large models or with large datasets.

> [!Tip]
> * See our 🎓 [**tutorial**](https://gomlx.github.io/gomlx/notebooks/tutorial.html) 🎓
> * See _Eli Bendersky_'s blog post ["GoMLX: ML in Go without Python"](https://eli.thegreenplace.net/2024/gomlx-ml-in-go-without-python/), 
>   (a bit outdated, but still useful)
> * A [guided example for Kaggle Dogs Vs Cats](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html).
> * [Installation here](#installation).

<div>
<p>It was developed to be a full-featured ML platform for Go, productionizable and easily to experiment with ML ideas
—see Long-Term Goals below.</p>

It strives to be **simple to read and reason about**, leading the user to a correct and transparent mental model 
of what is going on (no surprises)—aligned with Go philosophy.
At the cost of more typing (more verbose) at times.

It is also incredibly flexible and easy to extend and try non-conventional ideas: use it to experiment with new
optimizer ideas, complex regularizers, funky multitasking, etc.

Documentation is kept up to date (if it is not well-documented, it is as if the code is not there), 
and error messages are useful (always with a stack-trace) and try to make it easy to solve issues.
</div>

## 🗺️ Overview

**GoMLX** is a full-featured ML framework, supporting various well-known ML components  
from the bottom to the top of the stack. But it is still only a slice of what a major ML library/framework should provide 
(like TensorFlow, Jax, or PyTorch).


**Examples developed using GoMLX:**

  * [Adult/Census model](https://gomlx.github.io/gomlx/notebooks/uci-adult.html);
  * [How do KANs learn ?](https://gomlx.github.io/gomlx/notebooks/kan_shapes.html); 
  * [Cifar-10 demo](https://gomlx.github.io/gomlx/notebooks/cifar.html); 
  * [MNIST demo (library and command-line only)](https://github.com/gomlx/gomlx/tree/main/examples/mnist)
  * [Dogs & Cats classifier demo](https://gomlx.github.io/gomlx/notebooks/dogsvscats.html); 
  * [IMDB Movie Review demo](https://gomlx.github.io/gomlx/notebooks/imdb.html); 
  * [Diffusion model for Oxford Flowers 102 dataset (generates random flowers)](examples/oxfordflowers102/OxfordFlowers102_Diffusion.ipynb);
    * [Flow Matching Study Notebook](https://gomlx.github.io/gomlx/notebooks/flow_matching.html) based on Meta's ["Flow Matching Guide and Code"](https://ai.meta.com/research/publications/flow-matching-guide-and-code/).
  * [GoMLX/Gemma](https://github.com/gomlx/gemma), a **GoMLX** implementation of 
    [Google DeepMind's Gemma v2 model](https://github.com/google-deepmind/gemma) ([blog post](https://ai.google.dev/gemma))
  * [GNN model for OGBN-MAG (experimental)](examples/ogbnmag/ogbn-mag.ipynb).
  * Last, a trivial [synthetic linear model](https://github.com/gomlx/gomlx/blob/main/examples/linear/linear.go), for those curious to see a barebones simple model.
  * Neural Style Transfer 10-year Celebration: [see a demo written using GoMLX](https://github.com/janpfeifer/styletransfer/blob/main/demo.ipynb) of the [original paper](https://arxiv.org/abs/1508.06576).
  * [Triplet Losses](https://github.com/gomlx/gomlx/blob/main/ml/train/losses/triplet.go): various negative sampling strategies as well as various distance metrics.
  * [AlphaZero AI for the game of Hive](https://github.com/janpfeifer/hiveGo/): it uses a trivial GNN to evaluate
    positions on the board. It includes a [WASM demo (runs GoMLX in the browser!)](https://janpfeifer.github.io/hiveGo/www/hive/) and a command-line UI to test your skills!

**Highlights:**

> **🚀 NEW 🚀**: **Distributed Execution** (across multiple GPUs or TPUs) with little hints from the user.
> One only needs to configure a distributed dataset, and the trainer picks up from there.
> See code change in [UCI-Adult demo](https://github.com/gomlx/gomlx/blob/main/examples/adult/demo/main.go#L222)

> **🚀 NEW 🚀**: Fixed Mac support for the XLA backend, including installer. Only CPU for now, 
> see [jax/issues/32800](https://github.com/jax-ml/jax/issues/32800) for the request to Apple developers 
> to update their support for GPU XLA.

> **🚀 NEW 🚀**: `gomlx_checkpoints`, the command line tool to inspect checkpoints of train(-ing) models,
> now also **generate plots** with loss and arbitrary evaluation metrics using Plotly.
> See [example of training session](https://gomlx.github.io/gomlx/notebooks/gomlx_checkpoints_plot_example.html),
> with the effects of a learning rate change during the training.
> It also allows plotting different models together, to compare their evolution.

* Converting ONNX models to GoMLX with [onnx-gomlx](https://github.com/gomlx/onnx-gomlx): both as an alternative for `onnxruntime` (leveraging XLA),
  but also to further fine-tune models. See also [go-huggingface](https://github.com/gomlx/go-huggingface) to easily download ONNX model files from HuggingFace.
* [Docker "gomlx_jupyterlab"](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab) with integrated JupyterLab and [GoNB](https://github.com/janpfeifer/gonb) (a Go kernel for Jupyter notebooks)
* Two backends:
   1. **`xla`**: [OpenXLA](https://github.com/openxla/xla) backend for CPUs, GPUs, and TPUs. State-of-the-art as these things go. Only linux/amd64 for now.
      1.a: To be replaced by the new `stablehlo` backend, currently in beta. See the note above.
   2. **`go`**: a pure Go backend (no C/C++ dependencies): slower but very portable (compiles to WASM/Windows/etc.): 
      SIMD support is planned [when it becomes available](https://github.com/golang/go/issues/73787); See also [GoMLX compiled to WASM to power the AI for a game of Hive](https://janpfeifer.github.io/hiveGo/www/hive/)
* Autograd: automatic differentiation—only gradients for now, no jacobian.
* Context: automatic variable management for ML models.
* ML layers library with some of the most popular machine learning "layers": FFN layers,  
  various activation functions, layer and batch normalization, convolutions, pooling, dropout, Multi-Head-Attention
  (for transformer layers), LSTM, KAN (B-Splines, [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594), Discrete-KAN, PiecewiseLinear KAN),
  PiecewiseLinear (for calibration and normalization), various regularizations,
  FFT (reverse/differentiable), learnable rational functions (both for activations and [GR-KAN/KAT networks](https://arxiv.org/abs/2409.10594)),
  VNN (Vector Neural Networks) for SO(3)-Equivariant/Invariant layers, etc.
* Training library, with some pretty-printing. Including plots for Jupyter notebook, using [GoNB, a Go Kernel](https://github.com/janpfeifer/gonb).
  * Also, various debugging tools: collecting values for particular nodes for plotting, simply logging  the value
    of nodes during training, stack-trace of the code where nodes are created.
* SGD and Adam (AdamW and Adamax) optimizers.
* Various losses and metrics.
* Pre-Trained models to use: InceptionV3 (image model), many more from HuggingFace using [onnx-gomlx](https://github.com/gomlx/onnx-gomlx).
  See also [go-huggingface](https://github.com/gomlx/go-huggingface) to easily download ONNX model files from HuggingFace. 
* Read Numpy arrays into GoMLX tensors -- see package `github.com/gomlx/gomlx/pkg/core/tensors/numpy`.
* Support static linking of PJRT: slower to build the Go program, but deploying it doesn't require installing a PJRT plugin in the machine you are deploying it.
  Use `go build --tags=pjrt_cpu_static` or include `import _ "github.com/gomlx/gomlx/backends/stablehlo/cpu/static"`.

## 👥 Support

* Discussion in the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U) (you can [join the slack server here](https://invite.slack.golangbridge.org/)).
* [Q&A and discussions](https://github.com/gomlx/gomlx/discussions/categories/q-a)
* [Issues](https://github.com/gomlx/gomlx/issues)
* Random brainstorming on projects: just start a Q&A, and I'm happy to meet in discord somewhere or VC.
* [Google Groups: groups.google.com/g/gomlx-discuss](https://groups.google.com/g/gomlx-discuss)

## <a id="installation"></a>🛠️ + ⚙️ Installation (Only needed for the XLA backend)

If you want to use only a pure Go backend, simply do `import _ "github.com/gomlx/gomlx/backends/simplego"` and 
there is no need to install anything.

For XLA there are two simple options:

(1) [Use the Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab);

(2) Install the pre-built XLA/PJRT binaries (C/C++ libraries) for Linux (including CUDA) or macOS with the interactive self-explanatory installer:

```bash
go run github.com/gomlx/gopjrt/cmd/gopjrt_installer@latest
```

> [!NOTE]
> For now it works for (1) CPU PJRT on linux/amd64 (or Windows+WSL); (2) Nvidia CUDA PJRT on Linux/amd64; (3) CPU PJRT on Darwin (macOS);
> (4) TPUs in Google Cloud.
> I would love to support for AMD ROCm, Apple Metal (GPU), Intel, and others, but I don't have easy access to hardwre to test/maintain them.
> If you feel like contributing or donating hardware/cloud credits, please contact me.

## 🐳  [Pre-built Docker](https://hub.docker.com/r/janpfeifer/gomlx_jupyterlab)

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

It runs on Windows as well: _Docker Desktop_ uses WSL2 under the hood.

## 🧭 Tutorial

See the [tutorial here](examples/tutorial/tutorial.ipynb). It covers a bit of everything. 

After that, look at the demos in the [examples/](https://github.com/gomlx/gomlx/tree/main/examples) directory.

The library itself is well-documented (pls open issues if something is missing), and the code is
not too hard to read. 
_Godoc_ is available in [pkg.go.dev](https://pkg.go.dev/github.com/gomlx/gomlx).

Finally, feel free to ask questions: time allowing (when not at work), I'm always happy to help—I created 
[groups.google.com/g/gomlx-discuss](https://groups.google.com/g/gomlx-discuss), or use the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U).

### Inference

Inference or serving a model is done currently by using the Go code used to create the model along with the checkpoint
with the trained weights and hyperparameters used to train the model. In other words, it uses the same tools used
for training.

For a simple example of how to do this and export a model inference as a library, see 
[`.../examples/cifar/classifer`](https://github.com/gomlx/gomlx/blob/main/examples/cifar/classifier/classifier.go), 
and its use in the last cells of the [Cifar-10 demo](https://gomlx.github.io/gomlx/notebooks/cifar.html).

In the future we plan to also export models to ONNX or StableHLO, and one could use tools that serve those.

## 🎯 Long-term Goals

1. Building and training models in Go -- as opposed to Python (or some other language) -- with focus on:
   - Being easy(ier) to read and reason about, leading the user to a correct and transparent mental
     model of what is going on. Even if that means being more verbose when writing.
   - Clean, separable APIs: individual APIs should be self-contained and decoupled where possible.
   - Composability: Any component should be replaceable, so they can be customized and experimented with.
     That means sometimes more coding (there is not one magic train object that does everything),
     but it makes it clear what is happening, and to replace parts with custom versions.
   - Up-to-date documentation: if the documentation is not there or if it's badly written, it's as 
     if the code was not there either.
   - Clear and actionable error reporting
2. To be a productive research and educational platform to experiment with new ML ideas and learn.
   - Support mirrored training on multiple devices and various forms of distributed training (model and/or data
     parallelism), in particular to support for large language models and similarly large model training.
3. To be a robust and reliable platform for production. Some subgoals:
   - Support modern accelerator hardware like TPUs and GPUs.
   - Multiple backends beyond XLA, e.g:  llamacpp, WebNN (with Wasm), pure Go version, etc.
   - Import pre-trained models from [Hugging Face Hub](https://huggingface.co/models) and allow fine-tuning -- ONNX versions already working for many models in [onnx-gomlx](https://github.com/gomlx/onnx-gomlx).  
   - Compile models to binary as in C-libraries and/or WebAssembly, to be linked and consumed (inference) anywhere
     (any language).

## 🤔 FAQ

- **What are the environment variables used by GoMLX?** 
  - `GOMLX_BACKEND`: defines the backend engine to use (if using `backends.New()`). The value is formatted as "<backend_name>[:<backend_config>]",
    with the config part being optional. Examples:
    - `GOMLX_BACKEND=go`: Use the `SimpleGo` backend, the pure Go implementation that is very portable but slow.
    - `GOMLX_BACKEND="xla:cpu"`: Use XLA (the faster backend, only runs on Linux now) for CPU
    - `GOMLX_BACKEND="xla:cuda"`: Use XLA for for Nvidia CUDA
    - `GOMLX_BACKEND="xla:/path/to/my/pjrt_plugin.so"`: Use XLA with an arbitrary PJRT. PJRT is a plugin system for XLA to support different hardware.
      One can install PJRTs build for NVIDIA GPUs (there is an installation script for that), there is also one for ROCm (not tested by the author),
      for TPU (Google Cloud) and reports of PJRTs being built to even new accelerators (e.g.: [TensTorrent XLA](https://github.com/tenstorrent/tt-xla))
  - `PJRT_PLUGIN_LIBRARY_PATH`: the underlying XLA backend uses this variable as an extra directory to search for plugin locations.
    It searches for the systems library paths (`$LD_LIBRARY_PATH`, `/etc/ld.so.conf`), the default `/usr/local/lib/gomlx/pjrt` and
    `$PJRT_PLUGIN_LIBRARY_PATH` if set.
  - `XLA_FLAGS`: optional controls for XLA backend. It should be set to a semicolon (";") separated list of options. If you set to `--help` 
    the backend will print out some help for all options. There is also a description on the page [XLA Flags Guidance](https://openxla.org/xla/flags_guidance).
- **What backends to include when using GoMLX?**
  - The recommendation is to use `import _ "github.com/gomlx/gomlx/backends/default"` which will import `xla` (alias to `stablehlo`) and
    `go` (_SimpleGo_) backends. If you add `-tags=noxla` to the compiler it won't include XLA/stablehlo backend.
  - `import _ "github.com/gomlx/gomlx/backends/simplego"` to include only `go` (no C++ dependencies)
  - `import _ "github.com/gomlx/gomlx/backends/stablehlo"` to import only XLA (stablehlo).
  - A deprecated version of *XLA* backend (named `oldxla`) is available if you `import _ "github.com/gomlx/gomlx/backends/xla"`.
    This is temporary only and will be removed in future releases. It is also included if you 
    `import _ "github.com/gomlx/gomlx/backends/default"` and use `-tags=oldxla` when building.

## 🤝 Collaborating

The project is looking forward to contributions for anyone interested. 
Many parts are not yet set in stone, so there is plenty of space for improvements and re-designs for those interested
and with good experience in Go, Machine Learning, and APIs in general. See the [TODO file](docs/TODO.md)
for inspiration.

No governance guidelines have been established yet.

See the section Support above to get in touch (Slack channel or Google Groups)!

## 💖 Support the Project

If you find this project helpful, please consider 
[supporting our work through GitHub Sponsors](https://github.com/sponsors/gomlx). 

Your contribution helps us (currently mostly [me](https://github.com/janpfeifer)) dedicate more time to maintenance
and add new features for the entire GoMLX ecosystem.

It also helps us acquire access (buying or cloud) to hardware for more portability: e.g.: ROCm, Apple Metal (GPU), 
Multi-GPU/TPU, NVidia DGX Spark, Tenstorrent, etc.

## 🚀 Advanced Topics

* [CHANGELOG](docs/CHANGELOG.md)
* [TODO](docs/TODO.md)
* [Error Handling](docs/error_handling.md)
* [Developing](docs/developing.md)

##  💖 Thanks

* [Go](golang.org)
* [OpenXLA](https://github.com/openxla/xla)
* [TensorFlow](https://www.tensorflow.org/)
* [Jax](https://github.com/jax-ml/jax)
* [PyTorch](https://pytorch.org/)
* [ONNX](https://onnx.ai/)
* [HuggingFace](https://huggingface.co/)

## ⚖️ License 

> Copyright 2025 Jan Pfeifer

**GoMLX** is distributed under the terms of the [Apache License Version 2.0](https://github.com/gomlx/gomlx/blob/main/LICENSE).
Unless it is explicitly stated otherwise, any contribution intentionally submitted for inclusion in this project shall be licensed under [Apache License Version 2.0](https://github.com/gomlx/gomlx/blob/main/LICENSE)
without any additional terms or conditions.
