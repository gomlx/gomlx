# TODO List

It's endless ... while GoMLX has a thin vertical implementation from the ground 
up of a full ML framework, the breadth still doesn't compare with PyTorch/Jax.
But hopefully the functionality that is already there may cover a lot 
of the use cases.

Split by functionality, the most "desirable" TODOs are:

## Modeling

* Importing models from HuggingFace (and others?)
  * [ONNX-GoMLX](https://github.com/onnx/onnx-gomlx) already allows for many models to be imported as a graph.
    But not all ops are implemented yet, and not all models are supported.
  * Import generic classes of models from HuggingFace using safetensors directly, in particular transformer based
    models: extend the package `pkg/ml/model/transformer` to support as many as possible.
    * Create example of generator that take as input the hugging-face transformer model and a prompt and generate
      the text from it.
* More ML: latest publications, layers, optimizers, losses, regularizers, etc.
  * GNN Layer: tbd., it would be nice to have a GNN layer/scheme available as well.
* Checkpointing for gradient: to save memory during training.
* Adding support for Jaccobian (gradients with respect to non-scalar values)

## Graph

* Support dynamic shapes:
  * Input-dependend shapes (but not data-dependend), still fixed rank.
  * Data-dependend shapes, still fixed rank.
  * Symbolic shapes (named axes): symbolic shape inference.
  * Current PR: https://github.com/gomlx/gomlx/pull/306

## Backends

Note: the backend is going to be shared with the [GX project](https://github.com/gx-org/gx)

* Go backend (`backends/simplego`):
  * More "fused ops" for the Go backend.
  * More SIMD support: 
    with upcoming Go 1.26 and above and the [go-highway](https://github.com/ajroetker/go-highway/) library.
* ONNX as a backend (inside [ONNX-GoMLX](https://github.com/onnx/onnx-gomlx))
  * Add API to save the graph as ONNX model.
* WebGL/WebNN backend, for WASM.
* [llama.cpp](https://github.com/ggerganov/llama.cpp) backend using the "purego"
  [github.com/hybridgroup/yzma](https://github.com/hybridgroup/yzma) library, for CPU inference.
* Other backends ?

## Infrastructure

* Inference-only version: do we want to have save/load format of the model (that doesn't require 
  importing GoMLX, but just the backend) ? Maybe ONNX is enough ?
* Distributed training: mostly done, but not tested yet.

## API Improvements

Nothing specific here ... but while some thought was put into the API, it certainly can be improved.

- Replace `Context` object with plain Go struct (with annotations) for variables and hyperparameters:
  - This will require some thought and lots of code introspection (`reflect` package),
    since often the context is used go set global settings.

## Fixes / Code Maintenance