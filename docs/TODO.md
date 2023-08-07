# TODO List

It's a large list (many in the middle of the code) ... while GoMLX has a thin vertical implementation from the ground 
up of a full ML framework, it has very little breadth yet (my guess is < 1% of something like Jax, TensorFlow or
Pytorch). But hopefully the 1% of the functionality may cover a large percentage of the use cases.

Split by functionality, the most "desirable" TODOs are:

## Modeling

* CNNs: add SeparableConv2D. The goal would be to have an exact ResNet50 working.
* GNN Layer: tbd., but is one of the recent hotness, it would be nice to have a GNN layer/scheme available as well.
* Importing models from TensorFlow/Jax/Pytorch/Hugging Face
  * Both: (A) As a black-box for inference only; (B) for further fine-tuning.
    * But if only the weights, and the model is translated by hand would already be good.
  * Have a few of the standard models available: ResNet50 (older but a good reference), ViT, BERT, Chinchilla.
  * Have a clear story importing models from Hugging Face (at least of one type, like TF or Jax, since they
    also use XLA).
* Computation Graph extensions and manipulation tools: there are good reasons for someone to want to 
  change the Graph (splitting the graph for batch processing or distribution) or create arbitrary 
  extensions to it (custom operations in Go or C/C++) and be able to differentiate through those. 
  This is something that needs some design and thought.
* Detecting first occurrence of NaNs (and Inf): have a mode -- likely slower -- where these are automatically checked
  for and immediately prints a stack trace when they happen.

## Infrastructure

* Saving/Loading models:
  * Exporting to TensorFLow's "SavedModel" format -- so models can leverage all the production tools
    from TensorFlow -- using same mechanism as Jax is using, by converting code to StableHLO intermediary
    language -- that conversion is done already.
* Inference-only version: for now to run predictions one has to also import the whole engine and XLA machinery. 
  * Ahead-Of-Time (AOT) compilation of a computation graph to a library that doesn't require linking 
    the whole XLA. This can be the "official" save for inference method. Notice compiled graph will
    work only on the hardware it was compiled for. See [discussion in OpenXLA](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/0RXscLOHWtc).
* Distributed training: at least synchronous mirrored strategy (data parallelism but no model parallelism yet) 
  * Multi-device (GPU) set up.
  * Multi-tenancy (multiple hosts) set up.
* Docker:
  * (**Done in v0.0.2**) Dockerfile and a docker with Jupyter, GoNB and GoMLX.
  * Dockerfile for a devel version, with everything needed to compile the C++ bindings.
* Jupyter(-lab) Notebook/Colab integration: -- now implemented in GoNB. Missing features for GoMLX support:
  * Easy, automatic save/cache values of Tensors, so programs can be coded progressively without needing
    to re-run things, simply by recovering previous results.
  * Remote XLA server?: allow version that talks (through shared memory) with a remove XLA server. A bit
    slower (potentially lots slower depending on how much is communicated) but with close to zero
    start-up time, perfect for development.
* More supported data types (`DType`): missing the 16bit float types.

## Lower level
* Add support for multiple devices (e.g: multiple GPUs). In principle, it's already there (DeviceNum is supported)
  but it hasn't been tested. But `train.Trainer` and `context.Exec` will need special casing.
  * Implement a DistributedTrainer that will automatically distribute across multiple devices. Data 
    parallelism at first, not model parallelism yet.

## API Improvements

Nothing specific here ... but while some thought was put into the API, it certainly can be improved.
And since these are earlier days in the library, we expect things to change some.
