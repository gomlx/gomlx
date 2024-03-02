# GoMLX changelog

## Next

* Open Graph Benchmark OGBN-MAG dataset support and example models (FNN and GNN).
  * Added sampler library.
* Package `context`:
  * added `Manager()` accessor method.
  * added `SetParams` to set various parameters at once.
  * renaming name of parameters to be prefixed with "Param".
* Package `context/initializers`:
  * added `GlorotUniformFn`
  * random initializers use zeros for non-float variables by default (as opposed to crash)
  * default initializer now matches Keras (random uniform from `[-0.05, 0.05]`).
* Package `context/checkpoints`:
  * added `ExcludeVarsFromSaving` to allow preventing saving large static variables. 
* Package `shapes`:
  * Added `Check()` and `Assert()` to check for both, dtype and dimensions.
  * Added `EqDimensions()` to compare dimensions.
  * `Make(dtype, dimensions...)` now makes a copy of the `dimensions` slice given.
* `exceptions`: refactoring to use separate package `github.com/gomlx/exceptions`.
* Package `graph`:
  * Functions that take masked inputs are being renamed to use a "Masked" prefix (e.g.: `MaskedReduceSum`,
    `MaskedReduceMean`, `MaskedReduceMax`, `MaskedReduceAndKeep`).
  * Added `MaskedReduceMean`.
* Package `layers`:
  * Added `...FromContext` family of functions, that apply layers according to parameters set in the context: 
    `ActivationFromContext`, `DropoutFromContext`, `NormalizeFromContext` and `MaskedNormalizeFromContext`.
  * `LayerNormalization`: fixed shaping bug, and renamed `scale` to `gain`, more aligned with [original paper](https://arxiv.org/pdf/1607.06450v1.pdf)
    * **This will break previous models using LayerNormalization!**: this is not taken lightly, but as it is, it
      is wrong and depending on the shape it may be adversely affecting some models.
  * `LayerNormalization`: added `Mask` support; added defaults from context parameters.
  * `DropoutStatic`: Dropout api where one can pass a static dropout rate as a Go float.
  * `AddL2RegularizationStatic`: Add L2 regularization on values, where the amount of regularization is static. 
* Package `optimizers`:
  * Added `CosineAnnealingSchedule.FromContext`.
* Package `losses`:
  * Added support for `weights` and `mask`.
* Package `ml/data`:
  * Renamed `Map` -> `MapWithGraphFn`: to make it explicit that the transformation happens in accelerator.
  * Added `Map`: a map function to a dataset that runs in host (as opposed to in accelerator/XLA).
  * Added `Freeing`: a dataset wrapper that frees inputs and labels in between each call to `Yield`: to control GPU
    memory usage. It replaces `loop.FreeInput()`
* Package `commandline`:
  * `AttachProgressBar` now displays a continuously updated table with metrics generated during training.
    This only works in the commandline (not in notebooks).
  * Added `CreateContextSettingsFlag` and `ParseContextSettings`.

## 0.8.0 - 2023/11/28

* DType and Tensors:
  * Added support to Go's `int64` -- breaks compatibility because DType Int64 when converted back to Go becomes `int64`
    and not `int`.
  * Renamed Local.Flat -> Local.FlatCopy : not to be mixed with LocalRef.Flat (which is not a copy).
* C++ code integrating with XLA:
  * Enable copy elision -- which makes `std::move` not necessary.
  * Temporarily copied `xla/mlir/utils` library to `deps/xla_mlir`, since it is not available in all XLA distributions.
* Package `context`:
  * Added `context.GetParamOr` and `context.GetGraphParamOr`: it uses generics to cast to the desired type, and allowing a default value to be returned.
  * Added `Context.DeleteVariable` and `Context.DeleteVariablesInScope`.
* Package `checkpoints`: 
  * Added recovery of some basic types (numeric and slices) when loading params from Json.
  * Added unique incrementing id to checkpoint file names.
* Package `exceptions`: special case runtime panics to preserve its stack-trace.
* Package `train`: 
  * `Loop` automatically sets LoopStep to context's "global_step" parameter.
  * Models (e.g.: unsupervised) can return `nil` for predictions.
* Package `optimizer`: 
  * Added `GetGlobalStep`.
  * Interface now include `Clear(ctx)` to clear all variables used by an optimizer --> this also breaks
    compatibility for any custom optimizer, unfortunately. 
    But if it broke you, it should be a very easy fix, since most optimizers use a fixed scope for its variables, and
    `Context.DeleteVariablesInScope` will do the job.
  * Added `DeleteGlobalStep`.
* Package `context`: Added `Context.EnumerateVariablesInScope()` method.
* Package `graph`:
  * Added optional `reduceAxes` parameter to `L2Norm` and `L1Norm`.
  * Added `L2NormSquare`, `L2Normalize` and `L2NormalizeWithEpsilon`. 
* Package `nanlogger`: added `AttachToTrainer`; improved docs.
* Package `margaid`: 
  * automatic ending plot when loop finishes.
  * option to plot evaluation losses separately from training losses -- for when they include different terms.
* Example "Dogs vs Cats":
  * Added Byol (Bootstrap Your Own Latent) regularized models.
  * Added support for generating pairs of images for BYOL model.

## v0.7.2 - 2023/10/27

* Fixed C/C++ mismatching malloc/new/new[] and free/delete/delete[].
  * Formatted C/C++ code using clang-format.
* Increased static size threshold for string memory leak test.
* Small StableHLO support improvements.
* Fixed and updated devel docker ('janpfeifer/gomlx_devel:latest').

## v0.7.1 - 2023/10/26

* Fixed search of CUDA paths under /usr/local.
* Fixed(?) XLA ShapedBuffers issue causing spurious crashes after update. 
* JupyterLab docker image uses gomlx_xla C library from local disk (as opposed to downloading it).

## v0.7.0 - 2023/10/25

* Update OpenXLA/XLA dependencies:
  * Updated `devel/Dockerfile` with fixed dependencies, and better instructions to work around Bazel cache.
  * Fixed several build breaking issues intersecting XLA and CUDA.
  * Added automatic finding of CUDA directory (for `libdevice.10.bc` file).
* Oxford Flowers 102:  Added support for GoNB widgets; Improved image `Generator`.
* Fixed rare race-condition with GC and CGO.
* Minor typos and reformatting (no execution code change).
* Added various badges to README.md.

## v0.6.0 - 2023/08/07

* FFT, RealFFT, InverseFFT and InverseRealFFT operations.
  * Added a small notebook demo for FFT. 
* Added Complex/Imag/Real/Conj operations to manipulate complex numbers (and their gradients).
* Added support for complex numbers for ConvertType. Defined gradient for ConvertType.
* Added Complex128 and Complex64 dtypes support.
* Added "spacers" (like "*" for axis ranges) and `AxisElem()` for `Slice()`.
* Package `examples/notebook/gonb/margaid`: Added `Plots.AddValues` and `Plots.PlotToHTML`; 
  Fixed `Plots` returned by `New()` to be linear scale by default.
* Included build of tcmalloc (`gperftools`) from the `c/` directory, when building `libgomlx_xla.so`. 
  Still the `libtcmalloc.so` is needed in runtime. 
  A copy is included in the `gomlx_xla.tar.gz` package (under `lib/gomlx`) and can be copied from there if needed.
  This enables build for Macs — see #23.

## v0.5.0 - 2023/07/10

* Error handling revamp: using `panic` to report errors — it works as exceptions. This is a very large change
  affecting most of the code.
* Added `NewManager`, a simpler interface to create a `Manager` object with defaults.
* Added `margaid.NewDeafult`, simplifying adding of plots for the default cases.
* Examples: 
  * UCI-Adult: replaced `adult.Dataset` to the much simpler and powerful `data.InMemoryDataset`.
* Remove `tensor.Local.Data()`: now all access is done throw the `tensor.Local.AcquireData()` and release, to
  prevent a race condition with the garbage collector.
* Update of XLA C++ library.

## v0.4.1

* Diffusion example: Added conditioning on flower type; Improved documentation; several other small improvements.
* NanLogger: added tool to report back (with stack trace and scope) on the occurrences of NaN/Inf in the computation
  graph.
* Checkpoints: added `Handler.LoadedVariables()` method for inspection of loaded checkpoint. 
* Bug fixes: 
  * RandomNormal: fixed rare numerical issues in RandomNormal, that would generate -Inf.
  * Context: some rare condition on feeding variable values to executor.
  * InMemory dataset: handling cases where dataset returns the same tensor as input and label.
* Slices: refactored `IotaSlice()` to `Iota[T number]()`.

## v0.4.0

* Models: Diffusion example model (working draft); added Kernel Inception Distance (KID) metric implementation.
* Contexts: added `context.NumParameters()`, `context.Memory()`, `context.RandomUniform`, `context.RandomNormal`, 
  `context.RngStateWithSeed` and `context.RngStateReset`.
* Random numbers revamped, making graph purely functional. Also, 'context.Context' provides
  the facilities to carry around random number generator state.
* Added ops: `ArgMax`, `ArgMin`, `ExpandLeftToRank`, `RandomUniform` and `RandomNormal`.
* Datasets: `InMemoryFromData` (for testing); `Normalization()` returns mean and standard deviation for dataset;
  `Map()` creates new dataset that maps a function to wrapped dataset; `Take(n)` to take n elements from a dataset.
* Layers: Added `layers.Activation` that takes the activation type as a string (easy to plug to a flag).
* Metrics: added context as the first parameter to `metrics.BaseMetricGraph`.
* Plots (margaid): added support for saving and restoring points (when continue training); optional log-scale plots;
  allow for arbitrary rate of updates; added support for loading data from multiple models. 
* Losses: added `losses.MeanAbsoluteError`.
* Optimizers: added `optimizers.GetGlobalStepVar`.
* Training loop (`train.Loop`): added `MeanTrainingStepDuration()`; check for infinity and "nan" losses -- training
  is immediately interrupted with an error.
* Added to slices package: `Flag()`, `At()`, `Last()`, `Copy`.
* Force download of the correct version of the C++ library in the Jupyter docker -- this
  prevents Docker cache using an older version.
* Improved error messages in some cases.
* Tensors: added new dtypes `UInt32` and `UInt64`; changed return type of `tensor.FromAnyValue()` to `tensor.Tensor`.

## v0.3.1

* DogsVsCats: added inception model type; fix of metrics types for plotting.
* BatchNormalization: differentiable inference code; added Trainable() support.
* Fixed notebooks broken with v0.3.0 changes.
* Skip plotting batch loss (we keep the moving average of the batch loss though).

## v0.3.0, 2023-06-01

* Inception V3 model: including downloading pre-trained weights and various configurations.
* Tensors: added Load, Save for Local tensors.
* Added HDF5 format support for loading values.
* Skip evaluation during test of demos.
* Fixed dogsvscat demo's inconsistent mixed datasets issue, by yielding a correct spec.
* Added SumPool and MeanPool
* Changed API for defining images channels axis configuration (in pooling and convolution operations). 

## v0.2.1, 2023-05-20

* Tensors: clean up, fixed memory race (with Go's GC not knowing about and C++ pointers), improved 
  docs and test.
* Created tests from the Adult, Cifar, "Dog vs Cats" and Imdb demos.

## v0.2.0, 2023-05-18

* Added Oxford Flowers 102 Dataset example (no model yet).
* Added Datasets tools: Parallel (improved), Batch, InMemory.
* Added ops: GatherSlices (and its gradient), EinsumAxes, MaxScalar, MinScalar, ExpandAndBroadcast.
* Added Interpolate operation -- for series/image/videos resizing.
* Added support for int32 (`shapes.I32`) and uint8 (`shapes.UInt8` or `shapes.U8` for short).
* Added Set[] to `types` package.
* Added `types/tensor/image` with image conversion tools to/from tensors.
* Added Serialize and Deserialize for `tensor.Local` and `Shape`.
* Fixed issue with `tensor.Device` not using the correct clientId.

## v0.1.1, 2023-04-29

* Small fixes to example notebooks.
* Added documentation to the various dataset libraries.
* Renamed release asset not to include the version name to simplify downloading the latest one.
* Updated `docker/jupyterlab` for the new release. 

## v0.1.0, 2023-04-28

* Updated OpenXLA/XLA dependency to the current at 2023-04-28.
* Added `docker/devel` for development and building the Go/C++ bridge library.
* Changed `Exec.Call` method to return an error directly.
* Added `docker/` subdirectory.
* Added `docker/jupyterlab`: docker that includes JupyterLab and GoNB for quick getting started. Available in [janpfeifer/gomlx_jupyterlab](https://hub.docker.com/repository/docker/janpfeifer/gomlx_jupyterlab/general) for now.
* Fixed various documentations.
* Tutorial clean up and added links.
* Fixed cache issue in `ParallelDataset`.

## v0.0.1

* Initial upload of experimental but functional GoMLX including examples.