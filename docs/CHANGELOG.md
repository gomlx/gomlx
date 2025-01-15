# GoMLX changelog

# Next

* Added "Flow Matching" examples/demo.
* Added `layers.DropBlock`, a type of dropout for images.
* Added `layers.DropPath` and `layers.DropPathFromContext`, a type of dropout used in Residual connections, to drop full paths.
* Added `Context.RandomBenoulli` to sample from a Bernoulli (binary) distribution.
* Correctly pretty-print Float16 and BFloat16 tensors.
* Fixed nanlogger for Float16 and BFloat16; Also, it first prints other logged tensors, before failing with a NaN.
* Several fixes and small improvements to command-line tool `gomlx_checkpoint`.
* `nanlogger`:
  * Store only the stack-trace, and trim the stack into the nanlogger package.
  * Does not exit, simply report the NanLogger. User can define a handler, if they want the training to exit.
  * Use `IsFinite` to check for NaN and Infs: but we loose the type of NaN that happened.  
* `layers.LayerNormalization`: 
  * up-scale precision by default if input is a Float16 or BFloat16. Low-precision
    lead to NaNs when reducing values for normalization. Added also a hyperparameter to configure normalization DType.
* Package `losses`: 
  * Added `ParamLoss`: hyperparameter to define the loss, and many constant values.
  * Added `LossFromContext`, using `ParamLoss` hyperparameter. 
  * Added `MakeHuberLossFromContext`
  * Added experimental `MakeAdaptivePowerLoss` and `MakeAdaptivePowerLossFromContext`

# v0.16.1 - 🎄 2024/12/19 🎄 MatMul fixes
* MatMul fixed for some edge shape configuration and greatly accelerated in some cases.

# v0.16.0 - 🎄 2024/12/19 🎄 Benchmarks, Speed improvements with gopjrt v0.5.0, Shared buffers.

* XLA backend now accepts the absolute path to the PJRT plugin (`GOMLX_BACKEND="xla:<pjrt_path>"`)
* Updated GitHub action (`go.yaml`) to only change the README.md with the result of the change, if pushing to the
  `main` branch.
* Added `Pow()` gradient.
* Package `tensors`:
  * Added `Tensor` transfer to/from device benchmarks.
  * Added `Tensor.CopyFrom()` to transfer from one tensor (potentially on device) directly to another
    tensor -- handy for pre-allocated tensors.
  * Added the convenience `Tensor.AssignFromFlat[T](toTensor, fromFlat)`
  * Added "shared" tensors: `Tensor.IsShared()` to check if using it. This saves one copy when using a tensor
    as input, when it is changed by the host in-between executions of a graph.
  * `Tensor.ConstFlatData` now avoids a copy, if `Backend.BufferData` is available.
* Updated dependency to gopjrt v0.5.0, with support for shared buffers.
* Package `backends` and `backends/xla`:
  * Added `Backend.HasSharedBuffer`, `Backend.NewSharedBuffer` and `Backend.BufferData`.

# v0.15.3 - 2024/11/25

* Added pre-linking of CPU PJRT packages, both statically and dynamically.
* Re-enabling Mac version: currently only statically linked.

# v0.15.2 - 2024/11/17

* Fixed printing of `uint` tensors.
* Fixed Dockerfile.
* Example CIFAR -- changes will break previous checkpoints:
  * Added inference example for Cifar models.
  * Fixed model scope issue.
  * Fixed KAN model issue.
* Added `checkpoints.Load()`: just like `checkpoints.Build`, but it complains if a checkpoint doesn't exist.
* Package `graph`: 
  * Added `ReduceVariance` and an alias `Variance`. Fixed `ReduceAndKeep` if no axes are given.
  * Added `Stack`: similar to `Concatenate` but it creates a new axis.
* BSpline(Standard)-KAN:
  - Better initialization -- constant variance across layers.
  - Extrapolate constant.
  - Knots from -1.0 to 1.0.
* PiecewiseLinear-KAN: better initialization (constant variance across layers)
* Added `layers/lstm` to create LSTM layers (experimental), in use by ONNX conversion to GoMLX.
* Updated dependencies; gopjrt v0.4.7.

# v0.15.1 - 2024/11/11 Updated downloader, in support for 

* Updated dependency to **gopjrt** 0.4.5
* Moving package `huggingface` and `downloader` to "github.com/gomlx/go-huggingface": marked as deprecated.
* Added checks and better error report for misuse of rngState in random functions.
* Added graph.RandomIntN and context.Context.RandomIntN.

# v0.15.0 - 2024/11/01 Some API clean up; Added support for ONNX model conversion.

* Package `graph`:
  * Added `MatMul`, with semantics similar to `numpy.matmul`
  * Renamed `ExpandDims` to `InsertAxes` and added `ExpandAxes`: the old `ExpandDims` had a slightly different
    semantics than the usual (numpy) `expand_dims` that I hadn't realized. The name change reflect that difference,
    and the new `ExpandAxes` match the more common semantics of `expand_dims`. Added proper documentation.
    **BREAKING CHANGE**: easy to convert, but breaking anyway: it requires attention.
    We defined a deprecated 'ExpandedDims' that maps to `InsertAxes`, but it will be removed on the next release.
  * Graph/Node introspection: added `node.ConstantValue`, `node.IsConstantExpression`.
* Package `context`:
  * Fixed `ExecOnce`: it was missing the variadic args for the computation graph.
  * `InspectVariableInScope` and `InspectVariable` renamed to `GetVariable` and `GetVariableByScopeAndName`
    respectively. Alias to the older names left for compatibility (and marked as deprecated), but they
    will be removed in future versions.
* Package `tensors`:
  * Added `Tensor.Summary(precision int)` to pretty-print a summary of the values of a tensor, numpy-like.

## v0.14.0 - 2024/10/24

* Package `context`
  * New `VariableWithValueGraph` to create/get a variable with value set to the graph value (Node).
  * New `IterVariables` and `IterVariablesInScope` that use the new go 1.23 iterators.
* New directory `ui` with various front-ends for displaying training progress, plots, etc.
  * **BREAKING CHANGE**: Refactored all UI tools under `ui` directory. It only requires changing the import, the APIs are not changed.
  * New package `fyneui`, a window based training UI built using Fyne.io (EXPERIMENTAL)
* Package `commandline`:
  * `ParseContextSettings` now allows parsing settings from a text file. 
  *  Fixed `SprintContextSettings` for scoped hyperparameters.
  *  Added `SprintModifiedContextSettings` to enumerate only hyperparameters set on the command line.
* New package `cosineschedule`, refactored from `optimizers` package.
  * Added handling negative values for the hyperparameter `cosine_schedule_steps`: they set the period of the cosine schedule
    as fractions of the total number of steps being trained.
* Package `train`:
  * Extensions to `Dataset` interface through additional interfaces.
  * Added optional `IsOnwershipTransfer() bool` that allows a Dataset to specify it should maintain ownership of the 
    yielded tensors.
* Updated `gopjrt` v0.4.4 with the static XlaBuilder library, and experimental support for Apple/Metal.

## v0.13.0 - 2024/10/07

* Package `initializers`
  * All random initializers (`RandomUniformFn`, `RandomUniformFn`, `RandomNormalFn`, `GlorotUniformFn`, `XavierUniformFn`)
    changed to take the context as a parameter, instead of `initialSeed`. 
  * The `initialSeed` is instead read from the hyperparameter `initializers.ParamInitialSeed` ("initializers_seed")
    and default to `initializers.NoSeed` (0), which means the seed is randomly started.
* Added learnable rational functions (ml/layers/rational): can be used for activations or as univariate learnable
  functions for KAN.
  * Added rational notebook to generate initial values with approximations to arbitrary univariate functions.
* Package `graph`:
  * Added `ConstCachedTensor` to allow caching of constant tensors.
  * Fixed gradient of `Where` when operands are broadcast.
  * Added `ConsecutiveDifference`, `SliceAxis`, `BitsCount`, `IsFinite`.
* Package `context`:
  * Added `context.ExecOnce` and `context.ExecOnceN`.
  * `context.GetParamOr` now returns the default value for a hyperparameter, if it is set to nil. 
* Package `train`:
  * Added `GetTrainLastStepVar` with information about last step of training: used for setting up various schedules.
  * Added `ResetComputationGraphs` to allow the trainer to recreate computation graphs, if hyperparameters change in the middle
    of training -- for training with very different schedules, for instance with freezing variables.
* Added `initializers.BroadcastTensorToShape`: to allow variables to be initialized with a base value that is broadcast
  to each variable shape requested.
* Package `optimizers`
  * Added `MonotonicProjection` to project values (usually variables) to a monotonically increasing values, with a margin.
  * Added `ParamClipNaN` to prevent NaNs going into gradient updates.
* Added `regularizers.ConstantL1`
* Added `data.NewConstantDataset` with a dummy dataset that can be used when training a model that generates
  its own input and labels.
* Package `kan`:
  * Discrete-KAN:
    * Added separate (per input) split points.
    * Added support for hyperparameter configured split points.
    * Added monotonic projection of split points.
    * Added ConstantL1 regularizer for control points.
    * Added various types of schedules for smoothness: cosine, linear, exponential.
    * Added normal distribution based perturbation.
    * Added input grouping.
  * Added GR-KAN (Rational Functions)
  * Added PWL-KAN (Piecewise-Linear) with `kan.New().PiecewiseLinear()`.
* Fixed OGBN-MAG GNN tests and demo.

## v0.12.0 - 2024/09/23

* Updated dependency to gopjrt v0.4.0
* Added package `ml/data/downloader` for parallel downloads, with support for authentication tokens.
* Added package `ml/data/huggingface` to download and load HuggingFace models into tensors.
* Removed dependency to gonb/common. Added package `types/xsync` with the required synchronization constructs.
* Added `Shape.Dim(axis)` as a shortcut, where `axis` can use negative values.
* Package `graph`:
  * `Scalar()`, `AddScalar()`, `MulScalar()`, `DivScalar()`, ... are now generic, and take as input any non-complex
    number type, for improved convenience.
  * Added `ShapedLowerTriangular()`, `TakeLowerTriangular()` and `TakeUpperTriangular()`
* Added `activations.Gelu` and `activations.GeluApproximate`
* Added `Erf`, the "error function", used when integrating the normal distribution, and its gradient.
* Better Discrete-KAN support: configurable by hyperparameters.

## v0.11.3 - 2024/08/29

* Immediately free accelerator (GPU) memory, where possible -- as opposed to waiting for the garbage collector.
  * This impacts the train.Loop and train.Trainer: they both immediately finalize the inputs and labels after use.
* Fixed nil point exception, where initializer was not properly set if value of a variable was loaded from a checkpoint.
  * This impacted when restarting training with batch normalization.
* Fixes to the notebooks: some small things were broken on the v0.11.0 transition; large speed-up with v0.11.1 fixes.

## v0.11.2 (was v0.11.1) - 2024/08/28

* Added support for `dtypes.BFloat16`.
* Added `tensors.FromScalar`
* Updated to gopjrt v0.3.0
* Package `graph`:
  * Added `ExecOnce` and `ExecOnceN` 
  * Added `CumSum`
  * `ConvertDType` to the same dtype is now a no-op.
  * Added `LogicalAll` and `LogicalAny`
  * Added `DynamicSlice` and `DynamicUpdateSlice`
* Package `backend`:
  * Added `DynamicUpdateSlice`, `DynamicSlice`, `ReduceAnd` and `ReduceOr`.
* Package `tensors`:
  * Fixed race condition `Tensor.DonateBuffer`.
  * Fixed unnecessary copying of tensor data in `Tensor.MaterializeOnDevices`
* Small fixes to documentation.

## 0.11.0 BREAKING CHANGE: Multi-Backend support; Added XLA/PJRT support (with gopjrt); meaningful speed ups; No more C code (all goes through gopjrt) 

* MAJOR REFACTORING. Many breaking compatibility changes -- it would be a major release number change, if it were > v1 already.
* New package `backends`: no GoMLX can support different backends -- but for now only xla is implemented.
  * Sub-package `xla` implements the XLA/PJRT version, based on [`github.com/gomlx/gopjrt`](github.com/gomlx/gopjrt) project.
* Package `tensors':
  * `tensor` -> `tensors`, more inline with other package names, and allow one to use `tensor` as a variable name.
  * Now there is only one `Tensor` type (not an interface), that manages local and on-device storage.
  * Local storage using Go
  * On-device storage now using generic `backends.Backend` api.
  * Improved testing using xla, greatly simplified.
* Package `graph`:
  * Added support for donated tensors for execution.
  * Added Nodes to introspect nodes of the graph -- e.g.: investigate the largest nodes if one is running out of memory.
  * Updated `OneHot` to use `Where`.
  * Added `GrowLeft`, `GrowRigth`, `Infinity`, `LogSoftmax`, `MaskedLogSoftmax`
  * `BroadcastToDims` and `BroadcastToShape` will automatically expand x to match.
  * `AdjustAxisToOperandRank` made public.
* Package `layers`:
  * Added sub-package `fnn` for a simplified Feedforward Neural Networks implementation.
  * Added sub-package `kan` for Kolmogorov–Arnold Networks, and Discrete-KAN.
    * Included bspline GoMLX implementation.
  * Added sub-package `regularizers` with automatic regularizer configuration. Layers `Dense`, `DenseWithBias` and `kan` use it by default.
  * Added sub-package `activations` -- just a refactor of the code already in layers.
  * Added sub-package `batchnorm`: refactored out batch normalization code. 
    * Added `batchnorm.AveragesUpdate` to update the average of the means and variances used for normalization.
      Also connected it to evaluation in plots libraries.
* Package `initializers`:
  * Added `XavierFn` initializer.
* Package `losses`:
  * Fixed `CategoricalCrossEntropyLogits` and `SparseCategoricalCrossEntropyLogits`.
  * Added `MakeHuberLoss`
* Package `metrics`:
  * Fixed 
* Package `exceptions` moved to a separate repository in [`github.com/gomlx/exceptions`](github.com/gomlx/exceptions).
* Package `slices` renamed to `xslices`, not to mix up with the new standard pacakge `slices`.
* Package `tensors/image` renamed `tensors/images`.
  * Added all numeric dtypes support; Added conversion tests to all types.
  * Added support to `dtypes.Float16`.
* Package `context`
  * Renamed `context.NewContext` to `context.New`.
  * Added `Variable.Reset`: reset a variable, to be reinitialialized.
* Package `checkpoints`: added `ExcludeParams` and `ExcludeAllParams`.
* Package `plots`
  * Added `Point.Short` for short-name of metrics in saved metrics.
* C/C++ code:
  * Completely removed, all C/C++ dependencies are in `gopjrt` project now.
  * Removed reference to AOT compilation, see #52.
* Added command-line tool `gomlx_checkpoints` to introspect checkpoints.
* Added `cmd/run_coverage.sh`.

## 0.10.0 - 2024/06/12

* `types.shapes` package:
  * **Added support for `Float16` training -- tested with GNNs.**
    * Up-precision metrics dtypes if they are `Float16`.
    * Allow arbitrary dtype for `Adam` optimizer -- it requires at least `float32`, even if the model runs on `float16`.
    * DType dependent `epsilon` values for `Softmax` and `Adam` -- current values would lead to `NaN` with `float16`.
    * Added `DType.IsFloat16` to check for `Float16` or `BFloat16` (not yet well-supported).
  * Added support for `Int8`, `Int16`, `Uint8` and `Uint16`.
  * Renamed `UInt{X}` to `Uint{X}` and added a deprecated alias to the old form (so it still compiles).
* Added logging of time to build and compile graph. Last version improved a lot the execution time, but slowed the compilation.
* Context.Variable:
  * Fixed `Variable.SetValueGraph` when the shape changes. Improved some documentation.
  * Fixed `Variable.SetValuePreservingOld` when shapes change.
  * Fixed checking of loaded variables -- that they are not newly created.
* Package `optimizers`:
  * Fixed optimizer constructor `FromContext` to allow further configuration of the optimizer by setting other hyperparameters into context.   
  * Added hyperparameter `clip_step_by_value`, a clip by value applied to gradient updates.
  * `Adam` optimizer: `"clip_step_by_value", "adam_epsilon", "adam_dtype"` hyperparameters support.
  * **`MustOptimizerByName` now takes also the context for the optimizer hyperparameters.** -- this breaks the API.
* Package `checkpoints`:
  * Allow adding variables to exclude from saving after checkpoint is created -- for newly created variables
* Added `slices.CloseToEpsilon` to easily customize tests.
* `Scatter` doesn't assume indices are sorted or unique.
* Plotly training plots: added `WithCustomMetricFn` for custom metrics and `ScheduleEveryNSteps`.
* Added OGBN_MAG GNN example:
  * Including Layer-Wise Inference.
* Package graph:
  * Added `Shift`, `ShiftLeft`, `ShiftRight`, `ShiftWithScalar`, `ShiftWithValue`.
* Dummy package for xla.AOT and xla.StableHLO APIs enabled when using "google3" build tag: this allows the dependency
  to the corresponding C++ code to be dropped. (Thanks @tdegris).
* Removed xla.AOTExecute: see issue #52

## 0.9.1 - 2024/04/19

* XLA integration:
  * Added "SKIP_ABSL_INITIALIZE_LOG", for conflict cases, while https://github.com/abseil/abseil-cpp/issues/1656 is
    not solved.

## 0.9.0 - 2024/04/18

* Binary GOMLX+XLA distribution:
  * Now requires package `libnccl > 2.21` to be installed.
  * Updated to CUDA version `12.3` and Cudnn `8.9`.
  * Newer version GPU performance measured on a GNN model improved significantly (In one model the median train step went from 160ms to 110ms). 
    On CPUs measured on the "CSI Adult" dataset remained the same. 
* Open Graph Benchmark OGBN-MAG dataset support and example models (FNN and GNN).
  * Added sampler library.
* Package `graph`:
  * added `MirroredLog1P`.
  * Functions that take masked inputs are being renamed to use a "Masked" prefix (e.g.: `MaskedReduceSum`,
    `MaskedReduceMean`, `MaskedReduceMax`, `MaskedReduceAndKeep`).
  * Added `MaskedReduceMean`.
  * Added `IdentityWithCustomGradient`, to allow for manual tweaks to the gradient.
  * Fixed for special case of gradient on `broadcastInDimVJP`.
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
  * fixed issue with lazy-loading of variables.
* Package `shapes`:
  * Added `Check()` and `Assert()` to check for both, dtype and dimensions.
  * Added `EqDimensions()` to compare dimensions.
  * `Make(dtype, dimensions...)` now makes a copy of the `dimensions` slice given.
* `exceptions`: refactoring to use separate package `github.com/gomlx/exceptions`.
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
  * Added `CosineAnnealingSchedule.FromContext`. New `MinLearningRate` is 0.0 (same used in Keras).
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
  * Asynchronous display of updates: it works better with very fast training loops or if running
    over a slow terminal connection (network).
  * Added `CreateContextSettingsFlag` and `ParseContextSettings`.
* Package `plots`, `margaid` and `plotly`:
  * Added `margaid.Plots.PlotEveryNSteps`.
  * Remove `margaid.Plots.Done`, no longer needed, as closing of writing file is done automatically at the end of the
    training loop.
  * Added Plotly plots.
* Ahead-Of-Time compilation:
  * Not yet working, and actually broken. This still requires some XLA hacking to get right (if at all possible).

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
