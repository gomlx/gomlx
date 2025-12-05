# GoMLX changelog

# v0.25.0: Distributed execution; API cleanup (more Go idiomatic)

Hightlights:

- Distributed (cross-devices) execution: with AutoSharding and SPMD strategies; 
  Also added support for "portable device" execution.

- API changes: (will require simple fixes)
  - Most not graph building APIs now return errors (as opposed to panicking). Graph building functions
    still use panic to return error -- otherwise it's too painful to express math.
  - All "Rng" renamed to "RNG" -- acronyms in Go are usually capitalized.

Distributed computation improvements and refactorings:

- Package `graph`:
  - Fixed/improved documentation.
  - Added `IsNegative`, `IsPositive`, `IsNonNegative`, `IsNonPositive`.
  - Added `SubScalar` and tests for the '*Scalar' functions.
  - Added `Graph.WithDistributedStrategy`, `Graph.WithDeviceMesh`. `Graph.DeviceMesh` and `Graph.NumDevices`
  - Added `Graph.Distributed()` with "collective" (across devices) operations (like `AllReduce`).
  - Renamed: s/`Exec.InDevice`/`Exec.WithDevice`; s/`Exec.SetName`/`Exec.WithName`
  - Added `RunOnDevice`.
  - Added `Exec.AutoSharding` and `Exec.SPMD`.
- Package `context`:
  - Added `context.MustGetParam[T](ctx, key)` and `context.MustGetGraphParam[T](ctx, graph, key)`.
  - Added `Exec.AutoSharding` and `Exec.SPMD`.
  - Added `Variable.DistributedValue` and `Variable.SetDistributedValue`.
- Package `train`:
  - Added `train.DistributedDataset` and `train.BaseDataset`.
  - `Dataset.Reset` now returns an error.
  - `Trainer.TrainStep`, `Trainer.EvalStep` and `Trainer.Eval` now return errors as opposed to panicking.
  - Added `Trainer.WithDeviceAssignment`.
  - Added `Trainer.DistributedTrainStep`, `Trainer.DistributedEvalStep` and `Trainer.DistributedEval`.
- Package `datasets`:
  - Added `datasets.DistributedAccumulator`: converts a normal `Dataset` into a `DistributedDataset`.
  - Added `datasets.OnDevice`: pre-uploads data to devices.
- Package `backend`:
  - Added `Backend.CopyToDevice`
  - `Builder.Parameter()` now takes an optional `ShardingSpec` for sharded inputs.
  - Added ops: `AllReduce`
  - `Backend.NumDevices()` returns an int now.
  - Package `backends/notimplemented`:
    - Added dummy `Backend` that can be used to easily mock backends.
- Package `pkg/core/distributed`:
  -Added `DeviceMesh`, `ShardSpec` and `distributed.Tensor` objects.
- Package `pkg/core/tensors`:
  - Added `Tensor.CheckValid()`, `Tensor.Device()`, `Tensor.Backend()`
  - Changing it to return an error (as opposed to panic) where possible.

Other improvements:

- Package `simplego`:
  - Cleanups and improvements: thanks to @wunderbarb!
  - Fixed the issue with not handling the default value for the donate parameter in the Execute method.
- Package `cosineschedule`:
  - Added `WarmUpSteps` and `NumCycles` hyperparameters -- removed overloading of `periodSteps`.
- Added sponsorship badge and section to README.md. Also added the `FUNDING.yml` pointing to sponsorship.
- Added `.golangci.yml` and fixed many (still a long way to go) lint-warnings.
  - Based on https://gist.github.com/maratori/47a4d00457a92aa426dbd48a18776322
- GitHub actions (workflows):
  - Renamed tests to "Linux" and "Darwin."
  - Updated badges in README.md.
- Updated dependency to Gopjrt v0.8.5, fixing xlabuilder for new C compilers.
- Removed `ui/fyneui`:
  - It was incomplete, and it would be better offered as a separate package to avoid the dependencies.
- Package `graph`:
  - Added a negative and out-of-bounds indices test for `Gather`.
- Package `simplego`:
  - Partially fixed a race condition where the executable is finalized during the execution, causing crashes -- 
    Thanks @ajroetker!

# v0.24.1: 2025/10/23 Adding Darwin (Mac) support for CPU PJRT plugin

* Updated dependency to Gopjrt v0.8.4: added macOS (darwin/arm64) support and cpu PJRT plugin.
* Include `stablehlo` (== `xla`) by default for macOS in Darwin. 
* GitHub actions:
  * Added macOS tests.
  * Removed unnecessary `apt install` of packages.

# v0.24.0: 2025/10/21 **API change**: package tree restructured under `pkg`, `Exec` normalization; Backend `xla` now provided by `stablehlo`

* **Highlights** of this release:
  * Deprecating old "xla" backend (now called "oldxla") in favor of "stablehlo" (aliased to "xla" as well):
    in most cases nothing needs to be done (the `github.com/gomlx/gomlx/backends/default` will replace one by the other automatically),
    but in special cases there may require small changes.
  * Large refactoring: exported GoMLX packages moved under `/pkg`. The following changes: 
    * **This requires changes to the import paths**: core packages (`tensors`, `shapes` and `graph`) are under `pkg/core`;
      machine learning packages (`context`, `layers`, `train`, `datasets`, ...) are under `pkg/ml`;
      supporting packages (`fsutil`, `sets`, `xslices`, `xsync`) are under `pkg/support`.
    * Normalized `graph.Exec` and `context.Exec` slightly changed the API: 
      the `Exec.Exec...` methods now return an error, and the `Exec.MustExec...` methods panic (instead of the old `Exec.Call` format);
      The `graph.NewExec` and `context.NewExec` return errors, and the `graph.MustNewExec` and `context.MustNewExec` panic.
    * File utilities under the old `ml/data` now are under `pkg/support/fsutil`, and the package `ml/data` itself was 
      renamed `pkg/ml/datasets` and now only holds the various datasets types.
    * Packages that were not moved:
      * The `backends` package: it will move to its own repository later in the year (or early 2026)
      * The `ui` and `example` packages: since they are just extras, we keep them where they are for now. 
        The core `GoMLX` doesn't depend on them, so we are more lax with their external dependencies. 

<hr/>

* Copied external trivial `must` and `exceptions` packages to `/internal/...`, to remove external dependencies.
* Package `xla` (the old one): now **DEPRECATED** and called `oldxla`. The package `stablehlo` replaces it, including aliasing the `xla` backend name.
  * The old version is now registered as backend "oldxla".
  * Only included in `github.com/gomlx/gomlx/backends/default` if compiled with the tag `oldxla`.
* Package `stablehlo`:
  * Now completely replacing `xla` by default. Using `GOMLX_BACKEND=xla` will actually use the `stablehlo` backend.
  * Added `github.com/gomlx/gomlx/backends/stablehlo/cpu/dynamic` and `github.com/gomlx/gomlx/backends/stablehlo/cpu/static`
    to optionally force dynamic/static linking of the CPU PJRT plugin.
  * Disabled XLA logs by default by setting TF_CPP_MIN_LOG_LEVEL to 2 (errors level), if it is not already set.
* Package `graph`:
  * `NewExec`, `NewExecAny` `Exec`, `ExecOnce` and `ExecOnceN` now return an error on failure.
  * `MustNewExec`, `MustNewExecAny`, `MustExec`, `MustExecOnce` and `MustExecOnceN` panic on failure.
  * Introduced `Exec[1-4]` and `MustExec[1-4]` to execute the graph and return exactly 1-4 values.
  * If no seeds are given, initialize new random number generators with a cryptographically secure seed—on OSes that provide that.
  * Improved `Exec` tests.
* Package `context`:
  * `NewExec`, `NewExecAny` `Exec`, `ExecOnce` and `ExecOnceN` now return an error on failure.
  * `MustNewExec`, `MustNewExecAny`, `MustExec`, `MustExecOnce` and `MustExecOnceN` panic on failure.
  * Introduced `Exec[1-4]` and `MustExec[1-4]` to execute the graph and return exactly 1-4 values.
  * Improved documentation.
* Packages `pkg/support/...`:
  * Generic supporting functionality that is not core to GoMLX, but that users may also find useful to interact with it
    are now better (and hopefully more definitively) organized in packages under `pkg/support/`. The following packages were moved/created:
    * `xslices`, `xmaps`, `xsync`: extensions to the corresponding standard packages.
    * `set`: previously known as package `types/`
    * `fsutil`: file system handling utilities, previously in `data`.
* Package `inceptionv3` moved to `examples`
* Package `ui/commandline`: fixed progressbar in GoNB notebooks.
* Package `kan`: fixed `PiecewiseConstant*` layers for inputs of rank 1.
* Packages `downloader` and `huggingface`: had been already deprecated for a while, now removed. 
  See https://github.com/gomlx/go-huggingface for a replacement.
* Package `hdf5` moved to under `examples/inceptionv3`, for now the only example that uses it.
  If you need this, please let us know, maybe we move it to under support, or move it to https://github.com/gomlx/go-huggingface.
* Package `data` renamed to `datasets`; Split downloading functionality under `examples/downloader`.
* Package `commandline`:
  * Progressbar now shows the median step duration.
* Updated and refreshed all notebooks, including the tutorial.

# v0.23.2: 2025/10/01: Updated dependencies on `github.com/gomlx/stablehlo@v0.0.5` and `github.com/gomlx/gopjrt@v0.8.2`.

- Updated dependency to new Gopjrt v0.8.2 because of CUDA PJRT (lack of) backward compatibility issues.
- Package `stablehlo`:
  - Added support for comparison of bool values, and added corresponding tests.
  - Fixed wrong checking for during shapeinference.Gather

# v0.23.1: 2025/09/25: Small bug fixes.

* Package `backends`:
  * Removed op `Broadcast`: it was unnecessary, since `BroadcastInDim` is a superset.
* Package `graph`:
  * Backprop of `BroadcastPrefix` was not defined. Now that is uses `BroadcastInDim` instead, it works.
* Package `simplego`:
  * Only log ConvGeneral statistics on error.

# v0.23.0: 2025/09/21: beta `stablehlo` backend release

* Package `shapes`:
  * Added `FromAnyValue`: extract shape from a Go type.
* New backend: `stablehlo` (or simply _"hlo"_ for short) using https://github.com/gomlx/stablehlo.
  * All standard binary and unary ops implemented.
  * A handful of the standard ops also implemented.
  * If `backends/default` is compiled with `-tags=stablehlo` it will include the `stablehlo` backend.
  * Large cleanup of generators: most no longer depending on `gopjrt/xlabuilder`.
* Package `graph`:
  * `ArgMin`, `ArgMax`:
    * Fix of `ArgMin` now accepting negative axes.
    * For `stablehlo` and `go` backends NaNs will be deliberately selected (inline with Jax/TensorFlow/PyTorch)
  * `Clip` now uses the backend operation `Clamp`.
  * `Inverse` renamed to `Reciprocal` -- `Inverse` is now a deprecated alias to `Reciprocal`.
  * Added tests to various reduce operations.
  * Added `IsNaN`
  * Fixed `MaskedReduceMean`, when the mask provided is only a prefix rank to the input.
  * Package `nanlogger`:
    * `NanLogger.WithStopAtFirst` now can be used to control the default behavior of NanLogger.Trace.
* Package `backends`:
  * Ops are no longer auto-generated: now it is its own source of truth (as opposite to being generated from XLA code)
  * Added `IsNaN`
  * Many comments improvements.
  * Removed `SelectAndScatterSum`, which was wrong, and now is deprecated in `gopjrt`.
* Package `train`:
  * `Loop.EveryNSteps` takes into account the current global step (as opposed to always start the counting from 0).
  * Datasets implementing `train.Dataset` can now also implement `ShortName() string` to provide a short name to be
    used in metrics.
* Package `losses`:
  * `MeanSquaredError`: fixed weights/mask expected mask.
* Package `commandline`:
  * Exposed `RefreshPeriod` with frequency of command-line updates.
  * Fixed flickering of the progress bar / table of metrics.
  * Improved colors, "humanize" steps printing.
* `gomlx_checkpoints` CLI tool:
  * Added `-plot` to generate plots for all metrics. It accepts various models, so one can use it to compare models.

# v0.22.1: 2025/08/22 🌀 Convolutions 🌀 

(release v0.22.0 was skipped due to a bug notice slightly after release)

* Package `backends`:
  * `ConvGeneralDilated` renamed to `ConvGeneral`
* Package `backends/shapeinference`:
  * Added `ConvGeneralOp` to infer the output shape of a convolution.
* Package `backends/simplego`:
  * Implemented `ConvGeneral` operation: supporting strides, padding, dilations (input and kernel),
    and grouping (channels or batch), as well as transposing (arbitrary axes) convolutions.
* Package `types/shapes`: 
  * `Shape.Iter()` and `Shape.IterOn()` also yields the flat index being iterated.
  * Added `Shape.Strides()` and `Shape.IterOnAxes()`.
* Package `graph`:
  * Names of parameters for `ConvGeneral` were standardized to "input," "kernel" and "channels."
  * `ConvGeneralDilated` is being aliased to `ConvGeneral` and the former will be deprecated on
    a future version.
  * `ConvGeneral`: added gradient for grouped (by channels or by batch) convolutions.
  * Fixed shape of the kernel for `images.ChannelFirst` configuration.
  * Added `Split`.
  * `TransposeAllDims` -> `TransposeAllAxes`.
* Package `layers`:
  * Updated the configuration names for `Convolution`, to match the standards in the `graph` package.
  * Added `ChannelGroupCount()` and `BatchGroupCount()` to `Convolution` configuration.
* Updated to gopjrt v0.8.0, with the changes to the convolution API.

# v0.21.1: 2025/08/16 Added Zero-dim tensors support and other small improvements. 

* Package `tensors` and `graph`:
  * Added support for zero-dim tensors.
* Package `backends`:
  * Method **`New()` will return an error (as opposed to panic)**.
    The temporarily `NewOrErr` was marked as deprecated, use `New` instead.
* Package `optimizers`:
  * New `AdamConfig.WithBackoffSteps()` (or the hyperparameter `adam_backoff`) that prevents gradient steps
    from being taken until the given number of steps has executed. This allows a better estimate (moving average) of
    the gradients ("momentum") and their variances to be calculated before applying them.
  * New `optimizers.ParamAdamBeta1` and `optimizers.ParamAdamBeta2` hyperparameters to control Adam beta1 and beta2
    hyperparameters.
* Package `context`:
  * Added `Variable.DType()`.
  * Variable `#rngstate` marked as non-trainable during creation.
* `gomlx_checkpoints`:
  * Added `-perturb`.
  * Now it has its own `go.mod`, so it separated the dependencies.
* Docker:
  * Included `openssh-client` (ssh) and `dlv` (Go debugger) by default.
* `SimpleGo` ("go") backend:
  * Fixed mishandling of multi-output operations and race condition on parallel execution (#197)
  * Refactoring and clean up of execution loops.
  * Separated `TestDotGeneral_PerformanceTable` behind the build tag `perf`.

# v0.21.0: 2025/07/01 🌞 Summer Edition 🌞

* Package `simplego`:
  * Added `GetBackend` that returns a singleton backend, created with the default configuration at the first request.
* Package `ui/commandline`:
  * Added optional extra arbitrary metrics to print in the command-line with `AttachProgressBar`.
  * Added `FormatDuration` to pretty-print duration.
* Package `graph`
  * Added gradients of `Cos` and `Sin` that were missing.
  * Fixed (removed) the extra empty line in auto-generate functions comments that was preventing the documentation
     from being assigned to the functions.
  * Added parameters `sorted` and `unique` to `Scatter` (like the other functions `Scatter*`) -- **Small API change**.
  * Added `ScatterUpdate`, for now only for `unique=true`.
  * Package `nanlogger`:
    * Allow traces that only report also.
    * Created context parameter `optimizer.ParamNanLogger`: if set to NanLogger, it will trace all occurrences of
      of NaN values in gradient: great to debug where are the NaN appearing in the model first.
* Package `ml/train`:
  * Improved support for accumulated gradients. Fixed evaluation (context reuse) for when using accumulated gradients.
  * Added `Trainer.WithMaxExecutors`.
* Package `ml/train/metrics`:
  * `MeanMetric` allows for disabling dynamic batch weighting.  API slightly changed: `NewMeanMetric` now
    returns a `MeanMetric` struct, not an interface.
  * Added `StreamingMedianMetric`.
* Package `ml/train/optimizers`:
  * Added `RMSProp()` optimizer.
* Package `ml/layers`
  * Added normalizing 1/sqrt(d_k) factor to attention logits in the MultiHeadAttention layer: this will break current
    models using it.
  * Added `RMSNorm` normalizer.
* `gomlx_checkpoints` command-line tool:
  * Added support for multiple models to allow comparing models.
  * Fixed the printing of metrics with tiny values.
* Package `context`:
  * Allow VariableInitializers to use the `context.Context` itself, with its own random initializer.
  * `DefaultInitializer` now creates an initializer. The new default uses He initializer, the same used in PyTorch.
  * Package `initializers`:
    * They now use the `context` random number generator state, which simplifies things.
    * `ParamInitialSeed` removed, since the RNG is initialized by `Context.RngStateWithSeed()`.
* Fixed some flaky tests.

# v0.20.1: 2025/06/12 Trainer.AccumulateGradients (when the batch doesn't fit memory); VNN fixes; Numpy improvements.

* Package `train`:
  * Better handling of loss (without regularization) in metrics. Added `SetLossNoRegularization` and `GetLossNoRegularization`.
  * Added `Trainer.AccumulateGradients(n)` to accumulate n steps of gradients before applying them. This is useful if
    the desired batch size doesn't fit in memory, so it accumulates the gradients until the virtual batch size gradient
    is calculated.
* Package `optimizers`:
  * Added support for the new `train.OptimizeWithGradients` interface, to support gradient accumulators.
  * Cleaned up `StochasticGradientDescent` API. Added option to disable decay for testing.
* Pacakge `vnn`:
  * Added `Config.Scaler` to add a scaler operator just after the linear projection of a layer. It allows the VNN
    to operate on magnitude independent vectors.
  * Fixed the `LayerNormalization`, to make it more stable in backprop.
  * Fixed `Relu`: added support for non-shared non-linearities and a "leak" parameter ("vnn_relu_negative_slope").
  * Added `VNN().ActivationFn()` to allow setting arbitrary activation functions.
* Package `types/tensors/numpy`:
  * Added support for "Fortran order" files.
* Package `tensors`:
  * Attempting to finalize an "on-device" tensor whose backend has already been finalized is now a no-op -- as opposed to an panic.
  * Access to a on-device or shared buffer now checks that the backend hasn't been finalized.
    And if it has, it panics with a meaningful error message.
  * Added integration tests.

# v.0.20.0: Small API change: `backends.NewWithConfig()` changed to return an error.

* Package `backends`:
  * **API CHANGE**: Method `NewWithConfig()` changed
  * Method **`New()` will be changed to return an error (as opposed to panic) at next version**.
    Temporarily the methods `MustNew()` (which panics on errors, like today) and `NewOrErr` (which returns
    an error) were created to have a clear API, and `New()` was marked as deprecated. At the next version
    `New()` will change the API.
  * Added `IsFinalized()` to the Backend API, to better handle attempts to access finalized backends.
  * Fixed bug in `xla` backend where an error was not being sent when Backend was already finalized.
* Package `types/tensors/numpy` with methods to read and write tensors from/to `.npy` and `.npz` files.
* Package `simplego`:
  * Fixed bug introduced in parallelize version of Erf(x).
* Package `tensors`:
  * Added `Tensor.ToLocal()` to detach a tensor from its backend.
* Package `ui/gonb/plotly`:
  * Update dependencies to new go-plotly v0.7.0 (many changes to the API), while preserving as much as possible
    the GoMLX api offered.
* Updated example notebooks to use `github.com/gomlx/gomlx/backends/default` (instead of only `/xla`) and to
  use the new `backends.MustNew()`.

# v0.19.5: 2024/05/30 SimpleGo (go) backend optimizations

* Package `simplego`, the pure Go backend:
  * Added several benchmarks for SimpleGo DotGeneral. Run with:
    `go test ./backends/simplego/ -test.v -test.run PerformanceTable -perf`
  * DotGeneral reimplemented in 2 different versions:
    * Version for small inner matrices, with block iteration and loop unrolling.
    * Version for larger inner matrices: re-package inputs in ~4K blocks, and recursively partition matrices.
    * Added parallelization: at batch level and in the partitioning in the larger matrices.
  * Parallel execution of the Ops: that helps a lot during training (cut the training time almost in half for the adult
    dataset), but it may hurt inference if you are running many batches in parallel.
    So it dynamically decides to run sequentially or in parallel depending on the number of computations
    being executed concurrently.
    Added also configurations `GOMLX_BACKEND=go:ops_sequential` and `GOMLX_BACKEND=go:ops_parallel`
    to force one type of execution or another.
  * Parallelized Erf(x): this will become a model on how to parallelize other unary functions — probably
    when SIMD is available.

# v0.19.4: 2024/05/24 added Vector Neural Networks (VNNs)

* Vector Neural Networks (VNN): allows one to build 3D rotation (SO(3)) equivariant and/or invariant networks. See package `ml/layers/vnn`.
* Package `xla`
  * Remove dependencies to `gopjrt` internal protos: requires updated `Gopjrt`.
* Package `tensors`
  * Fixed pretty-print of booleans.

# v0.19.3: 2024/05/20 Many SimpleGo improvements.

* v0.19.2 skipped ... issues with the release.
* Package `simplego`:
  * Fixed `Gather` of scalar values.
  * Fixed `Where` checking of shape.
  * New ops: `NotEqual`, `Erf`, `ArgMinMax`, `ReduceWindow`, `ReduceBitwise{And,Or,Xor}` and
    `ReduceLogical{And,Or,Xor}`
  * Fixed initialization of re-used buffers where needed.
* Package `backends/default`:
  * Only include XLA by default on linux/amd64 platforms.
* Package `shapeinference`:
  * Changed to return errors instead of exceptions.
* Package `types/tensors`:
  * Removed dependency to `gopjrt/pjrt` -- otherwise we'll always need to install the C/C++ library.
* Package `types/shape`:
  * Added `Shape.Iter()` and `Shape.IterOn()`.
* Package `backend`:
  * `Backend` interface now returns errors instead of panicking.
* Package `graph`:
  * Added `NewExecOrError` and `Exec.CallOrError` as error-returning alternatives.
* gofmt cleanups by @zjtv

# v0.19.1: 2025/04/30 SimpleGo fixes and new ops; New XLA, requires Gopjrt v0.7.0 update.

* `go mod tidy`
* Package `simplego`:
  * "not implemented" error now includes the name of the corresponding method that was not implemented.
  * Several memory fixes.
  * Added `Slice` and `RngBitsGenerator` ops.
* Updated to Gopjrt v0.7.0, with more memory fixes. **Requires an update of the C++ libraries**.

# v0.19.0: 2025/04/29 Added SimpleGo, a pure Go backend

* Package `backends`:
  * Added `simplego`, a portable, simple albeit slow backend.
    * Implemented ~50 most common ops, see `backends/simplego/capabilities`, and most common numeric types (including BFloat16).
  * Added sub-package `notimplemented`: helper to implement new backends.
  * Added sub-package `shapeinference`: helper to implement new backends.
  * Added sub-package `default` which includes the default packages.
  * Added `List()` function that returns the currently registered (compiled-in) backends.
* Package `checkpoints`
  * Added `Config.FromEmbed` that allows loading a checkpoint from an embedded variable.
* Package `graph`:
  * `Gather` and `GatherSlices` now have and extra argument called `indicesAreSorted` that tells whether
    the start indices are guaranteed to be sorted, which allows some optimizations in some platforms.
  * Exposed `BackendGather`, `BackendScatterMax`, `BackendScatterMin` and `BackendScatterSum` for test and debugging
    purposes.
* Moved code generation tools from `cmd` to `internal/cmd` directory.

# v0.18.1: 2025/04/13 Many fixes, XLA update, Tensor clone.

* XLA Backend:
  * Updated gopjrt dependency: fix to Scatter flags.
* Package `graph`:
  * Removed spurious logging.
  * Added gradient for ScatterSum, ScatterMax, ScatterMin. Only for simple shapes for now.
  * Fixed ExecOnceN to return many outputs.
* Package `tensors`:
  * Added `Tensor.Clone` and `Tensor.OnDeviceClone`.
* Package `context`:
  * Removed deprecated `NewContext`
  * Added `Variable.CloneToContext`
  * Added `Context.Clone`
  * Variable graphToNodeId is now a `xsync.SyncMap`, solving issues for concurrency of multiple graphs being
    created/executed at the same time for the same Context.Exec object (with different shapes).
  * Added `Variable.Finalize` and `Context.Finalize`.
* Updated all dependencies and re-tested.

# v0.18.0: Ragged2D; XLA update; Fixed Scatter functions; Fixed memory leaks.

* XLA Backend:
  * Updated dependency to newest Gopjrt 0.6.3: small memory leak fixes
  * Updated CPU PJRT and XlaBuilder
  * Fixed Scatter* functions.
* Package `graph`:
  * Fixed `ScatterSum` (renamed from the now deprecated `ScatterAdd`), `ScatterMax` and `ScatterMin`. No gradients for `ScatterMax` and `ScatterMin` yet.
  * Added `Ragged2D` with some utilities, in particular `Ragged2D.Softmax`.
  * `DefaultNodeLogger` now accepts the `#full ` prefix that forces printing the full value of a tensor,
    in Go-code format.

# v0.17.1: 2025/02/26 CosineSimilarity, Bitcast and many fixes and improvements.

* Added MNIST example (thanks to @TuSKan).
* `gomlx_checkpoints` now displays the value of scalar variables.
* Package `checkpoints`:
  * Loading a checkpoint overwrites the values of variables already present in the context.
  * Fixes when saving, in particular if using `Immediate()` loading.
* Package `tensors`:
  * Allow shared tensors to be donated.
* Package `graph`:
  * Fixed when using axes != -1 for `L1Norm`.
  * Added `IsZero` shortcut.
  * Fixed `L2Normalize` to handle 0s without NaN, both in the forward evaluation, and in the gradient.
  * Renamed indicator functions to `PositiveIndicator`, `NonNegativeIndicator`, `NegativeIndicator` and `NonPositiveIndicator`.
  * Added backprop for `ReduceMin` that was missing (thx @TuSKan)
  * Added `CosineSimilarity`, numerically safe for 0 vectors.
  * Added `BitcastConvert`.
* Package `ml/context`:
  * Added support for string derived types for `context.GetParamsOr[T]`.
* Package `ml/train`:
  * Created `ExecPerStepUpdateGraphFn` for those creating custom "TrainStep" functions.
* Package `ml/train/losses`:
  * Triplet losses now work with context.
  * `CheckExtraLabelsForWeightsAndMask` now (1) accepts weights and mask in any order; (2) normalize weights such that the sum is (non-masked) bathSize,
    preserving the ratio. This way the mean will be 1.
  * Losses with masks and weights fixed so weights/mask can be given in any order.
    Also, now using MaskedReduceMean if there is a mask, and all losses return a scalar.
* Package `xla`:
  * Removed suppression of logging: new PJRTs are not outputting random debug messages anymore.
* Updated dependency to `gopjrt` v0.6.2.
* Replaced `stringer` by `enumer` everywhere.

# v0.17.0: bitwise ops, triplet losses, new layers, fixes, and more.

* Backend API change: separating Logical and Bitwise versions of various ops derived from And, Or, Xor and Not.
* Updated dependency to gopjrt v0.6.0.
* Added "Flow Matching" examples/demo.
* Package `layers`:
  * Added `layers.DropBlock`, a type of dropout for images.
  * Added `layers.DropPath` and `layers.DropPathFromContext`, a type of dropout used in Residual connections, to drop full paths.
  * `layers.LayerNormalization`:
    * up-scale precision by default if input is a Float16 or BFloat16. Low-precision
      lead to NaNs when reducing values for normalization. Added also a hyperparameter to configure normalization DType.
* Added `Context.RandomBenoulli` to sample from a Bernoulli (binary) distribution.
* Correctly pretty-print Float16 and BFloat16 tensors.
* Several fixes and small improvements to command-line tool `gomlx_checkpoint`.
* Package `nanlogger`:
  * Store only the stack-trace, and trim the stack into the nanlogger package.
  * Does not exit, simply report the NanLogger. User can define a handler, if they want the training to exit.
  * Use `IsFinite` to check for NaN and Infs: but we loose the type of NaN that happened.
  * Fixed nanlogger for Float16 and BFloat16; Also, it first prints other logged tensors, before failing with a NaN.
* Package `losses`:
  * Added `ParamLoss`: hyperparameter to define the loss, and many constant values.
  * Added `LossFromContext`, using `ParamLoss` hyperparameter.
  * Added `MakeHuberLossFromContext`
  * Added experimental `MakeAdaptivePowerLoss` and `MakeAdaptivePowerLossFromContext`
  * Added TripletLoss: various negative sampling strategies and distance metrics.
* Package `graph`:
  * More unit tests.
  * Aliases nodes: allow setting aliases to nodes, and to retrieve them by those aliases. Useful for layers
    or models to export intermediary nodes by their aliases. They are prefixed by scope. New methods are:
    `Node.WithAlias`, `Node.GetAlias`, `Graph.GetNodeByAlias`, `Graph.PushAliasScope`, `Graph.PopAliasScope`
    and `Graph.IterAliasedNodes`.
  * Added optional aliases nodes for `inceptionv3` model.
  * Added `ReduceSkewness` and the alias `Skewness`.
  * Added bitwise ops:
    * `BitwiseShiftLeft`, `BitwiseShiftRightLogical`, `BitwiseShiftRightArithmetic`, `BitwiseAnd`, `BitwiseOr`,
      `BitwiseXor`, `BitwiseNot`.
  * Kept an alias from `And`, `Or` and `Not` to the `LogicalAnd`, `LogicalOr`, `LogicalXor` and `LogicalNot`.

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
