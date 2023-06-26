# GoMLX changelog

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