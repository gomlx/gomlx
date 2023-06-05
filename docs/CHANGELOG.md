# GoMLX changelog

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