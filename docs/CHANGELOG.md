# GoMLX changelog

## Next

* Added ops: GatherSlices (and its gradient), EinsumAxes, MaxScalar, MinScalar, ExpandAndBroadcast.
* Added Interpolate operation -- for series/image/videos resizing.
* Added support for int32 (`shapes.I32`).
* Added Set[] to `types` package.

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