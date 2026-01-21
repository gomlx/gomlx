# GoMLX is machine learning framework (like PyTorch, JAX, TensorFlow, etc.) for Go.

GoMLX is deliberately "non-eager": one builds the computation graph, and then executes it on a backend.

It supports multiple backends, defined under `/backends`. Currently implemented:
- `go`: a pure Go backend, implemented in package `backends/simplego`.
- `xla` (`stablelho` is an alias): uses XLA for acceleration (it JIT-compiles and executes using XLA's PJRT). It is implemented in package `backends/xla`, which in turn uses github.com/gomlx/go-xla to interface with XLA.

Currently it uses "fixed shapes", like XLA. So a different shape requires a dynamic JIT-recompilation. There is work in progress for the Go backend to add support for dynamic shapes.

## File Structure

- `backednds`: abstract interface, and their concrete implementations (e.g. `backends/simplego`, `backends/xla`).
- `pkg`: public API of the GoMLX library. It is organized in subpackages, e.g. `pkg/nn` for neural networks, `pkg/optx` for optimizers, etc.
  - `core`: core API definitions, to allow building of graphs and their executions.
  - `ml`: ML support libraries, including training, variables handling (with `context`), optimizers, layers, datasets, checkpointing, etc.
  - `support`: support libraries, including xsync, xerrors, etc.
- `internal`: internal libraries and generators.
  - `cmd`: command line tools: mostly generators.
- `examples`: examples of usage of the library. Some well maintained, others may be outdated.
- `ui`: UI tools, for intance to display a training progress bar, or if using GoNB (a Jupyter kernel for Go), to dynamically plot a graph with the training progress.
  It is separated in its own top-level directory, to prevend UI packages dependencies to pure GoMLX.
- `.github/workflows`: continuous integration, one file per platform. 

Normal code files are prefixed with the following copyright line:

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

Files that start with `gen_` are auto-generated and don't include a copyright line
directly -- the copyright line is in their generators.
Many are created with generators included under `internal/cmd/...`, and the generated file 
includes a comment stating which tool was used to generate them.

