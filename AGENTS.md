# GoMLX is machine learning framework (like PyTorch, JAX, TensorFlow, etc.) for Go.

GoMLX is deliberately "non-eager": one builds the computation graph, and then executes it on a backend.

It supports multiple backends, defined under `/backends`. Currently implemented:
- `go`: a pure Go backend, implemented in package `backends/simplego`.
- `xla` (`stablelho` is an alias): uses XLA for acceleration (it JIT-compiles and executes using XLA's PJRT). It is implemented in package `backends/xla`, which in turn uses github.com/gomlx/go-xla to interface with XLA.

Currently it uses "fixed shapes", like XLA. So a different shape requires a dynamic JIT-recompilation. There is work in progress for the Go backend to add support for dynamic shapes.

## File Structure

- `backends`: abstract interface, and their concrete implementations (e.g. `backends/simplego`, `backends/xla`).
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

## Coding Style In GoMLX

### Auto-generated code

Files that start with `gen_` are auto-generated and don't include a copyright line
directly -- the copyright line is in their generators.
Many are created with generators included under `internal/cmd/...`, and the generated file 
includes a comment stating which tool was used to generate them.

### Graph Building Functions

GoMLX works by building computation graphs and then JIT-compiling and executing them in a backend.

The graph building functions (they take `*Node` and `*Context` as arguments, and return updated `*Node` of the graph)
are assumed to be executed sequentially (no concurrency between graph building functions), so no need for mutexes, etc.

Also, different than standard Go, graph building functions return errors by throwing (panicking) instead of returning 
an error, to simplify the code. But everywhere else, use standard Go error handling (by returning an error).

The compiled and execution of the graphs later is parallelized and can be executed in concurrently as one wishes.

Files that mostly define graph building functions, by convention, should dot-import the 
`github.com/gomlx/gomlx/pkg/core/graph` package: having `graph.` repeated everywhere makes the math harder to read.
This is commonly the case for libraries under `pkg/core/ml/layers`.

### Executing models and graphs:

If executing a model with variables, those are stored in `pkg/ml/context.Context` objects, and require `context.Exec` 
object to execute. Simpler graph functions can be simply executed with `pkg/core/graph.Exec` objects. They both
have very similar APIs, just one takes an extra `Context` object with the models variables and parameters.

Creating `Exec` objects implies in JIT-compiling for the first execution: that is expensive. When possible cache 
the executor (`Exec` object) and reuse it.

Also, currently GoMLX doesn't support dynamic shapes. That means each different input shape will trigger a different
compilation. If a program needs different shapes, consider using powers of 2 (or some other base) shapes, and pad
accordingly. Soon `simplego` (the pure Go backend) will support dynamic shapes. But XLA, the faster backend,
only supports static shapes, and will always require recompilation for different shapes. So consider padding
where appropriate.

### Error Handling

All errors should include a stack-trace, using the `github.com/pkg/errors` package.
Whenever printing an error, use `"%+v"` format so the full stack is printed.

Graph building functions return errors by throwing (panicking) instead of returning 
an error, to simplify the code. But everywhere else, use standard Go error handling (by returning an error).

### Modern Go Style

- Use generics where possible.
- Use `slices` and `maps` package for slice operations.
- Look also into `pkg/support/xslices` package for more slice and map helper methods.
- Look into `pkg/support/xsync` package for more syncronization helpers.
- Look into `pkg/support/sets` package for a generic `Set[T]` structure.
- Use iterators (package `iter`) where it makes sense.
- Use the `for range` construct for loops over slices, maps, etc.
- Use `any` instead of `interface{}`.
- Organize tests in hierarchies using `t.Run()` to group related tests.
### Copyright Notes

Normal code files are prefixed with the following copyright line:

```
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
```

Auto-generated files don't need a copyright, but should include a comment with the tool use to generate them.