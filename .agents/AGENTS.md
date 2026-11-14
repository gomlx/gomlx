# GoMLX is machine learning framework (like PyTorch, JAX, TensorFlow, etc.) for Go.

GoMLX is deliberately "non-eager": one builds the computation graph, and then executes it on a backend.

It supports multiple backends (`github.com/gomlx/compute.Backend` interface) to compile and execute the computation
graphs built from the user's GoMLX code. Currently implemented:
- `go`: a pure Go backend, implemented in package `github.com/gomlx/compute/gobackend`.
- `xla`: uses XLA for acceleration (it JIT-compiles and executes using XLA's PJRT). It is implemented in 
  package `github.com/gomlx/go-xla/compute/xla`.

Both these engines are imported by importing the meta-package `import _ "github.com/gomlx/gomlx/backends/default`.

The XLA backend only support static shapes (known in "graph building time"). Currently it uses "fixed shapes", like XLA. So a different shape requires a dynamic JIT-recompilation. There is work in progress for the Go backend to add support for dynamic shapes.

## File Structure

- `core`: core API definitions, to allow building of graphs and their executions, with no reference to trainable 
  variables and such.
  - `core/graph`: defines a `Graph` (a computation grpah building built) and `Node` (represents a value and the output
    of an operation in the `Graph`). It also defines `Exec` that takes a graph building function from the user code as
    a paramter and executes it -- building and JIT-compiling the graph to each new input shape.
  - `core/tensors`: define a `Tensor`, concrete multi-dimensional array (or scalar) with values used as input
    and outputs for graph execution.
- `ml`: ML support libraries, including training, variables handling (`ml/model`), optimizers, layers, datasets, 
  checkpointing, etc.
  - `ml/model`: defines `Variable` (trainable variables/weights) and a `Store` hierarchical container of variables.
    It also defines `Scope`, a pointer to a `Store` setting a "current directory" (scope) for new variables.
    Finally, it defines an `Exec` object that calls a user graph functions and automatically handle passing 
    used variables as "side inputs" and changed variaables as "side outputs" of the execution.
  - `ml/layers`: standard ML layers (Dense, Convolution, MultiHeadAttention, FNN, KAN, Norm, etc.) and post-linear lifecycle hooks (`PostLinearHook`).
  - `ml/layers/peft`: Parameter-Efficient Fine-Tuning (LoRA) implemented as a meta-layer via post-linear hooks.
- `support`: support libraries, including xsync, xerrors, etc.
- `internal`: internal libraries and generators.
  - `internal/cmd`: command line tools: mostly generators used by `go generate ...`. They output `gen_*.go` files.
- `examples`: examples of usage of the library. Some well maintained, others may need to be outdated.
- `ui`: UI tools, for intance to display a training progress bar, or if using GoNB (a Jupyter kernel for Go), to
  dynamically plot a graph with the training progress.
  It is separated in its own top-level directory, to prevend UI packages dependencies to pure GoMLX.
- `.github/workflows`: continuous integration, one file per platform. 

## Related Repositories (including this one)

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: Foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`), shared `dtypes`, `shapes`, `shapeinference`, and backend compliance tests (`compute/support/backendtest`).
  - It also includes the "go" backend: written in pure Go, portable (including WASM), optionally supporting SIMD.
- **`github.com/gomlx/go-xla`**: Depends **only** on `github.com/gomlx/compute` (never on `github.com/gomlx/gomlx`, to avoid circular dependencies).
- **`github.com/gomlx/compute-onnx`**: Implements a ONNX Runtime (ORT) based backend. Supports Dynamic Shapes. Depends **only** on `github.com/gomlx/compute`
- **`github.com/gomlx/go-darwinml`**: Implements the "mps" (Apple's MPSGraph based) and "mlx" (Apple's MLX based) backends. Depends **only** on `github.com/gomlx/compute`.
- **`github.com/gomlx/onnx-gomlx`**: Converts `.onnx` models to GoMLX models. 
- **`github.com/gomlx/go-huggingface`**: Hugging Face models tools: download repo files, iterate over datasets, tokenizer implementations, model implementations (using GoMLX). Currently it supports only a subset of the models ( mainly LLMs for text generation and instruction following) but is expandable.

When navigating the codebase of any of the related projects, read their .agents/AGENTS.md file to get more information about
the structure of the codebase and their specific conventions.

## Coding Style In GoMLX

### Auto-generated code

Files that start with `gen_` are auto-generated and don't include a copyright line
directly -- the copyright line is in their generators.
Many are created with generators included under `internal/cmd/...`, and the generated file 
includes a comment stating which tool was used to generate them.

### Graph Building Functions

GoMLX works by building computation graphs and then JIT-compiling and executing them in a backend.

The graph building functions take `*graph.Node`, `*graph.Graph` and/or `*model.Scope` (for variables) as arguments, 
and return updated `*graph.Node` of the graph.

Graph building functions are assumed to be executed sequentially (no concurrency between graph building functions), 
so no need for mutexes, etc. Once compiled they can be executed in parallel.

Also, different than standard Go, graph building functions return errors by throwing (panicking) instead of returning 
an error, to simplify the code. But everywhere else, use standard Go error handling (by returning an error).

The compiled and execution of the graphs later is parallelized and can be executed in concurrently as one wishes.

Files that mostly define graph building functions, by convention, should dot-import the 
`github.com/gomlx/gomlx/core/graph` package: having `graph.` repeated everywhere makes the math harder to read.
This is commonly the case for libraries under `ml/layers`.

### Testing

Never attempt to test everything (`go test ./...`)! Many of the tests are very long, and parallel testing 
doesn't work (because the XLA backend for CUDA allocates all the memory on startup, so there can't be more
than one instance at a time).

If you are testing more than one package at a time (using ellipsis "..."), remember to use the flag `-p 1` (e.g., `go test -p 1 ./ml/model/...`), since creating more than one CUDA backend simultaneously fails.

Instead test only one package at a time, or even, preferably, only the tests related to the code you are 
currently modifying (`go test <./.../package> -run <test_pattern>`).

### Executing models and graphs:

If executing a model with variables, those are stored in `ml/model.Store` objects (they can also be referenced by
`model.Scope` objects, pointers into a `Store` with a scope), and require `model.Exec` object to execute. 

Graph functions that do not use variables (i.e. do not use `model.Scope`) can be simply executed with `core/graph.Exec`
objects. They both have very similar APIs, just one takes an extra `model.Store` object with the models variables and
parameters.

Creating `Exec` objects implies in JIT-compiling for the first execution: that is expensive. Always reuse 
the executor (`Exec` object) when possible.

`graph.Exec.CompileUncached` and `model.Exec.CompileUncached` build a fresh graph with the current
configuration for one-off operations such as ONNX export. The caller owns and finalizes that graph.

Also, currently GoMLX doesn't support dynamic shapes. That means each different input shape will trigger a different
compilation. If a program needs different shapes, consider using powers of 2 (or see
`github.com/gomlx/compute/support.TwoBitBucketLen`) shapes, and pad accordingly. The `compute.Backend` API already
support dynamic shapes, but only the Go backend (`github.com/gomlx/compute/gobackend`) partially implements it for now.
We plan to add support for dynamic shapes (input shaped dependent) soon in GoMLX.


### Error Handling

All errors should include a stack-trace, using the `github.com/pkg/errors` package. Whenever printing an error, use
`"%+v"` format so the full stack is printed.

Graph building functions return errors by throwing (panicking) instead of returning an error, to simplify the code. But
everywhere else, use standard Go error handling (by returning an error).

There is a support library `support/exceptions` with helper functions like `exceptions.Panicf`, a wrapper for 
`panic(errors.Errorf(...))`, which automatically adds `fmt.Sprintf()` formatting and `errors.WithStack()`; and 
`Try`, `TryCatch` to capture exceptions as errors.

### Modern Go Style

- Use generics where possible.
- Use `slices` and `maps` package for slice operations.
- Look also into the following support packages used often (expand them when needed):
  - `github.com/gomlx/support/xslices`: more slices and maps (e.g.: `SortedKeys()`) helper methods.
  - `github.com/gomlx/support/xsync`: `Latch`, `Semaphore`, `SyncMap` (generic version), and `DynamicWaitGroup` (count
    can be changed while a goroutine is waiting on it)
  - `github.com/gomlx/support/sets`: `Set[T]` structure and helper methods.
  - `github.com/gomlx/support/humanize`: `Bytes(n)`, `Underscores(n)`, `Count(n)` (powers of 1000), `Speed(r, u)`
    (ratio/second), and `Duration(d)`.
  - `github.com/gomlx/support/envutil`: `ReadBool`, `ReadInt`, etc.
  - `./support/exceptions`: exception handling with `Panicf`, `Try`, `TryCatch`.
  - `./support/fsutil`: file system utilities: `FileExists`, `ReplaceTildeInDir`, `ValidateChecksum`, `ReportedClose`.   
- Use iterators (package `iter`) where it makes sense.
- Use the `for range` construct for loops over slices, maps, etc.
- Use `any` instead of `interface{}`.
- Organize tests in hierarchies using `t.Run()` to group related tests.

### Follow Existing Patterns

Before writing new code, read neighboring files in the same package to understand the established
patterns (buffer management, dtype dispatch, parallelization, etc.). Reuse existing infrastructure
rather than writing ad-hoc implementations. When in doubt, match the style and approach of the
closest existing operation.

### Copyright Notes

Normal code files are prefixed with the following copyright line:

```
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
```

Auto-generated files don't need a copyright, but should include a comment with the tool use to generate them.


### Documentation & Git Workflow

- Record user-visible or API changes in `docs/CHANGELOG.md` (with `docs/CHANGELOG-pjrt.md` and `docs/CHANGELOG-stablehlo.md` kept as historical archives).
- Keep `.agents/AGENTS.md` (symlinked from `GEMINI.md`) updated whenever architecture, conventions, or non-obvious behavior changes.
- **Never `git commit` or `git push`** without explicit user instruction; leave changes uncommitted in the working tree for review.
- When changing code, if not explicitly asked, always wait for the operator to review the changes before committing them.
- For every new feature, refactoring or API change, update the files under `.agents/skills/golang-gomlx/` accordingly.

### Git Workflow

Never `git commit` or `git push` changes before the user has reviewed them and explicitly asked to commit. Always leave changes uncommitted in the working tree for review.
