---
name: golang-gomlx
description: Machine learning models training and inference using GoMLX for Go. It provides an abstraction to create vectorized computation graphs, that can then be JIT-compiled (Just-In-Time) and executed very fast, with backends using XLA (for CPU/CUDA/TPU), Go and others. Includes a reach set of vector (tensors) operations on the graph, a rich ML library with various type of layers, support for training variables, optimizers, training loops, dataset iterators and more. Apply this skill when working ML projects, or needing to do very efficient vectorized (tensor) computations, like image processing, physics or chemestry simulation, etc."
user-invocable: true
license: MIT
compatibility: Designed for Claude Code or similar AI coding agents, and for projects using Golang.
metadata:
  author: samber
  version: "1.1.3"
  openclaw:
    emoji: "💉"
    homepage: https://github.com/samber/cc-skills-golang
    requires:
      bins:
        - go
    install: []
    skill-library-version: "2.0.0"
allowed-tools: Read Edit Write Glob Grep Bash(go:*) Bash(golangci-lint:*) Bash(git:*) Agent WebFetch mcp__context7__resolve-library-id mcp__context7__query-docs
---

**Persona:** You are a Go programmer and Machine Learning practitioner that needs to write, update, code-review a machine learning or vectorial computation task.

# Using GoMLX for Machine Learning or Vectorized Computation 


**Official Resources:**

- [pkg.go.dev/github.com/gomlx/gomlx](pkg.go.dev/github.com/gomlx/gomlx)
- [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)

This skill is not exhaustive. Please refer to library documentation and code examples for more information. 

```bash
go get -u github.com/gomlx/gomlx
```

## Core Concepts

- Shapes and Data Types (DTypes) (`github.com/gomlx/gomlx/pkg/core/dtypes` and `github.com/gomlx/gomlx/pkg/core/shapes`): ...
- Computation Graph (`github.com/gomlx/gomlx/pkg/core/graph`): The `Graph` object...
  - Executor:...
- Backends (`github.com/gomlx/gomlx/backends`): It abstracts backend engines to execute computations on devices (accelerators or the CPU itself). 
  One doesn't need to interact with it directly except if implementing one, just need to pass it around, and know that they exist.
  Usually, one imports (`import _ "github.com/gomlx/gomlx/backends/default"`) to include support for the default backends.
  And the end user can set the environment variable `GOMLX_BACKEND` to specify in runtime a different backend, if they want.
  The default backend uses XLA for GPU/TPU is available.
- Tensors: (`github.com/gomlx/gomlx/tensors`): These represent actual values, that can have local storage or "on-device" (accelerator) storage.
  Usually, they are only used as inputs and outputs of computations, or to save, load or print values. Most methods are about conversion or
  access to the underlying data.
- Context: (`github.com/gomlx/gomlx/pkg/ml/context`): ...

### Creating a graph computation -- package `github.com/gomlx/gomlx/pkg/core/graph`

- Computation building functions usually take only `*graph.Node` as input and outputs. 
- Computation building functions are never concurrent: they are always meant to be executed sequentially. 
  Later the JIT-compiled graph is executed with concurrency, but it's building is always sequential.
- Errors are returned with "execeptions" (panics with an error), not to clutter the math with contant error checking.
  The error should always contain the stacktrace, and preferably use the library `github.com/pkg/errors`.
  The use of exceptions (panics) is only when building graph computations, not for the the other packages.
- They are usually executed only once, or once per input shape -- if we compile the graph for more
  than one shape.
- For files that define large or various computations, it's common practice to "dot import" the `graph` package
  with `import . "github.com/gomlx/gomlx/pkg/core/graph"`, and move all graph computation building functions in its own `.go` file. 

Example:

```go
import . "github.com/gomlx/gomlx/pkg/core/graph"

func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}
```

- Each `Node` has a shape (and dtype). 
  When the shape of the `*Node` is known or fixed, it's often described as a side comment, or asserted (With something like `x.Shape().AssertDims(batchSize, embedDim)`) to make the code easy to read. Inputs or outputs of functions that that take a fixed shape should be documented in the function documentation.
- Notice the graph building is weakly typed for the shapes: so the code doesn't reflet it. 
  But invalid shape operations will raise an exception during the graph building (before the execution).

### Executing a graph -- the `graph.Exec` object

- ... New, Exec, Exec1, Exec... methods.
- Inputs are concrete `tensors.Tensor`, but can be any value that can be converted automatically (so slices or slice or slices).
- The Exec object will automatically recompile the graph, calling again the graph building function, if the shape of the inputs changes.
  It has a limited cache size for different shapes, and compiling a graph is orders of magnitude slower than executing it, so it's better to reuse the same input shapes where possible, using padding to fixed sizes.
- ... ExecOnce ... ExecN


### Tensors

- Local/On-Device ...
- Constructors ...
- Donation for execution...
- Automatic 

### Context: container for variables and hyperparameters -- package `github.com/gomlx/gomlx/pkg/ml/context`

- Scope:...
- `context.Context` has a reference semantics and works as a "current scope path", and can cheaply be copied.
- Variables:....
- Hyperparameters:...
- Checkpointing
- Trainable
- `Exec`: ... similar to `graph.Exec` but takes an extra `context.Context` parameter. Variables can be used and set in the graph building,
  it will handle the passing of its values as inputs and outputs (when updated)...

Example:


### Machine Learning Layers -- package `github.com/gomlx/gomlx/pkg/ml/layers` and sub-packages



### Training loop -- package `github.com/gomlx/gomlx/pkg/ml/train`

- Example from `examples/adult`...

