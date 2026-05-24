---
name: golang-gomlx
description: "Machine learning models training and inference using GoMLX for Go. It provides an abstraction to create vectorized computation graphs, that can then be JIT-compiled (Just-In-Time) and executed very fast, with backends using XLA (for CPU/CUDA/TPU), Go and others. Includes a reach set of vector (tensors) operations on the graph, a rich ML library with various type of layers, support for training variables, optimizers, training loops, dataset iterators and more. Apply this skill when working ML projects, or needing to do very efficient vectorized (tensor) computations, like image processing, physics or chemestry simulation, etc."
user-invocable: true
license: MIT
compatibility: Designed for Claude Code or similar AI coding agents, and for projects using Golang.
metadata:
  author: janpfeifer
  version: "1.0.0"
  openclaw:
    emoji: "🤖"
    homepage: https://github.com/gomlx/gomlx
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

- GoMLX:
  - [pkg.go.dev/github.com/gomlx/gomlx](https://pkg.go.dev/github.com/gomlx/gomlx)
  - [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)
- Compute Backend API and Go backend implementation:
  - [pkg.go.dev/github.com/gomlx/compute](https://pkg.go.dev/github.com/gomlx/compute)
  - [github.com/gomlx/compute](https://github.com/gomlx/compute)

This skill is not exhaustive. Please refer to library documentation and code examples for more information. 

```bash
go get -u github.com/gomlx/gomlx
go get -u github.com/gomlx/compute
```

## Core Concepts

- Shapes and Data Types (DTypes) (`github.com/gomlx/gomlx/compute/dtypes` and `github.com/gomlx/compute/shapes`):
  - `dtypes` define the underlying type of the data (e.g. `dtypes.Float32`, `dtypes.Int64`, `dtypes.Bool`).
  - `shapes.Shape` represents the multi-dimensional structure of a tensor, including its `DType` and its `Dimensions` (a
    slice of integers). Shapes are strictly checked during graph building. It also has experimental support for "dynamic
    shapes" (dependend on the input shape), where axis may be set as indeterminate (`shapes.DynamicDim` == -1), and
    named (`DynamicDim` axes must be named). But only the Go backend partially supports dynamic shapes, XLA doesn't.
- Computation Graph (`github.com/gomlx/gomlx/core/graph`): The `Graph` object is the container for computation nodes.
  - Computations are built (by a Go function) using `*Node` objects. Each node represents an operation or a value, and
    it always contain a reference to the graph it belongs to (`Node.Graph()`)
  - The graph building phase is separate from the execution phase. You build the graph first, and then execute it (there
    is a JIT-compilation that happens in between automatically, if handled by the `graph.Exec` object).
  - Executor: (`graph.Exec` or `model.Exec`) takes a graph-building function, JIT-compiles it, and provides methods to
    execute it.
- `compute.Backend` (github.com/gomlx/compute): It abstracts backend engines to execute computations on devices
  (accelerators or the CPU itself). One doesn't need to interact with it directly except if implementing one. One just
  needs to pass around the `compute.Backend` object in use. Usually, one imports (`import _
  "github.com/gomlx/gomlx/backends/default"`) to include support for the default backends. And the end user can set the
  environment variable `GOMLX_BACKEND` to specify in runtime a different backend, if they want. The default backend uses
  XLA for GPU/TPU is available, typical values would be: "go" (for the portable Go backend), "xla:cpu" for the XLA CPU
  backend, "xla:cuda" for the XLA NVIDIA GPU backend, and "xla:tpu" for the XLA TPU backend.
  Only the Go backend partially supports dynamic shapes, XLA doesn't.
- Tensors: (`github.com/gomlx/gomlx/core/tensors`): These represent actual values, that can have local storage or
  "on-device" (accelerator) storage. Usually, they are only used as inputs and outputs of computations, or to save, load
  or print values. Most methods are about conversion or access to the underlying data (e.g., `tensor.Value()` returns a
  generic value, or `tensor.Local().Copy()` for moving back to CPU memory).

- `model.Store`, `model.Scope`, `model.Exec` (github.com/gomlx/gomlx/ml/model): The `model` package introduces 
  `Variable` (representing model weights) and hyperparameters abstractions, organized in a "directory-like" tree.
  The `model.Store` is the container for a model's variable and it's passed around if the graph computation being built
  uses them (true for all ML models). The `model.Scope` is what is passed around, it contains a reference to the `Store`
  and a "scope" (similar to `current directory'), that helps in organizing the variables hierarchically. One can 
  enter nested scopes (sub-scopes) when constructing model layers.
  - `model.Exec`: it uses `graph.Exec` and has a very similar API, but it takes a `model.Store` as a construction
    argument and automatically adds used variables as "side-inputs" to the build computation graph, and modified 
    variables as "side-outputs". The variables values are automatically input/updated during the execution.

### Creating a graph computation -- package `github.com/gomlx/core/graph`

- Computation building functions usually take only `*Node` as input and outputs.
- Computation building functions are never concurrent: they are always meant to be executed sequentially. Later the
  JIT-compiled graph is executed with concurrency, but its building is always sequential.
- Errors are returned with "execeptions" (panics with an error), to not clutter the "math-y" code with constant error
  checking. The error should always contain the stacktrace, and preferably use the library `github.com/pkg/errors`. The
  use of exceptions (panics) is only when building graph computations, not for the the other packages. See
  `execptions.Panicf(format, args...)` (github.com/gomlx/gomlx/support/exceptions) for a convenient wrapper around
  `panic(errors.Errorf(format, args...))`.
- Graph building functions are usually executed only once, or once per input shape -- if we compile the graph for more
  than one shape (by calling `Exec.Call` more than once with different input shapes).
- For files that define large or various computations, it's common practice to "dot import" the `graph` package
  with `import . "github.com/gomlx/gomlx/core/graph"`, and move all graph computation building functions in its own `.go` file. 
- **See [`graph` package reference](./references/graph.md)** for a list of common functions and their PyTorch equivalents.

Example:

```go
import . "github.com/gomlx/gomlx/core/graph"

func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}
```

- Each `Node` has a shape (and dtype). When the shape of the `*Node` is known or fixed, it's often described as a side
  comment, or asserted (With something like `x.Shape().AssertDims(batchSize, embedDim)`) to make the code easy to read.
  Inputs or outputs of functions that that take a fixed shape should be documented in the function documentation.
- Notice the graph building is weakly typed for the shapes: so the code doesn't reflet it. But invalid shape operations
  will raise an exception during the graph building (before the execution).

### Executing a graph -- the `graph.Exec` object

- It is created with `graph.NewExec(backend, fn)`, where `fn` is the graph-building function.
- `exec.Call(inputs...)` is used to execute the compiled graph, taking `tensors.Tensor` or standard Go values (slices of
  slices) and returning `tensors.Tensor`.
- Inputs are concrete `tensors.Tensor`, but can be any value that can be converted automatically (so slices or slice or
  slices).
- The Exec object will automatically recompile the graph, calling again the graph building function, if the shape of the
  inputs changes. It has a limited cache size for different shapes, and compiling a graph is orders of magnitude slower
  than executing it, so it's better to reuse the same input shapes where possible, using padding to fixed sizes.


### Tensors -- package `github.com/gomlx/gomlx/core/tensors`

- Local/On-Device: Tensors can be instantiated on the local CPU (`tensors.FromValue(...)`) or directly on the backend
  device device (usually happens automatically for outputs of executions).
- Constructors: Use `tensors.FromValue(any)` or `tensors.FromShape(shape)` to create tensors.
- Donation for execution: You can "donate" a tensor to an execution to allow XLA to reuse its memory for outputs using
  `exec.Call(input1, input2)`. The donated tensor's memory will be overwritten, so it shouldn't be used afterward.

### Machine Learning Models: variables, hyperparameters, store and containers -- package `github.com/gomlx/gomlx/ml/model`

- `model.Store`: A container for a model's variables and hyperparameters, organized hierarchicaly, like a directory tree. It is passed around if the graph
  computation being built uses them (true for all ML models).
- `model.Scope`: Represents a reference to a `model.Store` (returned by `Scope.Store()`) with a scope ("current
   directory"). You can enter nested scopes (sub-scopes) as one is building a model layers, organized hierarchicaly:
   - `Scope.In(format, args...)`: enters a nested scope, allowing only one visit per sub-scope -- reusing a scope 
     triggers an error (panic). This is the usual method, and the check helps avoiding mistakes.
   - `Scope.Shared(format, args...)` to re-enter a scope, and calling it to enter a newly visited sub-scope is an error. 
     E.g.: to reuse the weights in a siamese tower model)
   - `Scope.At(format, args...)` if one wants to enter a sub-scope without regards if it has been visited before or not.
- Variables: Are created using `Scope.VariableWithValue(name, value)` or `Scope.VariableWithShape(name, shape)`. 
  Once created, they persist in the underlying `Store` and can be retrieved using `Scope.InspectVariable(name)`.
  One can also use the `Store` directly to retrieve variables using the full path to them (as opposed to variables
  in the current scope).
- Hyperparameters: Set with `Scope.SetParam("key", value)` and retrieved with 
  `model.GetParamOr(scope, "key", defaultValue)`.
- Checkpointing (saving/loading): `checkpoint.Build(store)` (github.com/gomlx/gomlx/ml/model/checkpoint) helps save
  and load the state of all variables in a `model.Store`.
- Trainable: Variables are by default trainable.
- `model.Exec`: it uses `graph.Exec` and has a very similar API, but it takes a `model.Store` as a construction
  argument and automatically adds used variables as "side-inputs" to the build computation graph, and modified
  variables as "side-outputs". The variables values are automatically input/updated during the execution.

Example:

```go
func DenseLayer(scope *model.Scope, x *Node, outputDim int) *Node {
  g := x.Graph()
	inputDim := x.Shape().Dimensions[len(x.Shape().Dimensions)-1]
	weightsVar := scope.VariableWithShape("weights", shapes.Make(x.DType(), inputDim, outputDim))
	biasVar := scope.VariableWithShape("bias", shapes.Make(x.DType(), outputDim))
	x = Dot(x, weightsVar.NodeValue(g)).Product()
	return Add(x, biasVar.NodeValue(g))
}
```

### Machine Learning Layers -- package `github.com/gomlx/gomlx/ml/layers` and sub-packages

- The `layers` package provides standard higher-level building blocks for ML models.
- Uses `*model.Scope` extensively to manage the weights/biases for each layer.
- Sub-packages include `activation` (Relu, Swish, etc.), `fnn` (feed-forward neural networks), `kan` (Kolmogorov-Arnold Networks), `regularizer`, `norm`, etc.
- **See [`layers` package reference](./references/layers.md)** for a list of common layers and their PyTorch equivalents.

### Training loop -- package `github.com/gomlx/gomlx/ml/train`

- Example from `examples/adult/demo`: Shows a full ML pipeline.
- `train.Trainer` orchestrates the model function, the loss function, and the optimizer.
  - Needs `model.Store`, a model function, a loss function (e.g., `loss.BinaryCrossentropyLogits`), and an optimizer (e.g., `optimizer.Adam`).
- Metrics (`ml/train/metric`): Used to evaluate model performance during training and evaluation.
  - Metrics are provided as lists during `train.NewTrainer` initialization (one list for train metrics, one for eval metrics).
  - Common metrics include `metric.NewMeanBinaryLogitsAccuracy()`, `metric.NewSparseCategoricalAccuracy()`.
- `train.Loop` manages the iterative process, feeding datasets to the `Trainer` and calling callbacks (e.g., checkpoint saving, plotting).

Example Training Pipeline:

```go
// Create an empty store for the variables and hypeparameters.
store := model.NewStore()
store.SetParam("learning_rate", *flagLearningRate)

// Create dataset
trainDS := CreateDataset(...)

// Metrics we are interested in.
meanAccuracyMetric := metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
movingAccuracyMetric := metric.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

// Create a train.Trainer: orchestrates running the model, feeding results to the optimizer, evaluating metrics.
trainer := train.NewTrainer(backend, store, Model, loss.BinaryCrossentropyLogits,
	optimizer.FromStore(store),
	[]metric.Interface{movingAccuracyMetric}, // trainMetrics
	[]metric.Interface{meanAccuracyMetric})   // evalMetrics

// Create a standard training loop
loop := train.NewLoop(trainer)

// Attach a progress bar to the loop.
commandline.AttachProgressBar(loop)

// Get hyperparameters and run the training loop
trainSteps := model.GetRootParamOr(store, "train_steps", 1000)
_, err := loop.RunToGlobalStep(trainDS, trainSteps)
if err != nil {
	return err
}
```

