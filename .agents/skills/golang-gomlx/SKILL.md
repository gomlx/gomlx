---
name: golang-gomlx
description: "Machine learning models training and inference using GoMLX for Go. It provides an abstraction to create vectorized computation graphs, that can then be JIT-compiled (Just-In-Time) and executed very fast, with backends using XLA (for CPU/CUDA/TPU), Go and others. Includes a reach set of vector (tensors) operations on the graph, a rich ML library with various type of layers, support for training variables, optimizers, training loops, dataset iterators and more. Apply this skill when working ML projects, or needing to do very efficient vectorized (tensor) computations, like image processing, physics or chemestry simulation, etc."
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

- [pkg.go.dev/github.com/gomlx/gomlx](https://pkg.go.dev/github.com/gomlx/gomlx)
- [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)

This skill is not exhaustive. Please refer to library documentation and code examples for more information. 

```bash
go get -u github.com/gomlx/gomlx
```

## Core Concepts

- Shapes and Data Types (DTypes) (`github.com/gomlx/gomlx/pkg/core/dtypes` and `github.com/gomlx/gomlx/pkg/core/shapes`):
  - `dtypes` define the underlying type of the data (e.g. `dtypes.Float32`, `dtypes.Int64`, `dtypes.Bool`).
  - `shapes.Shape` represents the multi-dimensional structure of a tensor, including its `DType` and its `Dimensions` (a slice of integers). Shapes are strictly checked during graph building.
- Computation Graph (`github.com/gomlx/gomlx/pkg/core/graph`): The `Graph` object is the container for computation nodes.
  - Computations are built using `*Node` objects. Each node represents an operation or a value.
  - The graph building phase is separate from the execution phase. You build the graph first, then execute it.
  - Executor: (`graph.Exec` or `context.Exec`) takes a graph-building function, JIT-compiles it, and provides methods to execute it.
- Backends (`github.com/gomlx/gomlx/backends`): It abstracts backend engines to execute computations on devices (accelerators or the CPU itself). 
  One doesn't need to interact with it directly except if implementing one, just need to pass it around, and know that they exist.
  Usually, one imports (`import _ "github.com/gomlx/gomlx/backends/default"`) to include support for the default backends.
  And the end user can set the environment variable `GOMLX_BACKEND` to specify in runtime a different backend, if they want.
  The default backend uses XLA for GPU/TPU is available.
- Tensors: (`github.com/gomlx/gomlx/pkg/core/tensors`): These represent actual values, that can have local storage or "on-device" (accelerator) storage.
  Usually, they are only used as inputs and outputs of computations, or to save, load or print values. Most methods are about conversion or
  access to the underlying data (e.g., `tensor.Value()` returns a generic value, or `tensor.Local().Copy()` for moving back to CPU memory).
- Context: (`github.com/gomlx/gomlx/pkg/ml/context`): A container for stateful variables (like model weights) and hyperparameters. It has reference semantics.

### Creating a graph computation -- package `github.com/gomlx/gomlx/pkg/core/graph`

- Computation building functions usually take only `*Node` as input and outputs.
- Computation building functions are never concurrent: they are always meant to be executed sequentially. 
  Later the JIT-compiled graph is executed with concurrency, but it's building is always sequential.
- Errors are returned with "execeptions" (panics with an error), not to clutter the math with contant error checking.
  The error should always contain the stacktrace, and preferably use the library `github.com/pkg/errors`.
  The use of exceptions (panics) is only when building graph computations, not for the the other packages.
- They are usually executed only once, or once per input shape -- if we compile the graph for more
  than one shape.
- For files that define large or various computations, it's common practice to "dot import" the `graph` package
  with `import . "github.com/gomlx/gomlx/pkg/core/graph"`, and move all graph computation building functions in its own `.go` file. 
- **See [`graph` package reference](./references/graph.md)** for a list of common functions and their PyTorch equivalents.

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

- It is created with `graph.NewExec(backend, fn)`, where `fn` is the graph-building function.
- `exec.Call(inputs...)` is used to execute the compiled graph, taking `tensors.Tensor` or standard Go values (slices of slices) and returning `tensors.Tensor`.
- Inputs are concrete `tensors.Tensor`, but can be any value that can be converted automatically (so slices or slice or slices).
- The Exec object will automatically recompile the graph, calling again the graph building function, if the shape of the inputs changes.
  It has a limited cache size for different shapes, and compiling a graph is orders of magnitude slower than executing it, so it's better to reuse the same input shapes where possible, using padding to fixed sizes.


### Tensors -- package `github.com/gomlx/gomlx/pkg/core/tensors`

- Local/On-Device: Tensors can be instantiated on the local CPU (`tensors.FromValue(...)`) or directly on the backend device device (usually happens automatically for outputs of executions).
- Constructors: Use `tensors.FromValue(any)` or `tensors.FromShape(shape)` to create tensors.
- Donation for execution: You can "donate" a tensor to an execution to allow XLA to reuse its memory for outputs using `exec.Call(input1, input2)`. The donated tensor's memory will be overwritten, so it shouldn't be used afterward.

### Context: container for variables and hyperparameters -- package `github.com/gomlx/gomlx/pkg/ml/context`

- Scope: A context represents a hierarchical scope (e.g., `/model/layer1`). You can enter sub-scopes via `ctx.In("layer1")`.
- `context.Context` has a reference semantics and works as a "current scope path", and can cheaply be copied.
- Variables: Are created using `ctx.VariableWithValue(name, value)` or `ctx.VariableWithShape(name, shape)`. Once created, they persist in the context and can be retrieved using `ctx.InspectVariable(...)`.
- Hyperparameters: Set with `ctx.SetParam("key", value)` and retrieved with `context.GetParamOr(ctx, "key", defaultValue)`.
- Checkpointing: `checkpoints.New(ctx, dir)` helps save and load the state of all variables in a context.
- Trainable: Variables are by default trainable.
- `Exec`: `context.Exec` is similar to `graph.Exec` but designed for ML models. Its builder function takes an extra `*context.Context` parameter. Variables can be used and set in the graph building, and `context.Exec` will handle passing their values as inputs and outputs.

Example:

```go
func DenseLayer(ctx *context.Context, x *Node, outputDim int) *Node {
	inputDim := x.Shape().Dimensions[len(x.Shape().Dimensions)-1]
	weightsVar := ctx.VariableWithShape("weights", shapes.Make(x.DType(), inputDim, outputDim))
	biasVar := ctx.VariableWithShape("bias", shapes.Make(x.DType(), outputDim))

	x = Dot(x, weightsVar.ValueGraph(x.Graph())).Product()
	return Add(x, biasVar.ValueGraph(x.Graph()))
}
```

### Machine Learning Layers -- package `github.com/gomlx/gomlx/pkg/ml/layers` and sub-packages

- The `layers` package provides standard higher-level building blocks for ML models.
- Uses `*context.Context` extensively to manage the weights/biases for each layer.
- Sub-packages include `activations` (Relu, Swish, etc.), `fnn` (feed-forward neural networks), `kan` (Kolmogorov-Arnold Networks), `regularizers`, etc.
- **See [`layers` package reference](./references/layers.md)** for a list of common layers and their PyTorch equivalents.

### Training loop -- package `github.com/gomlx/gomlx/pkg/ml/train`

- Example from `examples/adult/demo`: Shows a full ML pipeline.
- `train.Trainer` orchestrates the model function, the loss function, and the optimizer.
  - Needs `context.Context`, a model function, a loss function (e.g., `losses.BinaryCrossentropyLogits`), and an optimizer (e.g., `optimizers.Adam`).
- Metrics (`pkg/ml/train/metrics`): Used to evaluate model performance during training and evaluation.
  - Metrics are provided as lists during `train.NewTrainer` initialization (one list for train metrics, one for eval metrics).
  - Common metrics include `metrics.NewMeanBinaryLogitsAccuracy()`, `metrics.NewSparseCategoricalAccuracy()`.
- `train.Loop` manages the iterative process, feeding datasets to the `Trainer` and calling callbacks (e.g., checkpoint saving, plotting).

Example Training Pipeline:

```go
// 1. Create dataset
trainDS := CreateDataset(...)

// 2. Metrics we are interested in.
meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

// 3. Create a train.Trainer: orchestrates running the model, feeding results to the optimizer, evaluating metrics.
trainer := train.NewTrainer(backend, ctx, Model, losses.BinaryCrossentropyLogits,
	optimizers.FromContext(ctx),
	[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
	[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

// 4. Create a standard training loop
loop := train.NewLoop(trainer)

// 5. Attach a progress bar to the loop.
commandline.AttachProgressBar(loop)

// 6. Get hyperparameters and run the training loop
trainSteps := context.GetParamOr(ctx, "train_steps", 1000)
_, err := loop.RunToGlobalStep(trainDS, trainSteps)
if err != nil {
	return err
}
```

