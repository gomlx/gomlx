---
name: golang-gomlx
description: "Machine learning models training and inference using GoMLX for Go. It provides an abstraction to create vectorized computation graphs, that can then be JIT-compiled (Just-In-Time) and executed very fast, with backends using XLA (for CPU/CUDA/TPU), Go (portable and fast if using SIMD), ONNX Runtime, Apple MPS/MLX, and others. Includes a rich set of vector (tensor) operations on the graph, a rich ML library with various types of layers, support for training variables, optimizers, training loops, dataset iterators, and companion tools for HuggingFace and ONNX models. Apply this skill when working on ML projects or needing very efficient vectorized (tensor) computations, such as image processing, physics, or chemistry simulation."
user-invocable: true
license: MIT
compatibility: Designed for Claude Code or similar AI coding agents, and for projects using Golang.
metadata:
  author: janpfeifer
  version: "0.28.0"
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

**Persona:** You are a Go programmer and Machine Learning practitioner who needs to write, update, or code-review a machine learning or vectorized computation task.

# Using GoMLX for Machine Learning or Vectorized Computation

**Official Resources:**

- GoMLX:
  - [gomlx.github.io](https://gomlx.github.io/) - GoMLX Documentation
  - [pkg.go.dev/github.com/gomlx/gomlx](https://pkg.go.dev/github.com/gomlx/gomlx)
  - [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)
- Compute Backend API and Go backend implementation:
  - [pkg.go.dev/github.com/gomlx/compute](https://pkg.go.dev/github.com/gomlx/compute)
  - [github.com/gomlx/compute](https://github.com/gomlx/compute)
- Related Projects:
  - [github.com/gomlx/go-huggingface](https://github.com/gomlx/go-huggingface): downloading/caching HuggingFace repo files (remote, local, or `//go:embed`), pure-Go tokenizers and length bucketing, on-the-fly HuggingFace model conversion to GoMLX (SafeTensors, GGUF, transformers, SAM2, Timestral — plus ONNX via `onnx-gomlx`), and streaming Parquet dataset iteration. See [`go-huggingface` reference](./references/go-huggingface.md).
  - [github.com/gomlx/onnx-gomlx](https://github.com/gomlx/onnx-gomlx): converting `.onnx` models on-the-fly into GoMLX computation graphs for inference, graph composition, dynamic shapes, fused-op acceleration, and fine-tuning (including re-exporting updated weights back to `.onnx`). See [`onnx-gomlx` reference](./references/onnx-gomlx.md).

This skill is not exhaustive. Refer to the linked references, library documentation, and code examples for full details.

```bash
go get -u github.com/gomlx/gomlx
go get -u github.com/gomlx/compute
```

## Core Concepts

- **Shapes and Data Types (`github.com/gomlx/compute/dtypes` and `github.com/gomlx/compute/shapes`)**:
  - `dtypes` defines the underlying element type (e.g., `dtypes.Float32`, `dtypes.Int64`, `dtypes.Bool`).
  - `shapes.Shape` represents the multi-dimensional structure (`DType` + `Dimensions`), strictly checked during graph building. GoMLX also supports input-conditioned **dynamic shapes** (`shapes.DynamicDim` == `-1`, optionally named via `shapes.MakeDynamic(...)`).
- **Computation Graph (`github.com/gomlx/gomlx/core/graph`) & Autodiff (`github.com/gomlx/gomlx/core/autodiff`)**:
  - Computations are built by a Go function operating on `*graph.Node` values. Each node references its owning graph (`node.Graph()`), and parameters can be looked up by name via `g.GetParam(name)`.
  - Reverse-mode automatic differentiation is provided by `autodiff.Gradient(loss, targets...)`.
  - Graph building is separate from execution: `graph.Exec` / `model.Exec` JIT-compiles the built graph automatically before running it.
- **`compute.Backend` (`github.com/gomlx/compute`)**:
  - Abstracts execution devices (CPU/GPU/TPU). Typically import `import _ "github.com/gomlx/gomlx/backends/default"` and create with `compute.MustNew()`.
  - Override at runtime via `GOMLX_BACKEND`: `"go"` (portable pure-Go and fast if using SIMD), `"onnx"` / `"onnx:cuda"` (ONNX Runtime), `"xla:cpu"` / `"xla:cuda"` / `"xla:tpu"`, `"mps"` / `"mlx"` / `"mlx:dynamic"` (Apple Silicon).
- **Tensors (`github.com/gomlx/gomlx/core/tensors`)**:
  - Concrete values stored locally on CPU or on-device, created with `tensors.FromValue(any)` or `tensors.FromShape(shape)`.
  - Used for inputs/outputs, saving/loading, or inspection (`tensor.Value()`, `tensor.Local().Copy()`).
- **`model.Store`, `model.Scope`, `model.Exec` (`github.com/gomlx/gomlx/ml/model`)**:
  - `model.Store` holds model variables (weights) and hyperparameters in a hierarchical tree.
  - `model.Scope` pairs a `Store` reference with a current path ("directory") for organizing nested layers.
  - `model.Exec` wraps `graph.Exec` and automatically wires used variables as side-inputs and updated variables as side-outputs.

---

### Creating a Graph Computation -- `github.com/gomlx/gomlx/core/graph`

- Graph-building functions take `*Node` inputs/outputs and run **sequentially** (only once per unique input shape). Once compiled, the resulting `Exec` can be called concurrently.
- Errors during graph building are raised via exceptions (panics wrapping an error with stack trace via `github.com/pkg/errors` and `exceptions.Panicf` from `github.com/gomlx/gomlx/support/exceptions`) to keep math code uncluttered. Outside graph building, use standard Go error returns.
- For files defining large computations, dot-importing `import . "github.com/gomlx/gomlx/core/graph"` in a dedicated `.go` file is idiomatic.
- Document fixed shapes in comments or assert them with `x.Shape().AssertDims(batchSize, embedDim)`; invalid shape operations fail at graph-build time.
- **See [`graph` package reference](./references/graph.md)** for common operations and PyTorch equivalents.

```go
import . "github.com/gomlx/gomlx/core/graph"

func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}
```

### Executing a Graph -- `graph.Exec` and `model.Exec`

- Create with `graph.NewExec(backend, fn)` (stateless) or `model.NewExec(backend, store, fn)` (with variables), and run via `exec.Call(inputs...)` / `exec.MustCall1(...)` passing `*tensors.Tensor` or standard Go slices.
- `Exec` caches compiled graphs per concrete input shape; compilation is orders of magnitude slower than execution, so reuse `Exec` instances and avoid unbounded shape variation on static-shape backends.
- For one-off compilation (e.g., ONNX export via `onnx.Save()`), `exec.CompileStandalone(inputShapes...)` builds a fresh `*Graph` owned by the caller (who must call `Graph.Finalize()`).

### Dynamic Shapes (Input-Conditioned Shapes)

- Backends with `backend.Capabilities().HasDynamicShapes()` (e.g., `go`, `onnx`, `mlx:dynamic`) execute variable dimensions (batch size, sequence length) without recompilation when configured with `exec.WithDynamicAxes([]string{"batch", "seq"}, ...)`.
- Write polymorphic graphs using `DimensionSpecFor`, `DimensionSize`, `DynamicReshape`, `BroadcastLike`, and `IotaLike` (zero overhead on static graphs). Note that implicit broadcasting (`Add`, `Mul`, `Where`) is strictly compile-time.
- On static-shape backends (`xla`), pad/bucket inputs to discrete sizes (e.g., `github.com/gomlx/go-huggingface/tokenizers/bucket`).
- **See [`dynamic-shapes` reference](./references/dynamic-shapes.md)** for capabilities, dynamic ops, broadcasting rules, and bucketing guidance.

### Variables & Scopes -- `github.com/gomlx/gomlx/ml/model`

- Navigate sub-scopes via:
  - `scope.In(format, args...)`: enters a nested scope (panics if visited more than once — catches accidental weight sharing).
  - `scope.Shared(format, args...)`: re-enters an already-visited scope (e.g., Siamese towers).
  - `scope.At(format, args...)`: enters a sub-scope regardless of prior visits.
- Create variables with `scope.VariableWithValue(name, val)` or `scope.VariableWithShape(name, shape)` (trainable by default), and inspect with `scope.InspectVariable(name)`.
- Hyperparameters: set via `scope.SetParam("key", value)` and read via `model.GetParamOr(scope, "key", defaultVal)`.

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

### Machine Learning Layers -- `github.com/gomlx/gomlx/ml/layers`

- Provides standard ML building blocks using `*model.Scope` (`layers.Dense`, `layers.Convolution`, `layers.Dropout`, `activation`, `fnn`, `kan`, `norm`, `attention`, `regularizer`, `peft`), plus stateless fused primitives in `ml/nn`.
- **PEFT / LoRA**: `import _ "github.com/gomlx/gomlx/ml/layers/peft"` injects zero-code-change LoRA adapters into `Dense`, `fnn`, and `MultiHeadAttention` via hyperparameters (`peft_adapter`, `peft_rank`, `peft_alpha`, `peft_target_modules`).
- **See [`layers` package reference](./references/layers.md)** for layers, losses, optimizers, and PyTorch equivalents.

### Training Loop & Checkpointing -- `github.com/gomlx/gomlx/ml/train`

- `train.Trainer` ties together `backend`, `model.Store`, model function, loss (`ml/train/loss`), optimizer (`ml/train/optimizer`), and train/eval metrics (`ml/train/metric`).
- `train.Loop` iterates datasets, calls `Trainer`, and supports lifecycle hooks (`OnStart`, `OnFirstStep`, `OnStep`, `OnEnd`), progress bars (`commandline.AttachProgressBar`), and checkpointing (`checkpoint.Build(store)`).
- **See [`training` reference](./references/training.md)** for a full training pipeline example, checkpointing, and PEFT/LoRA details.

---

## Supporting Ecosystem Repositories

### Importing ONNX Models -- `github.com/gomlx/onnx-gomlx`

- Converts `.onnx` files on-the-fly into GoMLX computation graphs (`onnx/parser.ParseFile` / `Parse` / `ParseReader`), supporting any GoMLX backend, dynamic shapes (`exec.WithDynamicAxes`), custom pre-/post-processing composition, automatic fused-op matching (SDPA, QKV, Dense+GELU, quantized dense/MatMulNBits, LayerNorm), and fine-tuning with GoMLX autodiff + exporting updated weights back to `.onnx` (`ScopeToONNX`, `SaveToFile`/`Write`).
- **See [`onnx-gomlx` reference](./references/onnx-gomlx.md)** for workflows, API summary, external data readers, and debugging tips.

### HuggingFace Hub, Tokenizers, Models & Datasets -- `github.com/gomlx/go-huggingface`

- **Repository access (`hub`)**: Download/cache files sharing Python `huggingface_hub`'s cache layout (`hub.New`), or read offline from local folders (`hub.NewLocal`) or `//go:embed` (`hub.NewEmbed`).
- **Tokenizers & bucketing (`tokenizers`, `tokenizers/bucket`)**: Pure-Go WordPiece/BPE/Unigram/SentencePiece/GGUF-embedded tokenizers plus streaming length bucketing (`ByTwoBitBucket`, `ByPower`, `Unpadded`).
- **On-the-fly model conversion (`models/*`)**: Converts SafeTensors (`models/safetensors`), quantized GGUF (`models/gguf`, with native GGML quantized execution and `Q4_K_M`-first selection), transformers & sentence embeddings (`models/transformer`, with self-contained checkpoint save/reload), SAM2 (`models/sam2`), and Timestral (`models/timestral`) — plus HuggingFace `.onnx` repos via `onnx-gomlx`.
- **Datasets (`datasets`)**: Inspect splits/features, generate Parquet record Go structs (`cmd/generate_dataset_structs`), and stream records on-demand (`datasets.IterParquetFromDataset[T]`).
- **See [`go-huggingface` reference](./references/go-huggingface.md)** for package layout, code examples, and CLI tools.
