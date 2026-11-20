# ONNX-GoMLX Reference (`github.com/gomlx/onnx-gomlx`)

`onnx-gomlx` parses ONNX model files (`.onnx` protobufs) and converts them on-the-fly into GoMLX computation graphs. This lets you run inference on any GoMLX backend (`xla`, `go`, `onnx`, `mps`, `mlx`), compose ONNX models with custom GoMLX pre-/post-processing or other models, and fine-tune ONNX models with GoMLX automatic differentiation (`core/autodiff`) and training loops (`ml/train`) — optionally exporting updated weights back to `.onnx`.

> **Note**: Do not confuse `onnx-gomlx` with `compute-onnx`:
> - **`github.com/gomlx/onnx-gomlx`**: converts `.onnx` files *into* GoMLX computation graphs (runs on any GoMLX backend, supports fine-tuning & graph composition).
> - **`github.com/gomlx/compute-onnx`**: a `compute.Backend` implementation (`GOMLX_BACKEND=onnx` / `onnx:cuda`) that executes GoMLX graphs *using* ONNX Runtime.

```bash
go get -u github.com/gomlx/onnx-gomlx
```

## Public Packages

| Package | Role | Key Exports |
|---|---|---|
| `github.com/gomlx/onnx-gomlx/onnx` | Public `Model` interface and external data types | `Model`, `ExternalDataReader`, `ExternalDataInfo`, `ModelScope` (`"ONNX"`) |
| `github.com/gomlx/onnx-gomlx/onnx/parser` | Parses `.onnx` protobufs into `onnx.Model` (auto-registers fusion matchers and graph passes) | `Parse([]byte)`, `ParseFile(filePath)`, `ParseReader(io.Reader)` |

---

## Standard Inference Workflow

1. Parse the `.onnx` file via `parser.ParseFile` (or `parser.Parse` / `parser.ParseReader`).
2. Copy the ONNX initializers (weights) into a GoMLX `model.Store` with `onnxModel.VariablesToScope(store.RootScope())`.
3. Build a GoMLX executor (`model.NewExec` / `model.MustNewExec`) calling `onnxModel.CallGraph(scope, g, inputsMap, outputNames...)`.
4. Execute with tensors or Go slices and finalize resources when done.

```go
package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Registers default backends (xla, go, etc.)
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

func main() {
	// 1. Parse the ONNX model file.
	onnxModel, err := parser.ParseFile("path/to/model.onnx")
	if err != nil {
		panic(err)
	}
	defer onnxModel.Close()

	// Optional: inspect input/output names and shapes.
	inNames, inShapes := onnxModel.Inputs()
	outNames, outShapes := onnxModel.Outputs()
	fmt.Printf("Inputs: %v %v\nOutputs: %v %v\n", inNames, inShapes, outNames, outShapes)

	// 2. Load ONNX weights/initializers into a GoMLX model.Store.
	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		panic(err)
	}

	// 3. Build a GoMLX executor calling onnxModel.CallGraph.
	backend := compute.MustNew()
	defer backend.Finalize()

	exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
		g := inputIDs.Graph()
		outputs := onnxModel.CallGraph(scope, g, map[string]*graph.Node{
			"input_ids":      inputIDs,
			"attention_mask": attentionMask,
			"token_type_ids": tokenTypeIDs,
		})
		return outputs[0]
	})
	if err != nil {
		panic(err)
	}
	defer exec.Finalize()

	// 4. Execute with Go slices or *tensors.Tensor.
	inputIDs := [][]int64{{101, 2023, 2003, 2019, 2742, 6251, 102}}
	attentionMask := [][]int64{{1, 1, 1, 1, 1, 1, 1}}
	tokenTypeIDs := [][]int64{{0, 0, 0, 0, 0, 0, 0}}

	out := exec.MustCall1(inputIDs, attentionMask, tokenTypeIDs)
	defer out.FinalizeAll()
	fmt.Printf("Output shape: %s\n", out.Shape())
}
```

### Selecting Specific Outputs & Composing Graphs

- Passing `outputNames...` to `CallGraph(scope, g, inputs, "logits", "hidden_state")` returns only those requested ONNX node outputs in order; omitting `outputNames` returns the model's declared default outputs.
- Because `CallGraph` emits ordinary GoMLX `*graph.Node` values inside `g`, you can chain custom pre-processing before `CallGraph` (e.g., building masks or image normalization) and post-processing after it (e.g., pooling, L2 normalization, classification heads, or combining multiple models) into a single fused JIT-compiled graph.

---

## Dynamic Shapes Support

On backends supporting dynamic shapes (`GOMLX_BACKEND=go`, `GOMLX_BACKEND=onnx:cpu`), configure `exec.WithDynamicAxes(...)` so varying batch sizes or sequence lengths execute without graph recompilation:

```go
exec.WithDynamicAxes(
	[]string{"batch", "seq"}, // input_ids
	[]string{"batch", "seq"}, // attention_mask
	[]string{"batch", "seq"}, // token_type_ids
)

// Both calls reuse the same compiled graph:
out1 := exec.MustCall1(idsBatch1, maskBatch1, typesBatch1)
out2 := exec.MustCall1(idsBatch2, maskBatch2, typesBatch2)
```

On static-shape backends (`xla`), pad/bucket inputs to fixed shapes (e.g., using `github.com/gomlx/go-huggingface/tokenizers/bucket`) to avoid repeated JIT compilation.

---

## Fine-Tuning & Exporting Weights Back to ONNX

Because `onnxModel.CallGraph` builds a standard GoMLX computation graph over variables in `model.Store`, GoMLX reverse-mode autodiff (`core/autodiff`) and `ml/train` work out of the box:

1. Extract weights into a `model.Store`: `onnxModel.VariablesToScope(store.RootScope())`.
2. Call `onnxModel.CallGraph(scope, g, inputs)` inside your `train.Trainer` model function (optionally attaching new layers or PEFT/LoRA adapters).
3. Train using `train.Trainer` and `train.Loop` as usual.
4. Persist the fine-tuned model in either format:
   - **GoMLX checkpoint**: `checkpoint.Build(store).Dir(checkpointDir).Done()`.
   - **Updated `.onnx` file**: copy updated variables from `store` back into the in-memory ONNX proto and serialize:
     ```go
     if err := onnxModel.ScopeToONNX(store.RootScope()); err != nil {
         panic(err)
     }
     if err := onnxModel.SaveToFile("path/to/finetuned.onnx"); err != nil {
         panic(err)
     }
     // Or write to any io.Writer:
     // err := onnxModel.Write(writer)
     ```

---

## `onnx.Model` API Quick Reference

| Category | Method | Purpose |
|---|---|---|
| **Inspection** | `Name() string` | Returns the ONNX graph name. |
| | `Inputs() ([]string, []shapes.Shape)` | Input names and shapes (`shapes.DynamicDim` == `-1` for dynamic axes). |
| | `Outputs() ([]string, []shapes.Shape)` | Output names and shapes. |
| | `NumInputs() int` | Number of graph inputs. |
| | `ShapeForName(name string) shapes.Shape` | Inferred shape of any intermediate/output tensor by ONNX node output name. |
| | `ValidateInputs(inputShapes ...shapes.Shape) error` | Validates concrete shapes against the model's declared dynamic/static input shapes. |
| | `String() string`, `PrintGraph(w)`, `PrintVariables(w)`, `PrintGraphviz(w, targets...)` | Debugging and Graphviz export of the ONNX graph and initializers. |
| **Weights & Persistence** | `VariablesToScope(scope *model.Scope) error` | Uploads ONNX initializers as variables under `scope` (default sub-scope `onnx.ModelScope` = `"ONNX"`). |
| | `FreeUnusedVariables()` | Releases initializers not referenced by the graph. |
| | `ScopeToONNX(scope *model.Scope) error` | Copies updated GoMLX variable values back into the ONNX proto. |
| | `SaveToFile(path string) error`, `Write(w io.Writer) error` | Serializes the ONNX model to disk or a writer. |
| | `Close() error` | Releases resources (e.g., external data file handles). |
| **External Data (> 2 GiB models)** | `WithBaseDir(baseDir string) Model` | Sets directory used to resolve external tensor data files (defaults to `.onnx` file's directory or `"."`). Call before `VariablesToScope` / `CallGraph`. |
| | `WithExternalDataReader(reader ExternalDataReader) Model` | Supplies a custom `ExternalDataReader` (`ReadInto(info, buf)` / `Close()`) for remote, cloud, or embedded external weight files. |
| **Conversion Options** | `WithInputsAsConstants(map[string]any) Model` | Marks selected inputs as compile-time constants (enables constant folding of shape/metadata subexpressions). |
| | `AllowDTypePromotion() Model` | Enables automatic dtype promotion when an ONNX op has mismatched input dtypes. |
| | `PrioritizeFloat16() Model` | Prefers `Float16` over `Float32` during dtype promotion. |
| | `ForceApproximateGelu(enabled bool) Model` | Forces `GeluApproximate` (`tanh`) regardless of the ONNX node's `approximate` attribute. |
| | `DisableFusion()` | Disables automatic fused-op pattern matching (useful for debugging). |

---

## Automatic Fused-Op Matching & Debugging

Before node-by-node conversion, `onnx-gomlx` scans the ONNX graph for multi-op subgraphs and maps them to hardware-accelerated GoMLX `ml/nn` / `ml/layers` fused ops:
- **Attention**: `FusedScaledDotProductAttention` (`sdpa`), `FusedAttentionQKVProjection` (`qkvdense`), and quantized SDPA/QKV variants.
- **Linear & Quantized Linear**: `FusedDense` (including fused GELU/activations) and `FusedQuantizedDense` (e.g. `MatMulNBits` / `MatMulInteger` / block-quantized weights).
- **Normalization**: `FusedLayerNorm`.

If the active backend does not support a fused op configuration, GoMLX's internal `FusedOpCaller` catches `compute.ErrNotImplemented` and falls back transparently to the decomposed graph.

### Debugging Tips
- **Isolate fused vs. decomposed behavior**: run with `GOMLX_FUSION=false` in the environment, or call `onnxModel.DisableFusion()`.
- **Inspect `.onnx` files from the CLI**:
  ```bash
  go run github.com/gomlx/compute-onnx/cmd/onnx_printer@latest path/to/model.onnx
  ```
- **Downloading ONNX models from HuggingFace Hub**: combine `github.com/gomlx/go-huggingface/hub` with `onnx/parser` (see [`go-huggingface` reference](./go-huggingface.md#onnx-models-on-huggingface)).
