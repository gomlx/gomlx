# Parameter-Efficient Fine-Tuning (PEFT)

Package [`ml/layers/peft`](../ml/layers/peft) provides parameter-efficient fine-tuning (LoRA) for GoMLX.

PEFT in GoMLX works as a meta-layer via post-linear lifecycle hooks. You do **not** need to rewrite your model architecture or manually swap out dense layers. Simply import `peft` and set the desired hyperparameters on your `model.Store` or through command-line settings (`-set`).

## Quickstart: Automatic LoRA

Import the `peft` package:

```go
import _ "github.com/gomlx/gomlx/ml/layers/peft"
```

Then configure the adapter hyperparameters on your store before building or running the model:

```go
store.SetParams(map[string]any{
	peft.ParamAdapter: "task_a",         // Enables LoRA with adapter name "task_a"
	peft.ParamRank:    16,               // LoRA rank (r)
	peft.ParamAlpha:   float32(32),      // LoRA scaling numerator (alpha)
	peft.ParamDropout: float32(0.05),    // Dropout rate on intermediate projection (x @ A)
})
```

Or configure via the `-set` command-line flag:

```bash
go run ./my_model -set="peft_adapter=task_a;peft_rank=16;peft_alpha=32;peft_dropout=0.05"
```

When building layers with [`ml/layers/fnn`](../ml/layers/fnn), [`layers.Dense`](../ml/layers), or [`ml/layers/attention`](../ml/layers/attention), GoMLX automatically:
1. Freezes the base linear weights (`weights.SetTrainable(false)`).
2. Creates trainable low-rank matrices `A` and `B` under `<layer_scope>/<adapter_name>/`.
3. Adds the low-rank delta $\Delta y = \frac{\alpha}{r} \text{Dropout}(x A) B$ to the base projection before non-linearities (activations, layer norm, residuals).
4. Restricts gradient descent to only train the adapter variables.

If `peft_adapter` is empty (`""`), PEFT is completely inactive and models execute with standard trainable base weights and zero overhead.

## Hyperparameter Reference

| Hyperparameter | Scope Key (`peft.*`) | CLI / `-set` Name | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ParamAdapter` | `"peft_adapter"` | `peft_adapter` | `""` | Adapter sub-scope name (e.g. `"task_a"`, `"lora"`). If empty, PEFT is inactive. If `"off"`, base weights are frozen with no adapter. |
| `ParamRank` | `"peft_rank"` | `peft_rank` | `0` | Low-rank dimension ($r$). |
| `ParamAlpha` | `"peft_alpha"` | `peft_alpha` | equals `rank` | Scaling factor ($\alpha$). Scaling multiplier is $\frac{\alpha}{r}$. |
| `ParamDropout` | `"peft_dropout"` | `peft_dropout` | `0.0` | Dropout probability applied to $(x A)$. |
| `ParamBias` | `"peft_bias"` | `peft_bias` | `BiasNone` (`"none"`) | Controls base bias trainability (`"none"` or `"all"`). |
| `ParamTargetModules` | `"peft_target_modules"` | `peft_target_modules` | `""` | Comma/space-separated list or `[]string` of target module scope substrings. |

## Selective Module Targeting (`peft_target_modules`)

In large models such as Transformers, it is common to fine-tune only specific projections (e.g., query and value attention projections, but not key or output projections).

You can target specific modules by setting `peft.ParamTargetModules`:

```go
store.SetParams(map[string]any{
	peft.ParamAdapter:       "attn_lora",
	peft.ParamRank:          16,
	peft.ParamTargetModules: "query,value", // or []string{"query", "value"}
})
```

CLI flag:
```bash
-set="peft_adapter=attn_lora;peft_rank=16;peft_target_modules=query,value"
```

**Targeting Behavior:**
- Any linear projection whose scope matches one of the target substrings receives a LoRA adapter, and its base weights are frozen.
- Any linear projection whose scope does **not** match has its base weights **frozen** as well, but **no** LoRA adapter is created.

## Disabling Adapters ("Freeze-Only" Mode)

To freeze base weights without attaching any LoRA adapter, set `peft_adapter` to `"off"`, `"disabled"`, `"false"`, or `"none"`:

```go
// Freeze base weights everywhere without creating adapters:
store.SetParam(peft.ParamAdapter, "off")

// Or freeze a specific layer via sub-scope:
store.RootScope().In("fnn_output_layer").SetParam(peft.ParamAdapter, "off")
```

## Multi-Adapter Support & Coexistence

Different LoRA adapters can coexist within the same `model.Store` and checkpoint. Adapter weights are isolated inside sub-scopes named after the adapter:

```text
/model/fnn_hidden_layer_0/weights          (frozen base)
/model/fnn_hidden_layer_0/biases           (frozen base)
/model/fnn_hidden_layer_0/task_a/A         (trainable when peft_adapter="task_a")
/model/fnn_hidden_layer_0/task_a/B
/model/fnn_hidden_layer_0/task_b/A         (trainable when peft_adapter="task_b")
/model/fnn_hidden_layer_0/task_b/B
```

Switching between adapters during training or inference simply requires updating the `peft_adapter` hyperparameter:

```go
// Switch active adapter to task_b:
store.SetParam(peft.ParamAdapter, "task_b")
```

## Direct / Manual API

If you are building custom layers that do not use `fnn`, `layers.Dense`, or `attention`, you can still construct a LoRA layer manually:

```go
projectionScope := scope.In("model").In("custom_proj")
weight := projectionScope.VariableWithShape("weights", shapes.Make(dtypes.Float32, inputDim, outputDim))
bias := projectionScope.VariableWithShape("biases", shapes.Make(dtypes.Float32, outputDim))

output := peft.New(projectionScope, input, weight, bias).Done()
// Or shorthand:
output := peft.Apply(projectionScope, input, weight, bias)
```

---

## Developer Guide: Implementation Details

### The Post-Linear Hook Pattern

GoMLX implements PEFT by inverting dependencies: layer packages (`fnn`, `layers`, `attention`) do **not** import `peft`. Instead, `ml/layers` provides an extensible hook registry:

```go
type PostLinearHook func(scope *model.Scope, input, output *Node) *Node
func RegisterPostLinearHook(priority int, hook PostLinearHook)
func ApplyPostLinearHooks(scope *model.Scope, input, output *Node) *Node
```

- **Registration**: When `ml/layers/peft` is imported, its `init()` automatically registers `postLinearHook` with `HookPriorityDefault`.
- **Execution**:
  - `layers.DenseWithLayout`: Calls `ApplyPostLinearHooks` immediately after `nn.Dense`.
  - `fnn.Done()`: Calls `ApplyPostLinearHooks` after each linear transformation, before activation/dropout/normalization.
  - `attention.MultiHeadAttention`: Delegates its query, key, value, and output projections through `layers.DenseWithLayout`, automatically invoking the hooks.

### Lifecycle & Variable Creation

When `postLinearHook(scope, input, output)` executes:
1. It queries `scope` for `peft_adapter`. If empty, it immediately returns `output` (no-op).
2. It fetches the layer's base `"weights"` (and `"biases"`) from `scope`.
3. If `peft_adapter` is `"off"` or the scope fails `peft_target_modules` filtering, it sets `weights.SetTrainable(false)` and returns `output`.
4. If active:
   - Sets `weights.SetTrainable(false)` and `biases.SetTrainable(ParamBias != BiasNone)`.
   - Creates adapter variables under `scope.In(adapterName)`:
     - `A`: Shape `[inFeatures, rank]`, initialized with uniform random $\pm \frac{1}{\sqrt{\text{inFeatures}}}$.
     - `B`: Shape `[rank, outFeatures]`, initialized with zeros.
   - Computes:
     $$\Delta y = \frac{\alpha}{r} \text{Dropout}(x A) B$$
   - Reshapes $\Delta y$ (using `DynamicReshapeLike`) if necessary to match multi-dimensional outputs, and returns `Add(output, delta)`.
