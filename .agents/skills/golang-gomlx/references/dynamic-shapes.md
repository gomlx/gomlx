# GoMLX Dynamic Shapes Reference

GoMLX supports **input-conditioned dynamic shapes** where tensor dimensions can vary at runtime without rebuilding or recompiling the computation graph. Dynamic dimensions are represented by `shapes.DynamicDim` (`-1`) and optionally given axis names via `shapes.MakeDynamic(...)`.

---

## 1. Backend Capabilities

Check backend support at runtime via `backend.Capabilities().DynamicShapes` (`compute.DynamicShapesSupport`) or the boolean helper `backend.Capabilities().HasDynamicShapes()`:

| `compute.DynamicShapesSupport` | Meaning | Examples |
|---|---|---|
| `compute.DynamicShapesNone` | Backend requires static shapes; GoMLX recompiles per unique concrete input shape. | `xla` |
| `compute.DynamicShapesNative` | Backend compiles dynamic graphs once; zero runtime recompilation overhead across variable dimensions. | `go`, `onnx`, `mlx:dynamic` |
| `compute.DynamicShapesRecompiling` | Backend accepts dynamic graphs and shares constants/weights, but manages JIT kernel specialization internally. | — |

---

## 2. Configuring `Exec` for Dynamic Shapes

Declare which input axes are dynamic using `WithDynamicAxes` (or `WithDynamicAxesSpecs`) on `graph.Exec` or `model.Exec`:

```go
// For an Exec taking (tokens, seqLen), where tokens is [batch, seq] and seqLen is [batch]:
exec.WithDynamicAxes(
	[]string{"batch", "seq"}, // tokens dynamic axes
	[]string{"batch"},        // seqLen dynamic axes
)
```

- Static axes use `""` (empty string).
- **User-named axes** (like `"batch"`) enforce matching runtime sizes across inputs at compile/execute time.
- For **non-user-named dynamic dimensions** that only need runtime compatibility checks, generate unique `#`-prefixed names via `shapes.UniqueAxisName(prefix)`.

---

## 3. Operations Supporting Dynamic Shapes

### Dimension Abstraction
- `DimensionSpecFor(x, axis)`: Returns a `DimensionSpec` representing the dimension (static or dynamic with name).
- `DimensionSpecsFor(x)`: Returns a slice of `DimensionSpec`s for all axes of `x`.
- `DimensionSize(x, axis)`: Returns a scalar `*Node` (typically `Int64`, or backend's `DynamicDimDType`) with the dimension size (constant scalar if static, dynamic extraction if dynamic).

### Reshaping
- `DynamicReshape(operand, specs...)`: Reshapes according to `DimensionSpec`s (`StaticDim`, `DynamicDim`, `NamedDynamicDim`, `InferredDim`, `NamedInferredDim`). Automatically falls back to static `Reshape` if operand and all specs are static.
- `DynamicReshapeLike(operand, refNode)` / `ReshapeLike(operand, refNode)`: Reshapes operand to match the shape of `refNode`.
- `Reshape(operand, dims...)` / `ReshapeWithShape(operand, shape)`: Automatically delegates to dynamic reshape if operand has dynamic dimensions.

### Broadcasting
- **Implicit broadcasting (`Add`, `Mul`, `Where`, etc.) is strictly compile-time**: Only scalar operands or axes that are **statically `1`** broadcast implicitly (including `1 -> 0` → `0`).
  - Dynamic axes **never** implicitly broadcast at runtime if their concrete size happens to be `1`.
  - Pairing a dynamic axis with a non-`1` static dimension (e.g., `["batch", "#feat"]` with `["batch", 12]`) fails at graph-build time — use `DynamicReshape` to assert the static dimension, or `DynamicBroadcastInDim` / `BroadcastToShape` for explicit runtime `1`-or-match broadcasting.
- `DynamicBroadcastInDim(operand, broadcastAxes, specs...)`: Low-level broadcast to target `DimensionSpec`s (supports runtime `1`-or-match from dynamic operand axes, and narrows non-`1` static operand axes mapped to dynamic specs).
- `DynamicBroadcastLike(operand, refNode)` / `BroadcastLike(operand, refNode)`: Broadcasts to match `refNode` (static or dynamic).
- `BroadcastToShape(operand, shape)` / `DynamicBroadcastToShape(operand, shape)`: Broadcasts to target shape (static or dynamic).
- `BroadcastPrefix(operand, targetRank)`: Adds leading singleton axes and broadcasts to target rank.

### Iota & Generation
- `DynamicIota(g, dtype, iotaAxis, specs...)`: Creates a sequence tensor with dynamic target dimensions.
- `IotaLike(refNode, iotaAxis)`: Creates an `iota` matching `refNode`'s shape.

### Padding & Data-Dependent Ops
- `DynamicPad(operand, fillVal, padSpecs...)`: Pads with dynamic or static padding amounts.
- `NonZero(operand, config...)`: Returns indices of non-zero elements with a data-dependent dynamic count dimension (available only on backends supporting dynamic shapes; prefer boolean masking with `NotEqual` / `Where` on static-shape backends).

### Polymorphic Structural Ops
`ExpandAxes`, `InsertAxes`, `ExpandLeftToRank`, `Squeeze`, `Slice`, `Gather`, `ScatterUpdate`, `Convolve`, `Concatenate`, `Dot`, `Where`, `TopK`, `BottomK`, and `TopKMask` transparently handle both static and dynamic shapes.

---

## 4. Writing Polymorphic Layers

Write layers using `DimensionSpecFor`, `DimensionSize`, `DynamicReshape`, `BroadcastLike`, and `IotaLike`. These run with **zero overhead** on static graphs (falling back directly to static operations) while transparently supporting dynamic shapes when enabled.

If specialized logic is required for dynamic vs. static tensors, inspect `x.Shape().IsDynamic()` or `x.Shape().Dimensions[axis] == shapes.DynamicDim`.

---

## 5. Strategy When Dynamic Shapes Are Not Supported

On static-shape backends (such as `xla`), fluctuating input shapes cause JIT recompilation explosions. Use **bucketing and padding** to round variable dimensions to a small discrete set of bucket sizes (e.g. powers of 2, multiples of 32/64, or `github.com/gomlx/compute/support.TwoBitBucketLen`).

For tokenized text sequences, use `github.com/gomlx/go-huggingface/tokenizers/bucket` (see [`go-huggingface` reference](./go-huggingface.md#streaming-bucketing-tokenizersbucket)).
