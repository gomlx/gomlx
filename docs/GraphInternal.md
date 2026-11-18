# Refactoring `core/graph` into Public API and Internal Package(s)

Design scratchpad for separating the public GoMLX graph-building API (`core/graph`) from internal engine plumbing, `Internal*` helpers, and raw `Backend*` wrappers.

---

## 1. Goals & Decisions Made So Far

1. **Remove all `Internal*`, `Backend*`, and low-level engine plumbing from `core/graph`**:
   - Move `Internal*`, `Backend*`, `NodeXlaHandle` (delete — unused), `VJPRegistration`, `SingleOutputVJP`, `NodeInputs`, `Node.CloneWithInputs`, `Node.NumOutputs`, `Graph.Executable`, `Graph.AttachState`/`State`, `Graph.Assert*`, `Graph.CurrentFunc`, etc. out of the public API first.
   - If any of those turn out to be legitimately needed publicly later, we can selectively re-expose them then (much easier than removing later).
2. **Keep high-level public op implementations readable in `core/graph`**:
   - External users should be able to browse `core/graph` to read how high-level ops (`Dot`, `Convolve`, `MaxPool`, `Softmax`, `TopK`, `Mean`, `Ragged2D`, etc.) are implemented, and see clean `go doc` / `pkg.go.dev` documentation.
3. **Keep `backend*` / `Backend*` / `nodeInputs*` out of `core/graph`**:
   - Ideally, `graph_generator` generates all `Backend<Op>` and `nodeInputs<Op>` implementations in an internal package, and **only** the fully public backend ops (`Add`, `Sub`, `Abs`, etc.) are generated into `core/graph`.
4. **Fix `onnx-gomlx` via `nn.FusedScaledDotProductAttention` (`FusedSDPAConfig`)**:
   - External modules like `onnx-gomlx` must not call `Internal*` or `Backend*` methods.
   - Add `nn.FusedScaledDotProductAttention` + `nn.FusedSDPAConfig` in `ml/nn/` (independent of `model.Store`, following `nn.Dense` and `nn.LayerNorm`), converting `*graph.Node` fields (`Mask`, `Bias`, `QuerySeqLen`, `KVSeqLen`) to backend values internally via `internal/graphbridge`.
   - Have both `ml/layers/attention.Core` (for the non-dropout/non-coefficients/non-softcap path) and [quantized_sdpa.go](file:///home/janpf/Projects/gomlx/onnx-gomlx/internal/onnxgomlx/fusion/quantized_sdpa.go) call `nn.FusedScaledDotProductAttention`.
5. **Remove `ParamsMap` / deprecated `RunWithMap` from public API**:
   - `ParamsMap` (`map[*Node]any`) is only used by the already-deprecated `Graph.RunWithMap` (and one test in `ops_test.go`). Make `ParamsMap` / `RunWithMap` internal (or remove) and keep/check parameter lookup (`Graph.GetParameterByName(name)` already exists; consider `GetParam(name string) *Node`).
6. **Back-links on `GraphImpl` / `NodeImpl` and `graphbridge` naming**:
   - Store a back-link (`public any`) on `NodeImpl`, `GraphImpl`, `FunctionImpl`, `ExecImpl` pointing to the canonical `*graph.Node`, `*graph.Graph`, etc.
   - In `internal/graphbridge`, convert cleanly between public and impl types (naming options: `graphbridge.Node(impl)` / `graphbridge.NodeImpl(n)` and `graphbridge.Graph(impl)` / `graphbridge.GraphImpl(g)` — or `PubNode` / `GetNodeImpl`).

---

## 2. Inventory of Non-Public / Low-Level Symbols in `core/graph`

### 2.1 `Internal*` Functions (9 symbols)

| Symbol | Defined in | External Callers (in `gomlx` `ml/*`) | Other Callers |
| --- | --- | --- | --- |
| `InternalFusedOpCaller` | `core/graph/fused_ops.go` | `ml/nn` (`dense.go`, `layernorm.go`, `quantized_dense.go`, `quantized_gather.go`), `ml/layers/activation`, `ml/layers/attention` | `core/graph/ops_util.go` (`Softmax`) |
| `InternalFusedOpCallerMulti` | `core/graph/fused_ops.go` | `ml/layers/attention/qkvprojection.go` | `core/graph/ops_order.go` (`topKImpl`) |
| `InternalBackendOutputs` | `core/graph/fused_ops.go` | `ml/model/variable.go`, `ml/layers/attention` (`attention.go`, `fusion_bias_test.go`) | `onnx-gomlx/internal/onnxgomlx/fusion/quantized_sdpa.go` *(currently disabled in `init()`)* |
| `InternalConstBuffer` | `core/graph/ops.go` | `ml/model/variable.go` | `core/graph/ops.go`, `core/graph/ops_test.go` |
| `InternalSharedConstant` | `core/graph/ops.go` | `ml/model/variable.go` | — |
| `InternalFinalizeConstantValue` | `core/graph/ops.go` | `ml/model/variable.go` | — |
| `InternalBatchNormForTraining` | `core/graph/ops.go` | `ml/layers/norm/batchnorm.go` | `core/graph/ops_test.go`, `core/graph/rev_autodiff_test.go` |
| `InternalBatchNormForInference` | `core/graph/ops.go` | `ml/layers/norm/batchnorm.go` | `core/graph/ops_test.go` |
| `InternalBatchNormGradient` | `core/graph/ops.go` | — *(none)* | `core/graph/rev_autodiff.go` only |

### 2.2 `Backend*` Functions (18 symbols)

- **11 used only inside `core/graph`**: `BackendFusedSoftmax`, `BackendFusedActivationVJP`, `BackendFusedLayerNormVJP`, `BackendFusedDenseVJP`, `BackendFusedQuantizedDenseVJP`, `BackendFusedTopK`, `BackendReduceWindow`, `BackendGather`, `BackendScatterMax`, `BackendScatterMin`, `BackendScatterSum`.
- **7 used by `ml/*`**: `BackendFusedActivation`, `BackendFusedLayerNorm`, `BackendFusedDense`, `BackendFusedQuantizedDense`, `BackendQuantizedGather`, `BackendFusedAttentionQKVProjection`, `BackendFusedScaledDotProductAttention` (plus `onnx-gomlx` disabled file).

### 2.3 Other Low-Level Exports Moving Internal (or Deleted)
- **Delete**: `NodeXlaHandle` (unused anywhere).
- **Move internal**: `VJPRegistration`, `SingleOutputVJP`, `NodeInputs`, `Node.CloneWithInputs`, `Node.NumOutputs`, `Graph.Executable` (used by `ml/model/onnx`), `Graph.AttachState`/`State` (used by `ml/model`), `Graph.Assert*`, `Graph.CurrentFunc`, `Graph.UniqueName`, `Graph.DynamicDimDType`, `InvalidNodeId`, `InvalidParameterHandle`, `MinConstValueSizeToKeep`.
- **Keep public in `core/graph`**: `Quantization`, `QuantizedDenseConfig`, `FusionEnv`, `MustAdjustAxis`, `NewClosure`, `NewFunction`, `Node.Type`, `Node.ConstantValue`, `Graph.Nodes`, `Graph.GraphId`, `Node.Id`, `Graph.Backend`, `Graph.LoggedNodes`, `Node.LogMessage`, `Exec.GetNodeLogger`, `Exec.SetNodeLogger`, `DonateTensorBuffer`, `CanonicalExecGraphFn`, `NewExecCanonical`.

---

## 3. Why the Cycle Exists & How the Two Clean Designs Resolve It

If `core/graph` owns `Node` and `Graph` directly, then `internal/graphimpl` cannot declare `BackendDotGeneral(lhs, rhs *graph.Node, ...)` because `core/graph` must import `internal/graphimpl` to call it (import cycle).

There are **two clean ways** in Go to resolve this without breaking `pkg.go.dev` / `go doc`:

### Candidate 1: Split Wrapper (`internal/graphimpl` owns `NodeImpl`/`GraphImpl` + `Backend<Op>`; `core/graph` wraps them)

`internal/graphimpl` defines **its own** `NodeImpl`, `GraphImpl`, `FunctionImpl`, and `ExecImpl` types (so there is never confusion between `graph.Node` and `graphimpl.NodeImpl`), and `Backend<Op>` takes/returns `*graphimpl.NodeImpl` without importing `core/graph`. Meanwhile, `core/graph` defines thin wrapper structs (`type Node struct { impl *graphimpl.NodeImpl }`), generates **only** wrappers for the ~59 fully public backend ops, and keeps all readable high-level `ops_*.go` files.

Concrete example of how the packages fit together:

```go
// internal/graphimpl (no import of core/graph)
type NodeImpl struct { /* outputOps, outputShapes, inputs, inputNodes, wrapper any, ... */ }
type GraphImpl struct { /* backend, builder, mainFunc, nodes, wrapper any, ... */ }

func Add(lhs, rhs *NodeImpl) *NodeImpl { ... }
func BackendDotGeneral(lhs, rhs *NodeImpl, ...) *NodeImpl { ... }

// core/graph (imports internal/graphimpl)
type Node struct { impl *graphimpl.NodeImpl }
type Graph struct { impl *graphimpl.GraphImpl }

// Generated in core/graph/gen_public_ops.go (only public backend ops):
func Add(lhs, rhs *Node) *Node {
    return wrapNode(graphimpl.Add(lhs.impl, rhs.impl))
}

// Hand-written in core/graph/ops_dotgeneral.go (stays readable in core/graph):
func (b *DotBuilder) General(...) *Node {
    return wrapNode(graphimpl.BackendDotGeneral(b.lhs.impl, b.rhs.impl, ...))
}
```

#### Acyclic Escape Hatch (`internal/graphbridge` / `internal/graphops`)

To let `ml/*` packages convert `*graph.Node` <-> `*graphimpl.NodeImpl` without exporting `.Impl()` on `graph.Node` and without an import cycle:

```
internal/graphimpl   <── core/graph   (core/graph.init() registers unwrap/wrap closures into graphimpl)
       ▲                     ▲
       │                     │
       └──────────┬──────────┘
         internal/graphbridge
                  ▲
                  │
       ml/nn, ml/model, ml/layers/*
```

1. **`internal/graphimpl`** holds unexported-to-external hook variables (`UnwrapNodeHook func(any) *NodeImpl`, `WrapNodeHook func(*NodeImpl) any`, and equivalents for `GraphImpl`, `FunctionImpl`, `ExecImpl`). It does **not** import `core/graph` or `internal/graphbridge`.
2. **`core/graph`** imports `internal/graphimpl`. In `core/graph`'s `init()`, it assigns closures that unwrap `n.(*Node).impl` and call `wrapNode(impl)`. Because `core/graph` does **not** import `internal/graphbridge`, there is no cycle.
3. **`internal/graphbridge`** imports **both** `core/graph` and `internal/graphimpl`, exposing strongly typed helpers:
   - `GetNodeImpl(n *graph.Node) *graphimpl.NodeImpl`, `WrapNode(impl *graphimpl.NodeImpl) *graph.Node` (and `GetGraphImpl`/`WrapGraph`, etc.).
   - Plus `*graph.Node`-typed convenience wrappers for `ml/*` (`FusedOpCaller`, `FusedDense`, `ConstBuffer`, `BackendOutputs`, `BatchNormForTraining`, `Executable`, `AttachState`, etc.).
   - Go guarantees `core/graph.init()` executes before `internal/graphbridge` or any of its callers can run, so the hooks are always initialized.

#### Package Structure

1. **`internal/graphimpl`** (does **NOT** import `core/graph`):
   - Core engine types: `NodeImpl`, `GraphImpl`, `FunctionImpl`, `ExecImpl` (each with an unexported `wrapper any` back-pointer so `core/graph` preserves 1-to-1 pointer identity for `ParamsMap` and `n.Graph() == g`).
   - `gen_compute_ops.go`: ALL `nodeInputs*` structs (with exported fields so high-level VJPs in `core/graph` can inspect them), ALL `Backend<Op>` functions (`BackendDotGeneral`, `BackendConvGeneral`, `BackendReduceWindow`, `BackendFusedDense`, etc.), and the ~59 raw public ops (`Add`, `Sub`, `Abs`, etc.).
   - `gen_nodetype_enumer.go`: `NodeType` enum.
   - `fused_ops.go`: `FusedOpCaller`, `FusedOpCallerMulti`, `BackendFused*`, `BackendQuantizedGather`, `ConstBuffer`, `SharedConstant`, `FinalizeConstantValue`, `BatchNormForTraining`, `BatchNormForInference`, `BatchNormGradient`.
   - `public.go`: manifest of `NodeImpl`/`GraphImpl`/`FunctionImpl`/`ExecImpl` methods and public ops to wrap into `core/graph`.
   - Hook variables (`UnwrapNodeHook`, `WrapNodeHook`, etc.) populated by `core/graph.init()`.

2. **`core/graph`** (imports `internal/graphimpl`):
   - Wrapper structs: `Node`, `Graph`, `Function`, `Exec` (no `Internal(...)` or `Impl()` method exported!).
   - `core/graph/gen_public_ops.go`: generated by `graph_generator` — **only** the ~59 fully public backend ops (`Add`, `Sub`, `Mul`, `Abs`, etc.) + wrappers for the public methods on `Node`/`Graph`/`Function`/`Exec` listed in `public.go`. Zero `backend*`, zero `Backend*`, zero `nodeInputs*`!
   - High-level readable ops remain directly in `core/graph`: `ops_dotgeneral.go`, `ops_convolve.go`, `ops_pool.go`, `ops_util.go`, `ops_order.go`, `ops_sparse.go`, `ops_random.go`, `ops_signal.go`, `ops_interpolate.go`, `ops_bitwise.go`, `ops_dynamic.go`, `ops_controlflow.go`, `distributed.go`, `asserts.go`, `introspection.go`, `node_aliases.go`, `ragged2d.go`, `execaliases.go`.
     - Inside `package graph`, these call `graphimpl.BackendDotGeneral(lhs.impl, rhs.impl, ...)` and `wrapNode(...)` directly!
     - `DotBuilder`, `ConvolutionBuilder`, `PoolBuilder`, `DistributedOps`, `Ragged2D`, `DimensionSpec`, `Quantization`, `QuantizedDenseConfig` stay natively in `core/graph` (no double definitions).
   - `core/graph.init()` registers:
     - `Unwrap*`/`Wrap*` hooks into `internal/graphimpl`, and
     - The 7 high-level VJPs defined in `ops_*.go` (`dotGeneralVJP`, `convGeneralVJP`, `reduceWindowVJP`, `fftVJP`, `scatterSumVJP`, `scatterMaxOrMinVJP`, `fusedTopKVJP`) into `graphimpl.VJPRegistration`.

3. **`internal/graphbridge`** (imports `core/graph` and `internal/graphimpl`):
   - Provides `Node` / `NodeImpl` (or `PubNode` / `GetNodeImpl`), `Graph` / `GraphImpl`, and typed `*graph.Node`/`*graph.Graph` helpers for `ml/*` (`FusedOpCaller`, `FusedOpCallerMulti`, `BackendOutputs`, `ConstBuffer`, `SharedConstant`, `FinalizeConstantValue`, `BatchNormForTraining`, `BatchNormForInference`, `FusedDense`, `FusedActivation`, `FusedLayerNorm`, `FusedQuantizedDense`, `QuantizedGather`, `FusedAttentionQKVProjection`, `FusedScaledDotProductAttention`, `Executable`, `AttachState`, `State`).

**Pros of Candidate 1 (`Option C2`):**
- Literally achieves *"only the fully public backend ops generated in `core/graph`, all `Backend<Op>` and `nodeInputs*` in `internal/graphimpl`"*.
- Zero `Internal*`, `Backend*`, `backend*`, or `nodeInputs*` symbols anywhere in `core/graph` (neither exported nor unexported).
- High-level op implementations (`Dot`, `Convolve`, `MaxPool`, `Softmax`, etc.) remain browsable in `core/graph`.
- `go doc` / `pkg.go.dev` is 100% pure and native.

**Trade-offs of Candidate 1:**
- Requires wrapper structs (`Node`, `Graph`, `Function`, `Exec`) + `public any` pointer-identity back-pointer + `gen_public_ops.go` generator updates.
- When paired with `core/autodiff` (Section 4), `rev_autodiff.go` and `rematerializationSweep` no longer need to live in `internal/graphimpl` at all — `internal/graphimpl` stays purely focused on `NodeImpl`/`GraphImpl`/`FunctionImpl`/`ExecImpl`, `gen_compute_ops.go`, and `fused_ops.go`!

---

### Candidate 2: Capability Key (`internal/graphimpl` KeyType + `internal/graphops`) — Lowest Churn

- `Node`, `Graph`, `Function`, `Exec`, `gen_compute_ops.go` (with **unexported** `backend*` and `nodeInputs*`), and `rev_autodiff.go` stay in `core/graph`.
- `Node.Internal(graphimpl.KeyType)` and `Graph.Internal(graphimpl.KeyType)` expose `NodeData`/`GraphData` only to internal packages; `internal/graphops` provides clean `*graph.Node` helpers to `ml/*`.
- **Pros**: Much smaller refactor, zero wrapper structs, zero generator rewrites.
- **Cons**: `gen_compute_ops.go` (unexported `backend*` + `nodeInputs*`) remains physically inside `./core/graph/`, and `Internal(graphimpl.KeyType)` appears as 1 method on `Node` and `Graph` in `go doc`.

---

---

## 4. Separating `core/autodiff` from `core/graph`

Moving `rev_autodiff.go` and the backward-pass `rematerializationSweep` (from `gradcheckpoint.go`) into a dedicated **`github.com/gomlx/gomlx/core/autodiff`** package is **not only viable — it actually simplifies `internal/graphimpl` significantly**!

### Dependency DAG with `core/autodiff`

```
internal/graphimpl  <── core/graph  <── internal/graphbridge
       ▲                     ▲                    ▲
       │                     │                    │
       └─────────────────────┴────────────────────┴── core/autodiff
```

Because `core/autodiff` sits *above* `core/graph`, `internal/graphimpl`, and `internal/graphbridge`, it gets **both views** natively with zero cycles:
- **Public `core/graph` view**: builds gradient expressions using normal public ops (`graph.Add`, `graph.Ones`, `graph.ZerosLike`, `graph.ReduceSum`, `graph.DotGeneral`, `graph.Slice`, `graph.Concatenate`, `graph.SchedulingBarrier`, etc.).
- **Internal `graphimpl` view** (via `graphbridge.NodeImpl(n)` / `graphbridge.GraphImpl(g)`):
  - Traverses graph DAG (`gImpl.Nodes()`, `nodeImpl.Id()`, `nodeImpl.Inputs()`, `nodeImpl.VJPAlternateOutputs()`).
  - Performs rematerialization (`nodeImpl.CloneWithInputs(...)`, `IsCheckpointed()`, `NeedsRematerialization()`).
  - Inspects `nodeImpl.Inputs().(*graphimpl.NodeInputs...)` for VJPs (`ReduceSum`, `Slice`, `Gather`, `Pad`, `ConvGeneral`, `DotGeneral`, `ReduceWindow`, `FFT`, `FusedTopK`, etc.).
  - Calls fused/batchnorm backend VJPs (`graphimpl.BackendBatchNormGradient`, `graphimpl.BackendFused*VJP`).

### What Moves to `core/autodiff` vs What Stays in `core/graph`

- **Moves to `core/autodiff`**:
  - `Gradient(output *graph.Node, gradientNodes ...*graph.Node) []*graph.Node`
  - `VJPRegistration`, `SingleOutputVJP` (and `VJP` if desired — see `IdentityWithCustomGradient` below)
  - All VJP implementations (both primitive VJPs from `rev_autodiff.go` and the op-specific VJPs currently in `ops_dotgeneral.go`, `ops_convolve.go`, `ops_pool.go`, `ops_signal.go`, `ops_sparse.go`, `ops_order.go`, `ops.go`, `fused_ops.go`)
  - `rematerializationSweep` (backward-pass rematerializer from `gradcheckpoint.go`)
- **Stays in `core/graph`**:
  - Graph-building annotations used during forward graph construction: `StopGradient(x)`, `Node.StopGradient()`, `Node.Checkpoint()`, `Node.StopCheckpoint()`, `IdentityWithCustomGradient(x, fn)`, `Node.CustomGradient()`, and `type VJP` (so `TopK`'s decomposed `SortFunc` fallback and `IdentityWithCustomGradient` can attach a `customVJP` on a `*graph.Node` without `core/graph` importing `core/autodiff`). `core/autodiff` simply checks `node.CustomGradient()` before `VJPRegistration`.

### API Breakage Note & Temporary Stop-Gap (`graph.Gradient` -> `autodiff.Gradient`)

- Because `core/autodiff` imports `core/graph`, `core/graph` **cannot** import `core/autodiff` directly.
- Moving `Gradient` to `autodiff.Gradient` has **0 callers** in `onnx-gomlx`, `go-huggingface`, `hugot`, or `benchmarks`. Inside `gomlx`, [ml/model/scope.go](file:///home/janpf/Projects/gomlx/gomlx/ml/model/scope.go#L475) (`Scope.Gradient`), [examples/gomlx.github.io/core-concepts/gradient/gradient.go](file:///home/janpf/Projects/gomlx/gomlx/examples/gomlx.github.io/core-concepts/gradient/gradient.go#L17), and `*_test.go` files will be updated to call `autodiff.Gradient` (note `package graph_test` tests in `core/graph/` *can* import `core/autodiff` without a cycle!).
- **Temporary `// Deprecated:` stop-gap in `core/graph`**:
  - `internal/graphimpl` defines `var GradientHook func(output *NodeImpl, gradientNodes ...*NodeImpl) []*NodeImpl`.
  - `core/autodiff.init()` registers `GradientHook` (converting via `graphbridge.Node` / `graphbridge.NodeImpl`).
  - Because `ml/model` (`scope.go`) imports `core/autodiff` and `ml/train` imports `ml/model`, virtually every user program initializes `GradientHook` automatically, so a `// Deprecated: use autodiff.Gradient instead.` wrapper `graph.Gradient(...)` continues to work seamlessly for existing downstream code.
  - Standalone `core/graph`-only callers that invoke `graph.Gradient` without importing `core/autodiff` get a clear panic directing them to `core/autodiff.Gradient`.

---

## 5. Open Questions / Next Steps

1. **Confirm the 4-package layout**: `internal/graphimpl` ← `core/graph` ← `internal/graphbridge` ← `core/autodiff` (with temporary `GradientHook` stop-gap for deprecated `graph.Gradient`).
2. **Conversion naming in `internal/graphbridge`**: `graphbridge.Node(impl)` / `graphbridge.NodeImpl(n)` (and `Graph` / `GraphImpl`) vs. `PubNode` / `GetNodeImpl`?
3. **`GetParam(name string) *Node`**: `Graph` already has `GetParameterByName(name string) *Node` — add `GetParam(name string) *Node` (or keep `GetParameterByName`) while making `ParamsMap` / `RunWithMap` internal or removed?
4. **Implementation order** (once approved):
   1. Add `nn.FusedScaledDotProductAttention` + `FusedSDPAConfig`, migrate `attention.Core` and `onnx-gomlx`'s `quantized_sdpa.go`.
   2. Create `internal/graphimpl` (`NodeImpl`/`GraphImpl`/`FunctionImpl`/`ExecImpl`, `fused_ops.go`), and update `graph_generator` to output `gen_compute_ops.go` into `internal/graphimpl` and `gen_public_ops.go` into `core/graph`.
   3. Wire `core/graph` wrappers + `internal/graphbridge`, migrate `ml/*`.
   4. Extract `core/autodiff` (`Gradient`, VJPs, `rematerializationSweep`) with `GradientHook` stop-gap, and update `ml/model/scope.go`, examples, and tests across `gomlx` and `onnx-gomlx`.

---

## 6. Documentation Update Reminder

Before finishing the refactor, update documentation where relevant:
- [x] `gomlx/.agents/AGENTS.md` (and `GEMINI.md` symlink) — document `internal/graphimpl`, `internal/graphbridge`, `core/autodiff`, and updated `core/graph` layout.
- [x] `gomlx/.agents/skills/...` (e.g. `golang-gomlx`) — update references to `autodiff.Gradient`, `nn.FusedScaledDotProductAttention`, and `core/graph` structure.
- [x] `gomlx/docs/CHANGELOG.md` — add a new entry for the refactor.
- [x] `../gomlx.github.io/content/docs/` — update user-facing docs mentioning `graph.Gradient`, `Internal*`/`Backend*` ops, `ParamsMap`, `nn.FusedScaledDotProductAttention`, or `core/autodiff`.
- [x] `../onnx-gomlx/.agents/AGENTS.md` (if needed) — verify no references to removed `graph.Internal*` / `graph.Backend*` APIs remain.
