# Auto-Fusion and Pipelined Execution in the Two-Level JIT

## Context

This document builds on [`two-level-jit.md`](two-level-jit.md) (the two-level JIT with named axes) and describes how auto-fusion and pipelined execution fit into that architecture. These are complementary optimizations: fusion eliminates boundaries within known patterns, pipelining overlaps execution across the remaining boundaries.

---

## Current State: Manual Fusion Only

Today, fusion is entirely explicit. `nn.Dense`, `Softmax`, and `LayerNorm` call `InternalFusedOpCaller` at graph construction time (`pkg/core/graph/fused_ops.go:43-65`), providing both a fused and decomposed path. There are hand-written fused op executors (`FusedSoftmax`, `FusedLayerNorm`, `FusedGelu`, `FusedDense`, `FusedMultiHeadSDPA`, `FusedQKVDense`), no graph rewriting passes, and no automatic pattern detection.

The existing `vjpAlternateOutput` mechanism handles gradients: the fused node stores the decomposed subgraph, and autodiff runs through the decomposed path. This means any automatically-detected fusions get gradients for free.

### What "Auto-Fusion" Means Here

To be clear about scope: auto-fusion in this document means two things:

1. **Auto-detection of existing fused patterns** (Phase 2): Pattern matching on the compiled graph to recognize op chains that correspond to existing hand-written fused executors (e.g., detecting `matmul→add→gelu` and rewriting to use the existing `execFusedDense`). No new executors are generated — this is just automatic recognition of patterns that already have implementations.

2. **Generic fused elementwise** (Phase 1): A new executor that chains arbitrary elementwise ops via an interpreter loop (see "Recommended First Step" below). This does NOT require code generation or a compiler — it uses a switch-based interpreter that processes all ops per element, keeping data in registers instead of writing intermediate buffers to memory.

---

## How XLA Handles This

XLA solves the same problem differently:

- **Op fusion** for local dependent chains: matmul+bias+activation get fused into a single GPU kernel, eliminating intermediate buffers. This is done via HLO fusion passes during compilation.
- **Async scheduling** for communication overlap: the `async-start`/`async-done` pattern (documented at https://openxla.org/xla/async_ops) wraps collective ops, allowing the Latency Hiding Scheduler to place computation between start and done. This is primarily for inter-device communication, not local compute chains.
- **For CPU backends** (closer to simplego's domain), XLA relies on threading within individual ops (Eigen thread pool for matmul) rather than cross-op pipelining.

The tiled pipeline approach described here does not have a direct XLA analog. XLA solves local chain overlap via fusion; it solves communication overlap via async scheduling. The tiled pipelining idea is closer to what systolic array architectures or dataflow engines do.

---

## How the Two-Level JIT Enables Both Optimizations

The two levels map directly to two distinct optimization opportunities.

### Level 1 (Graph Compilation) -- Pattern Matching and Rewriting

This is where a **fusion pass** scans the symbolic graph for fusible patterns. Since this runs once per symbolic shape pattern, the cost is amortized:

```go
func (fe *FunctionExecutable) applyFusionPass() {
    // Walk the DAG looking for fusible chains.
    // Patterns to match:
    //   matmul -> add (bias) -> activation  ->  FusedDense
    //   reduce_max -> sub -> exp -> reduce_sum -> div  ->  FusedSoftmax
    //   sub_mean -> variance -> scale -> shift  ->  FusedLayerNorm
    //   arbitrary elementwise chains  ->  FusedElementwise (new)

    for nodeIdx := range fe.numNodesToProcess {
        if chain := fe.matchFusionPattern(nodeIdx); chain != nil {
            fe.rewriteAsFused(chain)
        }
    }
}
```

This works with symbolic shapes because fusion eligibility is usually shape-independent -- a matmul->add->gelu chain is fusible regardless of whether the batch dimension is `"batch"` or `32`. The symbolic graph has enough information to decide what to fuse.

After the fusion pass, **pipeline chain detection** runs on the coarser post-fusion graph to identify tileable chains of fused ops.

### Level 2 (Shape Specialization) -- Fused Kernel and Pipeline Parameters

Once concrete shapes are known, compute execution parameters for fused ops and pipeline stages:

```go
func (spec *ShapeSpecialization) computeFusedOpParams(node *Node) {
    switch node.opType {
    case OpTypeFusedDense:
        // With concrete shapes, choose:
        // - matmul algorithm (tiled vs blocked vs naive)
        // - tile size for the fused matmul+bias+activation loop
        // - whether to parallelize across batch or output dim
        spec.opParams[node.idx] = computeFusedDenseParams(
            spec.nodeShapes[node.inputs[0].idx],
            spec.nodeShapes[node.inputs[1].idx],
        )
    case OpTypeFusedElementwise:
        // For a chain of elementwise ops, compute:
        // - chunk size for the fused loop
        // - whether inner ops can be done in-place
        spec.opParams[node.idx] = computeFusedElementwiseParams(
            spec.nodeShapes[node.idx],
        )
    }
}
```

Pipeline tile sizes are also computed here, since they depend on the concrete batch dimension size and cache characteristics.

### Where Each Optimization Lives

```
Level 1: Graph Compilation (once per symbolic pattern)
+-- Existing: CSE deduplication (function_dedup.go, unchanged)
+-- New: Fusion pass (pattern match -> rewrite)
|   +-- Produces fewer, coarser nodes
+-- New: Pipeline chain detection (on the post-fusion graph)
|   +-- Identifies tileable chains of fused ops
+-- Existing: Dependency counting (countNodeUsesAndDependents)

Level 2: Shape Specialization (once per axis binding)
+-- New: Resolve concrete shapes
+-- New: Fused op parameters (algorithm, tile size)
+-- New: Pipeline tile sizes (based on concrete batch dim)
```

---

## How Fusion and Pipelining Compose

After the fusion pass, the graph has fewer, coarser nodes. Pipelining then operates on the fused graph, not the original:

```
Original graph (7 nodes, all dependent):
  matmul1 -> add1 -> gelu1 -> matmul2 -> add2 -> gelu2 -> layernorm

After fusion (3 nodes):
  FusedDense1 -> FusedDense2 -> FusedLayerNorm

Pipeline the fused graph (tile along batch):
  FusedDense1:     [tile0][tile1][tile2][tile3]
  FusedDense2:           [tile0][tile1][tile2][tile3]
  FusedLayerNorm:              [tile0][tile1][tile2][tile3]
```

Fusion eliminates intra-pattern boundaries (matmul+add+gelu become one op). Pipelining eliminates inter-pattern boundaries (overlapping the fused ops via tiling). Both benefits stack.

---

## Pipelined Execution Design

### TiledBuffer -- Per-Tile Readiness Signaling

```go
type TiledBuffer struct {
    *Buffer
    tileAxis  int           // which axis is tiled
    numTiles  int
    tileSize  int
    ready     []atomic.Bool   // per-tile ready flag
    readyChan []chan struct{} // per-tile notification
}

func (tb *TiledBuffer) MarkTileReady(tileIdx int) {
    tb.ready[tileIdx].Store(true)
    close(tb.readyChan[tileIdx])
}

func (tb *TiledBuffer) WaitForTile(tileIdx int) {
    <-tb.readyChan[tileIdx]
}
```

### Tiled Executors

Expensive ops (matmul, fused dense) produce output tiles incrementally:

```go
func execDotGeneralTiled(backend *Backend, node *Node,
    inputs []*Buffer, inputsOwned []bool, output *TiledBuffer) error {
    for tileIdx := range output.numTiles {
        tileOffset := tileIdx * output.tileSize
        computeMatMulTile(inputs[0], inputs[1], output, tileOffset)
        output.MarkTileReady(tileIdx)
    }
    return nil
}
```

Downstream elementwise/fused ops wait per-tile:

```go
func execAddTiled(backend *Backend, node *Node,
    inputs []*TiledBuffer, output *TiledBuffer) error {
    for tileIdx := range inputs[0].numTiles {
        inputs[0].WaitForTile(tileIdx)
        inputs[1].WaitForTile(tileIdx)
        computeAddTile(inputs[0], inputs[1], output, tileIdx)
        output.MarkTileReady(tileIdx)
    }
    return nil
}
```

### Pipeline Chain Detection

During `newFunctionExecutable`, after fusion, identify linear chains where pipelining is profitable:

```go
type pipelineChain struct {
    nodes    []*Node // ordered producer -> consumer
    tileAxis int     // axis to tile along (typically batch)
    tileSize int     // computed during shape specialization
}

func (fe *FunctionExecutable) detectPipelineChains() []pipelineChain {
    // Criteria for a pipelineable chain:
    // 1. Linear dependency (each node has exactly one dependent in the chain)
    // 2. All ops support tiled execution
    // 3. Output shape along tileAxis is large enough to be worth tiling
    // 4. The leading op is expensive (matmul, conv) to amortize overhead
}
```

### Modified Execution Loop

In `executeParallel`, pipeline chains are dispatched as a unit:

```go
// When encountering a pipeline chain head:
if chain := fe.pipelineChains[nodeIdx]; chain != nil {
    // Launch all stages concurrently; they synchronize via TiledBuffer
    for _, stageNode := range chain.nodes {
        backend.workers.WaitToStart(func() {
            fe.executeTiledNode(backend, stageNode, execBuf)
        })
    }
    // Chain completion signals dependents of the last stage
}
```

---

## Integration Point: newFunctionExecutable

Both optimizations plug into the compilation phase:

```go
func newFunctionExecutable(f *Function) (*FunctionExecutable, error) {
    // ... existing setup ...

    // New: fusion pass (before dependency counting)
    if backend.autoFusion {
        fe.applyFusionPass()
    }

    // New: pipeline chain detection (after fusion, on coarser graph)
    if backend.pipelining {
        fe.detectPipelineChains()
    }

    // Existing: count uses and dependents (now on the fused/pipelined graph)
    for _, output := range f.outputs {
        fe.countNodeUsesAndDependents(output)
    }
    // ...
}
```

---

## Practical Considerations

### Tile Size Selection

Too small = overhead from synchronization; too large = no overlap. Needs tuning per op type and hardware cache sizes. Computed during Level 2 specialization since it depends on concrete shapes.

### Not All Ops Are Tileable

Reductions, transposes, and gathers may break the pipeline. Chain detection needs to handle these boundaries and only pipeline through ops that support tiling along the same axis.

### Buffer Ownership

The current ownership model (`inputsOwned`) assumes whole-buffer transfers. Tiled execution needs partial ownership or a different model where tile-level signaling replaces buffer-level ownership.

### Sequential Fallback Interaction

When `numLiveExecutions > 1`, the dynamic mode falls back to sequential (`function_exec.go:219-225`). Pipelining should still work in that case since it is intra-chain parallelism, not inter-graph.

### Graph Rewriting Complexity

The fusion pass must replace a subgraph of nodes with a single fused node while preserving DAG invariants (topological ordering, input references, `vjpAlternateOutput` linkage). The existing `InternalFusedOpCaller` pattern already solves the gradient problem, so the auto-fusion pass mimics what manual `InternalFusedOpCaller` does but at the compiled graph level instead of at construction time.

For Phase 2 (auto-detection of existing patterns), the rewriting maps to hand-written executors that already exist — no new executor code is generated. For Phase 1 (generic fused elementwise), the "executor" is the interpreter loop described above, which handles arbitrary elementwise chains without code generation.

---

## Recommended First Step: Generic Fused Elementwise

Before building a full pattern matcher for existing fused ops, add a **generic fused elementwise** op that chains arbitrary elementwise operations (add, mul, activation functions, etc.) into a single loop over the data. This is the highest-value fusion because:

1. **Memory-bandwidth-bound**: Elementwise ops are bandwidth-limited, so fusing N ops gives close to Nx speedup by reading/writing data once instead of N times.
2. **Trivial pattern matching**: Any chain of unary/binary elementwise ops with single-use intermediates is fusible.
3. **No code generation needed**: Uses an interpreter loop (see below).
4. **Composes with pipelining**: The fused elementwise chunk becomes a pipeline stage.

This is distinct from the existing hand-tuned fused ops. A generic fused elementwise op catches everything those miss.

### Execution Without Code Generation

The fused elementwise executor uses an **interpreter loop** — no bytecode, no compiler, no dynamic linking. The chain of operations is stored as data, and a switch statement dispatches per element:

```go
// Chain description, built during graph compilation
type elemOp struct {
    opType   backends.OpType  // Add, Mul, Neg, Gelu, Tanh, etc.
    inputIdx int              // for binary ops: which input buffer to use as second operand
}

type nodeFusedElementwise struct {
    ops    []elemOp    // ordered chain of operations
    inputs []int       // indices of input buffers (external inputs to the chain)
}

// Executor: single pass over the data, all ops applied per element
func execFusedElementwise[T float32 | float64](
    chain []elemOp, inputs [][]T, output []T,
) {
    sqrt2Inv := T(1.0 / math.Sqrt(2.0))
    for i := range output {
        val := inputs[0][i]
        for _, op := range chain {
            switch op.opType {
            case backends.OpTypeAdd:
                val = val + inputs[op.inputIdx][i]
            case backends.OpTypeMul:
                val = val * inputs[op.inputIdx][i]
            case backends.OpTypeNeg:
                val = -val
            case backends.OpTypeTanh:
                val = T(math.Tanh(float64(val)))
            case backends.OpTypeFusedGelu:
                val = val * 0.5 * (1.0 + T(math.Erf(float64(val*sqrt2Inv))))
            // ... other elementwise ops
            }
        }
        output[i] = val
    }
}
```

**Why this works without code generation**: The bottleneck for elementwise chains is memory bandwidth — each intermediate buffer is written to and read from main memory. A chain of 5 elementwise ops does 5 reads + 5 writes of the full tensor. The fused interpreter does 1 read + 1 write, with the intermediate values staying in CPU registers. The switch dispatch overhead (~1 branch mispredict per element per op) is negligible compared to the memory bandwidth savings.

For comparison, XLA generates LLVM IR for fused kernels, which eliminates the dispatch overhead but requires a full compiler pipeline. The interpreter approach is much simpler and captures the main benefit for a CPU backend where memory bandwidth is the bottleneck.

---

## Implementation Phases

### Phase 1: Generic Fused Elementwise
- Add `OpTypeFusedElementwise` with a list of sub-operations (`[]elemOp`)
- Pattern matcher: find chains of elementwise ops with single-use intermediates
- Executor: interpreter loop applying all ops per element (no code generation)
- Integrate into `newFunctionExecutable` as an optional pass
- This is self-contained and can be implemented independently of the two-level JIT

### Phase 2: Auto-Detection of Existing Fusion Patterns
- Pattern matchers for the existing hand-written fused executors: FusedDense, FusedSoftmax, FusedLayerNorm
- Detect op chains in the compiled graph and rewrite them to use existing executors
- Keep manual `InternalFusedOpCaller` calls as a fast path (skip pattern matching when already fused)
- No new executor code needed — this is pure graph rewriting to existing ops

### Phase 3: Pipelined Execution
- `TiledBuffer` type with per-tile readiness signaling
- Tiled variants for matmul and fused ops
- Pipeline chain detection in `newFunctionExecutable`
- Modified `executeParallel` to dispatch pipeline chains as units

### Phase 4: Two-Level JIT Integration
- Fusion pass runs at Level 1 (graph compilation, symbolic shapes)
- Tile sizes and fused kernel parameters computed at Level 2 (shape specialization)
- Pipeline chain tile sizes vary per axis binding

---

## Summary

| Optimization | What It Eliminates | Where in JIT | Speedup Source |
|-|-|-|-|
| Auto-fusion | Intermediate buffers between adjacent ops | Level 1 (pattern match) + Level 2 (kernel params) | Reduced memory bandwidth |
| Pipelining | Idle time between dependent fused ops | Level 1 (chain detect) + Level 2 (tile sizes) | Overlapped execution |
| Combined | Both | Both levels | Stacked benefits |

Fusion reduces the graph to fewer coarser nodes. Pipelining overlaps those coarser nodes. The two-level JIT provides the right architecture for both: Level 1 for shape-independent decisions (what to fuse, what to pipeline) and Level 2 for shape-dependent decisions (how to execute the fused ops, what tile sizes to use).
