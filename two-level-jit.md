# Two-Level JIT: Unified Dynamic Shapes Architecture for SimpleGo

## Executive Summary

This document proposes a **two-level JIT** architecture that solves multiple dynamic shapes challenges with a single unified approach: **lazy shape specialization with named axes**.

**Key Insights**:
1. Separate graph structure (compiled once with dynamic shapes) from shape-specific execution context (specialized cheaply per concrete shape)
2. Use **named axes** from the start to enable compile-time validation, better error messages, and efficient specialization keying

---

## Scope and Terminology

### What We Support

| Type | Description | Example | Supported |
|------|-------------|---------|-----------|
| **Dynamic Dimensions** | Axis length unknown at compile time | `[batch:-, 512]` | Yes |
| **Named Axes** | Axes have symbolic names | `[batch, seq_len, hidden]` | Yes |
| **Rank Dynamism** | Number of dimensions unknown | `[?, ?, ...]` | No |
| **Data-Dependent** | Shape depends on tensor values | `Where(x>0)` output | No |

### Terminology

```go
// Shape with named dynamic axis
shape := shapes.Make(F32, "batch", 512)  // [batch:-, 512]

// IsDynamic: any axis has unknown length
shape.IsDynamic()  // true - "batch" is dynamic

// HasNamedAxes: some axes have names
shape.HasNamedAxes()  // true - "batch" is named

// IsFullyConcrete: all dimensions are known integers
shape.IsFullyConcrete()  // false
```

---

## Why Named Axes from the Start?

Named axes add modest complexity but provide significant benefits:

### Without Named Axes

```go
paramA := builder.Parameter("a", shapes.Make(F32, -1, 512))  // [-1, 512]
paramB := builder.Parameter("b", shapes.Make(F32, -1, 512))  // [-1, 512]
// Problem: No way to express "these -1s must be equal"
// Problem: Error message: "dimension 0 mismatch: 32 vs 64"
// Problem: Specialization key: "[32, 512, 64, 512]" (verbose)
```

### With Named Axes

```go
paramA := builder.Parameter("a", shapes.Make(F32, "batch", 512))  // [batch, 512]
paramB := builder.Parameter("b", shapes.Make(F32, "batch", 512))  // [batch, 512]
// Benefit: Compiler knows both "batch" dimensions must match
// Benefit: Error message: "batch dimension mismatch: a=32, b=64"
// Benefit: Specialization key: "{batch: 32}" (compact)
```

### Benefits Summary

| Benefit | Impact |
|---------|--------|
| **Compile-time validation** | Catch shape mismatches before execution |
| **Better error messages** | `"batch mismatch: 32 vs 64"` vs `"dim 0 mismatch"` |
| **Compact specialization keys** | Key by axis bindings (`{batch: 32}`) instead of full shape tuples (`[32, 128, 512, 32, 128, 768, ...]`) — fewer values to hash, smaller key space |
| **Self-documenting shapes** | `[batch, seq_len, hidden]` is readable |

Note: Existing node deduplication already works on input identity (pointer equality) and is independent of shape representation. Named axes don't change dedup behavior — the win is in validation and error messages.

---

## Problems We Need to Solve

When moving from static to dynamic shapes, three interconnected problems emerge:

### Problem 1: Buffer Pool Inefficiency

**Static shapes**: Buffer pools keyed by `(dtype, exactSize)` work perfectly because the same sizes are reused.

**Dynamic shapes**: Every new concrete shape needs different buffer sizes. Options considered:
- **Power-of-2 bucketing**: Wastes memory (up to 2x), fragments pools
- **No pooling**: Allocation overhead on every execution
- **Per-shape pools**: But how to manage lifecycle?

### Problem 2: DotGeneral Algorithm Selection

DotGeneral (matrix multiplication) has multiple execution paths (tiled, blocked, naive) selected based on matrix dimensions. Currently this is decided at compile time.

**Dynamic shapes**: Dimensions unknown at compile time. Options considered:
- **Runtime decision every execution**: Adds overhead to hot path
- **Conservative algorithm**: Suboptimal for many shapes
- **Dynamic dispatch node**: Complexity in graph structure

### Problem 3: Node Deduplication Under Dynamic Shapes

SimpleGo deduplicates nodes with identical `(opType, inputs, data, shape)` to avoid redundant computation. The primary dedup filter is input identity (pointer equality), which is shape-independent.

**Dynamic shapes**: Deduplication continues to work because input identity is the main criterion. Shapes with `-1` or named axes like `[batch, 512]` are compared symbolically, and two nodes with the same symbolic shape and same inputs will still dedupe. Named axes don't change dedup behavior, but they do prevent potential confusion where two `-1` dimensions represent different things.

---

## Why a Unified Solution?

All three problems share a common structure:

| Problem | Compile Time | Execution Time |
|---------|--------------|----------------|
| Buffers | Don't know sizes | Know exact sizes |
| Algorithms | Can't select optimal | Can select optimal |

**The pattern**: Information unavailable at compile time becomes available at execution time, but we don't want to pay the cost of computing it on every execution.

**The solution**: Compute shape-dependent information **once per axis binding**, cache it, reuse it.

(Node deduplication continues to work at compile time as before, since it's based on input identity rather than concrete shapes.)

---

## Two-Level JIT Architecture

### Level 1: Graph Compilation (Once per symbolic shape pattern)

```
Input: Graph function with named dynamic shapes [batch, 512]
Output: Executable with node structure, op types, symbolic shapes, compile-time dedup
Cost: Heavy (milliseconds) - full graph traversal, validation, optimization
```

### Level 2: Shape Specialization (Once per axis binding)

```
Input: Axis bindings {batch: 32}
Output: ShapeSpecialization with resolved shapes, buffer pools, algorithm choices
Cost: Light (microseconds) - just arithmetic, no graph building
```

### Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Executable (compiled once)                    │
│              Symbolic shape: [batch, 512]                        │
│                                                                  │
│   Graph structure: nodes[], edges, opTypes, symbolic shapes      │
│   Compile-time dedup: nodes with same symbolic shape merged      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Specializations: map[AxisBindings]*Specialization              │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│   │ {batch: 8}   │  │ {batch: 32}  │  │ {batch: 128} │          │
│   ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│   │ nodeShapes[] │  │ nodeShapes[] │  │ nodeShapes[] │          │
│   │ bufferPool   │  │ bufferPool   │  │ bufferPool   │          │
│   │ opParams[]   │  │ opParams[]   │  │ opParams[]   │          │
│   │ dedupMap     │  │ dedupMap     │  │ dedupMap     │          │
│   └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: Axis Bindings as Specialization Keys

With named axes, the specialization key is just the axis bindings, not the full shapes:

```go
// Graph with parameters:
//   a: [batch, seq_len, 512]
//   b: [batch, seq_len, 768]
//   c: [batch, 1024]

// Without named axes, key would be: "[32, 128, 512, 32, 128, 768, 32, 1024]"
// With named axes, key is just:     "{batch: 32, seq_len: 128}"

// This means fewer specializations when shapes share axis names
```

---

## Shape System Design

### Shape Struct

```go
type Shape struct {
    DType      dtypes.DType
    Dimensions []int      // concrete values, or -1 for dynamic
    AxisNames  []string   // axis names, empty string "" for unnamed axes
}

// Constructor helpers
func Make(dtype dtypes.DType, dims ...any) Shape {
    // dims can be int (concrete) or string (named dynamic)
    // shapes.Make(F32, "batch", 512) → [batch:-, 512]
    // shapes.Make(F32, 32, 512)      → [32, 512]
    // shapes.Make(F32, -1, 512)      → [-1, 512] (unnamed dynamic)
}
```

### Axis Bindings

```go
// AxisBindings maps axis names to concrete values
type AxisBindings map[string]int

// Resolve replaces named axes with concrete values
func (s Shape) Resolve(bindings AxisBindings) Shape {
    result := s.Clone()
    for i, name := range s.AxisNames {
        if name != "" {
            if val, ok := bindings[name]; ok {
                result.Dimensions[i] = val
            }
        }
    }
    return result
}

// ExtractBindings gets axis bindings from a concrete shape matching a pattern
func ExtractBindings(pattern, concrete Shape) (AxisBindings, error) {
    if pattern.Rank() != concrete.Rank() {
        return nil, errors.New("rank mismatch")
    }
    bindings := make(AxisBindings)
    for i, name := range pattern.AxisNames {
        if name != "" {
            if existing, ok := bindings[name]; ok && existing != concrete.Dimensions[i] {
                return nil, errors.Errorf("axis %q has conflicting values: %d vs %d",
                    name, existing, concrete.Dimensions[i])
            }
            bindings[name] = concrete.Dimensions[i]
        } else if pattern.Dimensions[i] != -1 && pattern.Dimensions[i] != concrete.Dimensions[i] {
            return nil, errors.Errorf("dimension %d mismatch: pattern has %d, got %d",
                i, pattern.Dimensions[i], concrete.Dimensions[i])
        }
    }
    return bindings, nil
}
```

### Shape Unification (for shape inference)

```go
// UnifyAxisName combines two axis names during shape inference
func UnifyAxisName(a, b string) (string, error) {
    if a == "" { return b, nil }  // unnamed adopts name
    if b == "" { return a, nil }  // unnamed adopts name
    if a == b  { return a, nil }  // same name OK
    return "", errors.Errorf("incompatible axis names: %q vs %q", a, b)
}

// Used in binary ops:
// [batch, 512] + [batch, 512] → [batch, 512]  ✓
// [batch, 512] + [-1, 512]    → [batch, 512]  ✓ (unnamed adopts)
// [batch, 512] + [time, 512]  → ERROR          ✗ (name conflict)
```

---

## Implementation Design

### ShapeSpecialization Struct

```go
// ShapeSpecialization holds all shape-dependent execution context.
// Created lazily on first execution with a given axis binding.
type ShapeSpecialization struct {
    // bindings are the axis values this specialization was created for.
    bindings AxisBindings

    // nodeShapes holds the resolved concrete shape for each node.
    // Computed by resolving symbolic shapes with bindings.
    nodeShapes []shapes.Shape

    // opParams holds pre-computed operation parameters.
    // For DotGeneral: batchSize, contractingSize, algorithm choice, block shapes.
    // For Conv: strides, dilated dims, etc.
    opParams []any  // indexed by node.builderIdx
}
```

Note: Buffer pooling is handled separately via power-of-2 bucketing on the backend's existing pool (see Solution 1). Per-specialization pools and runtime deduplication are potential later optimizations but not needed for v1.

### Executable Changes

```go
type Executable struct {
    // ... existing fields ...

    // hasDynamicAxes indicates this executable has named dynamic axes.
    hasDynamicAxes bool

    // axisNames is the set of all dynamic axis names in the graph.
    axisNames []string

    // specializations caches shape-specific execution contexts.
    // Keyed by axis bindings, not full shapes.
    specializations sync.Map  // bindingsKey -> *ShapeSpecialization
}
```

### Execution Flow

```go
func (e *Executable) Execute(inputs []*Buffer) ([]*Buffer, error) {
    if !e.hasDynamicAxes {
        // Static path: unchanged, no overhead
        return e.executeStatic(inputs)
    }

    // Extract axis bindings from concrete inputs
    bindings, err := e.extractBindings(inputs)
    if err != nil {
        return nil, err  // e.g., "batch axis mismatch: input 0 has 32, input 1 has 64"
    }

    // Get or create specialization for these bindings
    key := bindings.Key()
    spec, ok := e.specializations.Load(key)
    if !ok {
        spec, err = e.createSpecialization(bindings)
        if err != nil {
            return nil, err
        }
        e.specializations.Store(key, spec)
    }

    return e.executeWithSpecialization(spec.(*ShapeSpecialization), inputs)
}

func (e *Executable) extractBindings(inputs []*Buffer) (AxisBindings, error) {
    bindings := make(AxisBindings)

    for i, input := range inputs {
        paramShape := e.builder.inputs[i].shape  // symbolic shape
        concreteShape := input.shape

        paramBindings, err := ExtractBindings(paramShape, concreteShape)
        if err != nil {
            return nil, errors.Wrapf(err, "parameter %d", i)
        }

        // Merge bindings, checking for conflicts
        for name, val := range paramBindings {
            if existing, ok := bindings[name]; ok && existing != val {
                return nil, errors.Errorf("axis %q has conflicting values: parameter %d has %d, but earlier parameter had %d",
                    name, i, val, existing)
            }
            bindings[name] = val
        }
    }

    return bindings, nil
}
```

### Specialization Creation

```go
func (e *Executable) createSpecialization(bindings AxisBindings) (*ShapeSpecialization, error) {
    spec := &ShapeSpecialization{
        bindings:   bindings,
        nodeShapes: make([]shapes.Shape, len(e.builder.nodes)),
        opParams:   make([]any, len(e.builder.nodes)),
    }

    // 1. Resolve all node shapes by applying bindings to symbolic shapes
    for i, node := range e.builder.nodes {
        spec.nodeShapes[i] = node.shape.Resolve(bindings)
    }

    // 2. Pre-compute operation parameters (DotGeneral algorithms, etc.)
    for i, node := range e.builder.nodes {
        spec.opParams[i] = computeOpParams(node, spec.nodeShapes)
    }

    return spec, nil
}
```

---

## Node Deduplication

Existing compile-time deduplication (`function_dedup.go`) continues to work unchanged with dynamic shapes. It deduplicates nodes with identical `(opType, inputs, data, shape)`, where inputs are compared by pointer identity. Since input identity is the primary filter, dedup is largely shape-independent — two nodes with the same symbolic shape (e.g., `[batch, 512]`) and same inputs will dedupe just as they do today with static shapes.

Named axes don't change dedup behavior. The benefit of named axes is elsewhere: validation, error messages, and specialization keys.

---

## How Each Problem Is Solved

### Solution 1: Buffer Pooling

**V1 approach: Power-of-2 bucketing.** The existing buffer pool (keyed by exact `(dtype, length)`) works perfectly for static shapes because the same sizes recur. For dynamic shapes, many distinct sizes may appear. The simplest fix is power-of-2 bucketed pools:

```go
func bucketSize(size int) int {
    if size <= 0 { return 0 }
    // Round up to next power of 2
    return 1 << bits.Len(uint(size-1))
}

// Pool key uses bucketed size instead of exact size
func (b *Backend) getBufferForShape(shape shapes.Shape) *Buffer {
    bucketed := bucketSize(shape.Size())
    buf := b.getBuffer(shape.DType, bucketed)
    buf.shape = shape  // Track actual shape, not bucketed
    return buf
}
```

**Why this works for v1**:
- At most 2x memory overhead per buffer (acceptable)
- Reuses the existing pool infrastructure
- No per-specialization pools needed — the bucketing handles size variation
- Simple to implement and reason about

**Later optimization**: Per-specialization exact-sized pools (since specializations cache exact shapes, pools keyed by specialization would see the same sizes repeatedly). But this is unnecessary for v1.

### Solution 2: DotGeneral Algorithm Selection

```go
func computeDotGeneralParams(node *Node, nodeShapes []shapes.Shape) *dotGeneralParams {
    data := node.data.(*dotGeneralNodeData)
    lhsShape := nodeShapes[node.inputs[0].builderIdx]
    rhsShape := nodeShapes[node.inputs[1].builderIdx]

    return &dotGeneralParams{
        batchSize:       computeBatchSize(lhsShape, data.lhsBatchAxes),
        contractingSize: computeContractingSize(lhsShape, data.lhsContractingAxes),
        lhsCrossSize:    computeCrossSize(lhsShape, data),
        rhsCrossSize:    computeCrossSize(rhsShape, data),
        algorithm:       selectAlgorithm(lhsShape, rhsShape),
        blockShape:      computeBlockShape(lhsShape, rhsShape),
    }
}
```

**Why this works**:
- Algorithm selected once per axis binding
- No runtime overhead on hot path
- Optimal algorithm for each concrete shape

### Solution 3: Deduplication

Existing compile-time deduplication works unchanged with dynamic shapes. It's driven by input identity (pointer equality), which is shape-independent. Nodes with the same inputs, opType, data, and symbolic shape will dedupe at compile time just as they do today.

No runtime deduplication layer is needed for v1.

---

## Cost Analysis

| Operation | Frequency | Cost | Notes |
|-----------|-----------|------|-------|
| Graph compilation | Once per symbolic pattern | Heavy (ms) | Includes compile-time dedup |
| Specialization creation | Once per axis binding | Light (μs) | Just arithmetic |
| Specialization lookup | Every execution | O(1) | Hash on axis bindings |
| Execution | Every call | Same as static | No additional overhead |

### Specialization Key Efficiency

The specialization key is what we hash to look up a cached specialization. With named axes, this key is compact:

```go
// Without named axes - key must encode all parameter shapes:
// Parameters: a:[32, 128, 512], b:[32, 128, 768], c:[32, 1024]
// Key: "[32, 128, 512, 32, 128, 768, 32, 1024]"  (8 values)

// With named axes - key is just the axis bindings:
// Parameters: a:[batch, seq_len, 512], b:[batch, seq_len, 768], c:[batch, 1024]
// Key: "{batch:32, seq_len:128}"  (2 values)

// Fewer values to hash, smaller key space, faster lookup
```

Without named axes, every concrete dimension appears in the key — including static dimensions that never change. With named axes, the key contains only the dynamic axis values, which is typically much smaller.

### Memory Overhead

```
Per Specialization:
  - bindings:      O(num_axis_names) ~100 bytes
  - nodeShapes[]:  N * sizeof(Shape) ~N * 40 bytes
  - opParams[]:    N * sizeof(pointer) ~N * 8 bytes

For a graph with 1000 nodes: ~48KB per specialization
```

### Bounding Memory

```go
// Optional: LRU eviction of specializations
type Executable struct {
    maxSpecializations int
    specLRU            *lru.Cache
}

func (e *Executable) getSpecialization(bindings AxisBindings) *ShapeSpecialization {
    key := bindings.Key()
    if spec, ok := e.specLRU.Get(key); ok {
        return spec.(*ShapeSpecialization)
    }

    spec := e.createSpecialization(bindings)
    e.specLRU.Add(key, spec)
    return spec
}
```

---

## Comparison with Alternatives

### Alternative 1: Dynamic Shapes without Named Axes

```
Compile: Once per pattern (e.g., [-1, 512])
Specialize: Once per full concrete shape tuple
```

**Rejected**:
- No compile-time validation of axis consistency
- Poor error messages
- Verbose specialization keys
- No compile-time deduplication

### Alternative 2: Full Recompilation per Shape

```
Compile: Once per concrete shape (current static approach)
Execute: Fast
```

**Rejected**: Compilation is expensive. Many shapes = many compilations.

### Alternative 3: Bucketed Shapes

```
Compile: Once per bucket
Execute: Use bucket's values
```

**Rejected**: Complex bucketing, suboptimal edge cases, memory waste.

### Chosen: Two-Level JIT with Named Axes

```
Compile: Once per symbolic pattern, with compile-time dedup
Specialize: Once per axis binding (cheap)
Execute: Fast
```

**Benefits**:
- O(1) compilations
- O(unique axis bindings) specializations
- Compile-time validation and deduplication
- Clear error messages
- Compact specialization keys

---

## Implementation Phases

### Phase 0: Named Axes Foundation
- Add `AxisNames []string` to `shapes.Shape`
- Add `AxisBindings` type and methods
- Update `shapes.Make()` to accept string axis names
- Implement `Shape.Resolve(bindings)`
- Implement `ExtractBindings(pattern, concrete)`
- Implement `UnifyAxisName` for shape inference
- Update shape inference to propagate axis names

### Phase 1: Core Two-Level Infrastructure
- Add `ShapeSpecialization` struct
- Add `specializations` map to `Executable`
- Implement axis binding extraction from inputs
- Implement basic specialization creation (shape resolution)

### Phase 2: Buffer Pool with Bucketing
- Add power-of-2 bucketing to the existing buffer pool
- No per-specialization pools needed — bucketing handles size variation
- Verify performance is acceptable with up to 2x memory overhead

### Phase 3: Operation Parameters
- Add `opParams` to specialization
- Implement `computeDotGeneralParams`
- Implement `computeConvParams`
- Update executors to use pre-computed params

### Phase 4: Polish
- Add optional LRU eviction of specializations
- Add metrics/monitoring
- Performance testing
- Documentation

---

## Ahead-of-Time (AOT) Specialization

For production deployments with known shapes, pre-create specializations:

```go
// Create specializations before any execution
exec.PreSpecialize(
    AxisBindings{"batch": 1},    // inference
    AxisBindings{"batch": 32},   // training
    AxisBindings{"batch": 64},   // large batch
)

// First execution with batch=32 is fast - specialization already exists
result := exec.MustExec(batchOf32)
```

This eliminates the latency spike on first execution with a new shape.

---

## Future Extensions

### Axis Constraints (Optional)

More advanced symbolic relationships:

```go
// Express: output_seq = input_seq - kernel + 1
shapes.Make(F32, "batch", "output_seq").
    Where("output_seq", "input_seq - kernel + 1")
```

This is not needed for Phase 3 but could be added later for more precise validation.

### Cross-Execution Caching

Cache results across executions when inputs are identical:

```go
type ResultCache struct {
    // (inputHashes, axisBindings) -> outputs
}
```

### Backend Capability Flag

```go
type Capabilities struct {
    // ...
    DynamicAxes bool  // SimpleGo: true, XLA: false
    NamedAxes   bool  // SimpleGo: true, XLA: false
}
```

XLA requires static shapes at compile time, so it continues using per-shape compilation.

---

## Summary

The two-level JIT architecture with named axes provides a unified solution:

1. **Named axes** enable compile-time validation and clear error messages
2. **One compiled graph** handles infinite shape variations
3. **Cheap specializations** keyed by compact axis bindings (`{batch: 32}`)
4. **Power-of-2 bucketed buffers** for v1 (simple, at most 2x overhead)
5. **Pre-computed algorithms** without runtime overhead
6. **Existing deduplication** works unchanged — it's input-identity-based

This approach combines the clarity of symbolic shapes with the efficiency of lazy specialization, providing the best of both worlds.
