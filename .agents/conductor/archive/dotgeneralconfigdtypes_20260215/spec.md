# Specification: DotGeneral Configured Accumulator and Output DTypes

## Overview

This track introduces `AccumulatorDType` and `OutputDType` parameters to the generic backend's `DotGeneral()` operation's `config`. This enhancement aims to provide greater control over numerical precision during computations, allowing for optimized performance and memory usage by specifying intermediate accumulation and final output data types.

## Functional Requirements

1.  **Signature Update:** The generic `DotGeneral()` operation (and its implementations in `backends/notimplemented`, `simplego`, and `xla`) must be updated to accept a `config` parameter that includes optional `AccumulatorDType` and `OutputDType` fields.
2.  **Data Type Support:** The `AccumulatorDType` and `OutputDType` parameters should support all numeric data types available in GoMLX.
3.  **`simplego` Backend Implementation:**
    *   If `AccumulatorDType` is defined in the `config`, the `simplego` backend must introduce graph nodes to convert the inputs of the `DotGeneral()` operation to the specified `AccumulatorDType` before execution.
    *   If `OutputDType` is defined in the `config` and is different from the effective `AccumulatorDType`, the `simplego` backend must add a graph node at the end of the operation to convert the result to the specified `OutputDType`.
4.  **`xla` Backend Implementation:** The `xla` backend must utilize the `stablehlo` API to appropriately set the precision (for accumulation) and output data types according to the `AccumulatorDType` and `OutputDType` parameters provided in the `config`.
5.  **Parameter Precedence and Invariant:** Backends have flexibility in how they implement the internal logic of `AccumulatorDType` and `OutputDType`. However, the critical invariant is that if `OutputDType` is set in the `config`, the final data type of the operation's output must match the specified `OutputDType`.

## Acceptance Criteria

*   The `DotGeneral()` operation in the generic backend interface and its concrete implementations (`notimplemented`, `simplego`, `xla`) successfully accept `AccumulatorDType` and `OutputDType` fields within its `config` parameter.
*   The `simplego` backend correctly performs necessary type conversions for inputs to `AccumulatorDType` and for the final output to `OutputDType` (if specified and different from accumulator).
*   The `xla` backend successfully configures `stablehlo` operations to respect the `AccumulatorDType` and `OutputDType` specified in the `config`.
*   The `DotGeneral()` function in the `notimplemented` backend is updated to match the new signature.
*   For any `DotGeneral()` execution where `OutputDType` is specified in the `config`, the resulting output tensor's DType precisely matches the configured `OutputDType`.
*   No new public API changes are introduced to expose `AccumulatorDType` and `OutputDType` to end-users in this iteration.

## Out of Scope

*   Exposing `AccumulatorDType` and `OutputDType` functionality directly to end-users through the GoMLX public API. This will be addressed in a subsequent track.
*   Detailed performance profiling or optimization of the `simplego` or `xla` backend implementations beyond ensuring correctness.
*   Changes to `DotGeneral()` operations in other backends (if any exist beyond `simplego`, `xla`, `notimplemented`).
