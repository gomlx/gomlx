# Implementation Plan: DotGeneral Configured Accumulator and Output DTypes

## Phase 1: Update Generic Backend and `notimplemented` Backend [checkpoint: 912551c]

- [x] **Task:** Update the `DotGeneral()` signature in the generic backend interface (`backends/backends.go`) to include the new `config` parameter with `AccumulatorDType` and `OutputDType` fields.
- [x] **Task:** Update the `DotGeneral()` signature in the `notimplemented` backend (`backends/notimplemented/notimplemented.go`) to match the new generic backend interface.
- [ ] **Task:** Conductor - User Manual Verification 'Phase 1: Update Generic Backend and notimplemented Backend' (Protocol in workflow.md)

## Phase 2: Implement `simplego` Backend

- [ ] **Task:** Write failing tests for the `simplego` backend's `DotGeneral()` implementation to verify the new functionality.
    - [ ] Sub-task: Create a test case where `AccumulatorDType` is specified and verify that input types are converted correctly.
    - [ ] Sub-task: Create a test case where `OutputDType` is specified and verify that the output type is converted correctly.
    - [ ] Sub-task: Create a test case where both `AccumulatorDType` and `OutputDType` are specified.
- [ ] **Task:** Implement the `DotGeneral()` functionality in the `simplego` backend (`backends/simplego/simplego.go`).
    - [ ] Sub-task: Add logic to introduce graph nodes for input type conversion to `AccumulatorDType`.
    - [ ] Sub-task: Add logic to introduce a graph node for output type conversion to `OutputDType`.
- [ ] **Task:** Conductor - User Manual Verification 'Phase 2: Implement simplego Backend' (Protocol in workflow.md)

## Phase 3: Implement `xla` Backend

- [ ] **Task:** Write failing tests for the `xla` backend's `DotGeneral()` implementation to verify the new functionality.
    - [ ] Sub-task: Create a test case that specifies `AccumulatorDType` and `OutputDType` and verifies that the `stablehlo` precision and output types are set correctly.
- [ ] **Task:** Implement the `DotGeneral()` functionality in the `xla` backend (`backends/xla/xla.go`) by configuring the `stablehlo` operation with the specified `AccumulatorDType` and `OutputDType`.
- [ ] **Task:** Conductor - User Manual Verification 'Phase 3: Implement xla Backend' (Protocol in workflow.md)

## Phase 4: Update GoMLX Graph's `DotGeneral`

- [ ] **Task:** Update the `graph.DotGeneral` function to accept the new configuration options, allowing them to be passed down to the backend. This change should be internal and not exposed to the end-user at this stage.
- [ ] **Task:** Conductor - User Manual Verification 'Phase 4: Update GoMLX Graph's DotGeneral' (Protocol in workflow.md)
