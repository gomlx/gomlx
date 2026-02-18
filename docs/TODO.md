  # GoMLX Roadmap & Ownership

  This document tracks feature development, ownership, and priorities.

  ---

  ## Status Legend

  - `idea` – not scoped yet
  - `ready` – clearly defined, free to claim
  - `in_progress` – actively worked on
  - `blocked` – waiting on dependency
  - `done` – completed

  ## Priority Legend

  - `P0` – critical / ecosystem enabling
  - `P1` – important feature expansion
  - `P2` – nice to have / exploratory

  ---

  # Modeling

  ---

  ## [MODEL-1] HuggingFace Transformer Import (safetensors direct)

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P0
  - **Area:** `pkg/ml/model/transformer`
  - **Depends on:** GRAPH-1
  - **Description:** Import generic transformer architectures directly from safetensors without ONNX.
  - **Definition of Done:**
    - Load at least 2 transformer families (e.g., GPT-style, BERT-style)
    - Weights load successfully from safetensors
    - Minimal documentation of supported architecture subset

  ---

  ## [MODEL-2] Text Generation Example (HF Transformer + Prompt)

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P0
  - **Depends on:** MODEL-1
  - **Description:** Provide example generator using imported transformer model.
  - **Definition of Done:**
    - `examples/transformer_generate`
    - Prompt → generated output
    - Reproducible with seed

  ---

  ## [MODEL-3] ONNX Import Coverage Expansion

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Repo:** ONNX-GoMLX
  - **Description:** Improve op coverage for ONNX graph import.
  - **Definition of Done:**
    - Missing-op report tool
    - Implement top 5 most common missing ops
    - Validate against 3 popular models

  ---

  ## [MODEL-4] New Layers / Optimizers / Losses

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P1
  - **Description:** Add recent ML layers, optimizers, regularizers.
  - **Definition of Done:**
    - At least 2 new optimizers
    - At least 2 new layers
    - Benchmark example

  ---

  ## [MODEL-5] GNN Layer Support

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P2
  - **Description:** Add Graph Neural Network layer abstraction.
  - **Definition of Done:**
    - Basic message-passing GNN
    - Example on sample dataset

  ---

  ## [MODEL-6] Gradient Checkpointing

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Description:** Reduce memory usage during training.
  - **Definition of Done:**
    - Memory usage reduced in benchmark
    - Same convergence behavior

  ---

  ## [MODEL-7] Jacobian Support (non-scalar gradients)

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Description:** Support gradients wrt vector/tensor outputs.
  - **Definition of Done:**
    - API to compute Jacobian
    - Unit tests with analytic comparison

  ---

  # Graph

  ---

  ## [GRAPH-1] Dynamic Shapes (Input-Dependent, Fixed Rank)

  - **DRI:** _unassigned_
  - **Status:** in_progress
  - **Priority:** P0
  - **Links:** PR #306
  - **Definition of Done:**
    - Input-dependent shapes supported
    - Full shape inference test coverage

  ---

  ## [GRAPH-2] Data-Dependent Shapes (Fixed Rank)

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P1
  - **Depends on:** GRAPH-1
  - **Definition of Done:**
    - Runtime shape propagation works
    - Fallback for unsupported cases

  ---

  ## [GRAPH-3] Symbolic Shapes (Named Axes)

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P0
  - **Depends on:** GRAPH-1
  - **Definition of Done:**
    - Named axis abstraction
    - Symbolic inference working in transformer example

  ---

  # Backends

  ---

  ## [BE-1] SimpleGo Backend – Fused Ops Expansion

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Definition of Done:**
    - 3 fused patterns implemented
    - Benchmark shows measurable improvement

  ---

  ## [BE-2] SimpleGo Backend – SIMD Support (Go 1.26+)

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P1
  - **Depends on:** Go 1.26, go-highway
  - **Definition of Done:**
    - SIMD path for matmul or conv
    - Benchmark comparison

  ---

  ## [BE-3] ONNX Export API

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Description:** Save GoMLX graph as ONNX model.
  - **Definition of Done:**
    - `ExportONNX()` API
    - Model validated with ONNX runtime

  ---

  ## [BE-4] WebGL/WebNN WASM Backend

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P2
  - **Definition of Done:**
    - Proof-of-concept inference in browser

  ---

  ## [BE-5] llama.cpp Backend (purego / yzma)

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Definition of Done:**
    - Load llama.cpp model
    - Inference parity with reference

  ---

  ## [BE-6] Additional Backend Exploration

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P2

  ---

  # Infrastructure

  ---

  ## [INFRA-1] Inference-Only Distribution Mode

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P1
  - **Definition of Done:**
    - Backend-only load mode
    - Load pre-exported model
    - Minimal example

  ---

  ## [INFRA-2] Distributed Training Validation

  - **DRI:** _unassigned_
  - **Status:** blocked
  - **Priority:** P0
  - **Definition of Done:**
    - Multi-node test example
    - Stability verification
    - Documentation

  ---

  # API Improvements

  ---

  ## [API-1] Replace Context with Plain Go Struct + Annotations

  - **DRI:** _unassigned_
  - **Status:** idea
  - **Priority:** P1
  - **Description:** Replace Context object with struct-based annotated config.
  - **Definition of Done:**
    - RFC document
    - Reflection-based prototype
    - Migration example

  ---

  # Maintenance

  ---

  ## [MAINT-1] Code Cleanup & Refactoring Pass

  - **DRI:** _unassigned_
  - **Status:** ready
  - **Priority:** P2
  - **Definition of Done:**
    - Remove deprecated APIs
    - Improve test coverage

  ---

  # Priority Overview

  ## P0 (Critical)
  - MODEL-1
  - MODEL-2
  - GRAPH-1
  - GRAPH-3
  - INFRA-2

  ## P1 (Important)
  - MODEL-3
  - MODEL-4
  - MODEL-6
  - MODEL-7
  - BE-1
  - BE-2
  - BE-3
  - BE-5
  - INFRA-1
  - API-1

  ## P2 (Exploratory)
  - MODEL-5
  - BE-4
  - BE-6
  - MAINT-1
