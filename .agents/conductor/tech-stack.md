# Tech Stack: GoMLX

## 1. Programming Languages

*   **Go:** The primary programming language used for GoMLX development.

## 2. Frameworks and Libraries

*   **GoMLX:** The core machine learning framework itself, designed for building, training, and fine-tuning ML models in Go. It supports multiple backends for flexible deployment and accelerated computation.
*   **StableHLO:** XLA's intermediate representation (IR) used by GoMLX for representing machine learning operations. It is a high-level, domain-specific language that is used to describe the operations in a machine learning model. The `stablehlo` API is part of the `github.com/gomlx/go-xla` project.
*   **PJRT:** The Parallel Just-In-Time Runtime (PJRT) API used by GoMLX to execute the compiled machine code on a specific hardware platform. It's the standard API for executing _StableHLO_ on a hardware platform. It's also part of the `github.com/gomlx/go-xla` project.
*   **go-highway**: Used for the in-development SIMD support for the pure-go backend (in package `backends/simplego`). It provides a generic API for SIMD operations that is then parsed and translated to various target architecture's assembly instructions. It also generates a dynamic dispatcher, so the same binary can support multiple different versions of SIMD instructions, within the same architecture.

