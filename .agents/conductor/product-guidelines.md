# Product Guidelines: GoMLX

## 1. Tone and Voice

GoMLX documentation and external communications will adopt a balanced tone that is both **Technical and Authoritative** and **Approachable and Educational**. This means:

*   **Precision and Accuracy:** Emphasizing precise, accurate, and in-depth technical explanations suitable for experienced developers and researchers who require detailed information.
*   **Clarity and Support:** Focusing on clear, easy-to-understand language and a supportive tone, making it accessible for newcomers and those learning machine learning within the Go ecosystem.

## 2. Visual Identity

GoMLX's visual identity will be **Go-Branded**, strongly incorporating elements of the Go language's branding. The Gopher mascot, particularly the `gomlx_gopher2.png` image, will be a central and recognizable component of all visual materials, reinforcing GoMLX's identity as a native Go framework.

## 3. Documentation and Error Handling Principles

### Documentation

GoMLX documentation will adhere to the following core principles:

*   **Comprehensive and Up-to-date:** All public APIs and important concepts will be thoroughly documented, accompanied by relevant examples and clear explanations. Documentation will be actively maintained to remain current with the codebase, ensuring its reliability and usefulness.

### Error Handling

Error handling in GoMLX will be guided by these principles:

*   **Clear and Actionable Error Messages:** Error messages will be designed to be highly informative, guiding the user towards understanding the root cause of an issue and providing actionable advice for resolution. This includes providing full stack traces where appropriate to aid debugging.
*   **Context-Specific Error Management:**
    *   **Graph Building Functions:** For "graph building" functions (which are primarily mathematical operations and are expected to be executed sequentially without concurrency), errors will be handled by panicking with a detailed stack trace. This approach simplifies the code by avoiding explicit error returns in these math-intensive sections.
    *   **General GoMLX Code:** In all other parts of the GoMLX codebase, standard Go error handling mechanisms will be employed (returning errors). Occasionally, convenience functions like "Must<Func>" will be provided for cases where error checking is guaranteed or handled upstream.
