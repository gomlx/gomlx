# Effective Go Style Guide Summary

This document summarizes key rules and best practices from the official "Effective Go" guide for writing idiomatic Go code.

## 1. Formatting
- **`gofmt`:** All Go code **must** be formatted with `gofmt` (or `go fmt`). This is a non-negotiable, automated standard.
- **Indentation:** Use tabs for indentation (`gofmt` handles this).
- **Line Length:** Go has no strict line length limit. Let `gofmt` handle line wrapping.

## 2. Naming
- **`MixedCaps`:** Use `MixedCaps` or `mixedCaps` for multi-word names. Do not use underscores.
- **Exported vs. Unexported:** Names starting with an uppercase letter are exported (public). Names starting with a lowercase letter are not exported (private).
- **Package Names:** Short, concise, single-word, lowercase names.
- **Getters:** Do not name getters with a `Get` prefix. A getter for a field named `owner` should be named `Owner()`.
- **Interface Names:** One-method interfaces are named by the method name plus an `-er` suffix (e.g., `Reader`, `Writer`).

## 3. Control Structures
- **`if`:** No parentheses around the condition. Braces are mandatory. Can include an initialization statement (e.g., `if err := file.Chmod(0664); err != nil`).
- **`for`:** Go's only looping construct. Unifies `for` and `while`. Use `for...range` to iterate over slices, maps, strings, and channels.
- **`switch`:** More general than in C. Cases do not fall through by default (use `fallthrough` explicitly). Can be used without an expression to function as a cleaner `if-else-if` chain.

## 4. Functions
- **Multiple Returns:** Functions can return multiple values. This is the standard way to return a result and an error (e.g., `value, err`).
- **Named Result Parameters:** Return parameters can be named. This can make code clearer and more concise.
- **`defer`:** Schedules a function call to be run immediately before the function executing `defer` returns. Use it for cleanup tasks like closing files.

## 5. Data
- **`new` vs. `make`:**
  - `new(T)`: Allocates memory for a new item of type `T`, zeroes it, and returns a pointer (`*T`).
  - `make(T, ...)`: Creates and initializes slices, maps, and channels only. Returns an initialized value of type `T` (not a pointer).
- **Slices:** The preferred way to work with sequences. They are more flexible than arrays.
- **Maps:** Use the "comma ok" idiom to check for the existence of a key: `value, ok := myMap[key]`.

## 6. Interfaces
- **Implicit Implementation:** A type implements an interface by implementing its methods. No `implements` keyword is needed.
- **Small Interfaces:** Prefer many small interfaces over one large one. The standard library is full of single-method interfaces (e.g., `io.Reader`).

## 7. Concurrency
- **Share Memory By Communicating:** This is the core philosophy. Do not communicate by sharing memory; instead, share memory by communicating.
- **Goroutines:** Lightweight, concurrently executing functions. Start one with the `go` keyword.
- **Channels:** Typed conduits for communication between goroutines. Use `make` to create them.

## 8. Errors
- **`error` type:** The built-in `error` interface is the standard way to handle errors.
- **Explicit Error Handling:** Do not discard errors with the blank identifier (`_`). Check for errors explicitly.
- **`panic`:** Reserved for truly exceptional, unrecoverable situations. Generally, libraries should not panic.

*Source: [Effective Go](https://go.dev/doc/effective_go)*
---

## 9. GoMLX Specific Coding Style

### Auto-generated code

Files that start with `gen_` are auto-generated and don't include a copyright line
directly -- the copyright line is in their generators.
Many are created with generators included under `internal/cmd/...`, and the generated file 
includes a comment stating which tool was used to generate them.

### Graph Building Functions

GoMLX works by building computation graphs and then JIT-compiling and executing them in a backend.

The graph building functions (they take `*Node` and `*Context` as arguments, and return updated `*Node` of the graph)
are assumed to be executed sequentially (no concurrency between graph building functions), so no need for mutexes, etc.

Also, different than standard Go, graph building functions return errors by throwing (panicking) instead of returning 
an error, to simplify the code. But everywhere else, use standard Go error handling (by returning an error).

The compiled and execution of the graphs later is parallelized and can be executed in concurrently as one wishes.

Files that mostly define graph building functions, by convention, should dot-import the 
`github.com/gomlx/gomlx/pkg/core/graph` package: having `graph.` repeated everywhere makes the math harder to read.
This is commonly the case for libraries under `pkg/core/ml/layers`.

### Generators

Files that generate code (under `internal/cmd/...`) should assume they are running from the directory where they are
going to generate files -- this is what happens when one runs `go generate ./...`.

### Modern Go Style

- Use generics where possible.
- Use `slices` and `maps` package for slice operations.
- Look also into `pkg/support/xslices` package for more slice and map helper methods.
- Look into `pkg/support/xsync` package for more syncronization helpers.
- Look into `pkg/support/sets` package for a generic `Set[T]` structure.
- Use iterators (package `iter`) where it makes sense.
- Use the `for range` construct for loops over slices, maps, etc.
- Use `any` instead of `interface{}`.
- Organize tests in hierarchies using `t.Run()` to group related tests.

### Copyright Notes

Normal code files are prefixed with the following copyright line:

`// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0`

Auto-generated files don't need a copyright, but should include a comment with the tool use to generate them.
