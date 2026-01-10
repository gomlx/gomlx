//go:build !(darwin && cgo)

// Package coreml provides a CoreML backend for GoMLX on macOS.
// This file is a no-op stub for non-darwin platforms and when CGO is disabled.
package coreml

// On non-darwin platforms or when CGO is disabled, the backend is not registered.
// The package can still be imported but the backend cannot be used.
