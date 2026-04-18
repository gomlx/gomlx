// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package notimplemented is an alias to compute/notimplemented package. It's here for historical reasons.
//
// Deprecated: use the [github.com/gomlx/compute/notimplemented] package instead.
package notimplemented

import "github.com/gomlx/compute/notimplemented"

// Backend is a dummy backend that can be imported to create mock compute.
//
// Deprecated: it's just an alias to [notimplemented.Backend], use that instead.
type Backend = notimplemented.Backend

// Builder implements compute.Builder and returns the NotImplementedError wrapped with the stack-trace.
//
// Deprecated: it's just an alias to [notimplemented.Builder], use that instead.
type Builder = notimplemented.Builder

// Function implements compute.Function and returns NotImplementedError for every operation.
//
// Deprecated: it's just an alias to [notimplemented.Function], use that instead.
type Function = notimplemented.Function

// NotImplementedError is returned by every method.
//
// Deprecated: it's just an alias to [notimplemented.NotImplementedError], use that instead.
var NotImplementedError = notimplemented.NotImplementedError
