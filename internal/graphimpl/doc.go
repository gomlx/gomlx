// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package graphimpl contains the internal computation-graph engine for GoMLX:
// NodeImpl, GraphImpl, FunctionImpl, ExecImpl, auto-generated backend ops,
// and fused/backend-level graph primitives.
//
// End-users should use package github.com/gomlx/gomlx/core/graph (and core/autodiff).
package graphimpl

//go:generate go run ../../internal/cmd/graph_generator
