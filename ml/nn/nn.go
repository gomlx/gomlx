// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package nn provides low-level implementations of common neural network operations.
//
// It handles routing to backend-specific fused ops when available, and otherwise
// falls back to standard graph computations. It does not manage variables directly:
// functions take and return [*github.com/gomlx/gomlx/core/graph.Node] values instead of
// accepting a [github.com/gomlx/gomlx/ml/model.Scope] or [github.com/gomlx/gomlx/ml/model.Store].
//
// Typically, users building models will not call these functions directly. Instead,
// they should use higher-level layer packages (such as [github.com/gomlx/gomlx/ml/layers]
// or [github.com/gomlx/gomlx/ml/layers/fnn]), which automatically manage model variables
// and call the underlying implementations here.
package nn
