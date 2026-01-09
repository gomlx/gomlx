// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package context

// Aliases to GoMLX basic types.
//
// Usually one would simply import the full `graph` with a period (".") so
// the ops don't need a package qualifier. But we can't do this in `context`
// package because we have conflicting symbols (`Exec`, `MustNewExec`).

import (
	"github.com/gomlx/gomlx/backends"
	graph "github.com/gomlx/gomlx/pkg/core/graph"
)

// Backend is an alias to graph.Backend.
type Backend = backends.Backend

// Graph is an alias to graph.Graph.
type Graph = graph.Graph

// Node is an alias to graph.Node.
type Node = graph.Node
