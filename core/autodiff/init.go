// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package autodiff

import (
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

func init() {
	// Register GradientHook so the deprecated graph.Gradient stop-gap forwards here.
	graphimpl.GradientHook = func(output *graphimpl.NodeImpl, gradientNodes ...*graphimpl.NodeImpl) []*graphimpl.NodeImpl {
		return graphbridge.NodeImpls(Gradient(graphbridge.Node(output), graphbridge.Nodes(gradientNodes)...))
	}
}
