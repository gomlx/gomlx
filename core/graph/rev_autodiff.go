// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/exceptions"
)

// Gradient creates new nodes for the gradients of the output with respect to each node in gradientNodes.
// The output must be a scalar -- otherwise this would be called Jacobian.
//
// Deprecated: use github.com/gomlx/gomlx/core/autodiff.Gradient instead.
// This stop-gap delegates to the autodiff package if it has been imported (which `ml/model` and
// `ml/train` do automatically via blank import). If `core/autodiff` has not been imported yet,
// calling graph.Gradient will panic with instructions to migrate to autodiff.Gradient or blank-import
// `github.com/gomlx/gomlx/core/autodiff`.
func Gradient(output *Node, gradientNodes ...*Node) []*Node {
	if graphimpl.GradientHook == nil {
		exceptions.Panicf("graph.Gradient requires importing github.com/gomlx/gomlx/core/autodiff (or migrating directly to autodiff.Gradient)")
	}
	return wrapNodes(graphimpl.GradientHook(unwrapNode(output), unwrapNodes(gradientNodes)...))
}
