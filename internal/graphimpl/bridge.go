// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

// Wrapper hooks registered by core/graph.init() so internal/graphbridge can convert
// between public (*graph.Node, *graph.Graph, *graph.Function, *graph.Exec) and
// internal (*NodeImpl, *GraphImpl, *FunctionImpl, *ExecImpl) types without introducing
// import cycles or exporting internal accessors on public types.
var (
	UnwrapNodeHook     func(any) *NodeImpl
	WrapNodeHook       func(*NodeImpl) any
	UnwrapGraphHook    func(any) *GraphImpl
	WrapGraphHook      func(*GraphImpl) any
	UnwrapFunctionHook func(any) *FunctionImpl
	WrapFunctionHook   func(*FunctionImpl) any
	UnwrapExecHook     func(any) *ExecImpl
	WrapExecHook       func(*ExecImpl) any

	// GradientHook is registered by core/autodiff.init() as a temporary stop-gap for
	// the deprecated graph.Gradient forwarding wrapper.
	GradientHook func(output *NodeImpl, gradientNodes ...*NodeImpl) []*NodeImpl
)
