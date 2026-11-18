// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import "github.com/gomlx/gomlx/internal/graphimpl"

func init() {
	graphimpl.UnwrapNodeHook = func(v any) *graphimpl.NodeImpl {
		if v == nil {
			return nil
		}
		return unwrapNode(v.(*Node))
	}
	graphimpl.WrapNodeHook = func(impl *graphimpl.NodeImpl) any {
		return wrapNode(impl)
	}
	graphimpl.UnwrapGraphHook = func(v any) *graphimpl.GraphImpl {
		if v == nil {
			return nil
		}
		return unwrapGraph(v.(*Graph))
	}
	graphimpl.WrapGraphHook = func(impl *graphimpl.GraphImpl) any {
		return wrapGraph(impl)
	}
	graphimpl.UnwrapFunctionHook = func(v any) *graphimpl.FunctionImpl {
		if v == nil {
			return nil
		}
		return unwrapFunction(v.(*Function))
	}
	graphimpl.WrapFunctionHook = func(impl *graphimpl.FunctionImpl) any {
		return wrapFunction(impl)
	}
	graphimpl.UnwrapExecHook = func(v any) *graphimpl.ExecImpl {
		if v == nil {
			return nil
		}
		return unwrapExec(v.(*Exec))
	}
	graphimpl.WrapExecHook = func(impl *graphimpl.ExecImpl) any {
		return wrapExec(impl)
	}
}
