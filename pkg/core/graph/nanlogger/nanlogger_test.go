// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nanlogger

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestNanLogger(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	var numHandlerCalls int
	var lastHandledScope []string
	handler := func(info *Trace) {
		numHandlerCalls++
		lastHandledScope = info.Scope
	}

	// Create a NanLogger and a trivial executor that will trigger NaN and Inf.
	l := New().WithHandler(handler)
	e := MustNewExec(backend, func(values *Node) *Node {
		l.PushScope("scope1")
		v1 := Sqrt(values)
		l.TraceFirstNaN(v1)
		l.PopScope()
		l.PushScope("base")
		v2 := Reciprocal(values)
		l.TraceFirstNaN(v2, "scope2")
		l.PopScope()
		return Add(v1, v2)
	})
	l.AttachToExec(e)

	// Checks that without any NaN, nothing happens.
	require.NotPanics(t, func() { e.MustExec([]float32{1.0, 3.0}) })
	assert.Equal(t, 0, numHandlerCalls)

	// Check that NaN is observed, with the correct scope.
	require.NotPanics(t, func() { e.MustExec([]float32{-1.0, 1.0}) })
	require.Equal(t, 1, numHandlerCalls)
	require.Equal(t, []string{"scope1"}, lastHandledScope)

	// Check now that Inf is observed, with the correct scope.
	// Notice we are also using float32, so it should just work.
	require.NotPanics(t, func() { e.MustExec([]float32{0.0, 1.0}) })
	require.Equal(t, 2, numHandlerCalls)
	require.Equal(t, []string{"base", "scope2"}, lastHandledScope)

	// Check that the NaN happens before the Inf, and should be the one
	// reported.
	require.NotPanics(t, func() { e.MustExec([]float32{0.0, -1.0}) })
	require.Equal(t, 3, numHandlerCalls)
	require.Equal(t, []string{"scope1"}, lastHandledScope)
}
