/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package nanlogger

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/xla"
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
	e := NewExec(backend, func(values *Node) *Node {
		l.PushScope("scope1")
		v1 := Sqrt(values)
		l.TraceFirstNaN(v1)
		l.PopScope()
		l.PushScope("base")
		v2 := Inverse(values)
		l.TraceFirstNaN(v2, "scope2")
		l.PopScope()
		return Add(v1, v2)
	})
	l.AttachToExec(e)

	// Checks that without any NaN, nothing happens.
	require.NotPanics(t, func() { e.Call([]float32{1.0, 3.0}) })
	assert.Equal(t, 0, numHandlerCalls)

	// Check that NaN is observed, with the correct scope.
	require.NotPanics(t, func() { e.Call([]float32{-1.0, 1.0}) })
	require.Equal(t, 1, numHandlerCalls)
	require.Equal(t, []string{"scope1"}, lastHandledScope)

	// Check now that Inf is observed, with the correct scope.
	// Notice we are also using float32, so it should just work.
	require.NotPanics(t, func() { e.Call([]float32{0.0, 1.0}) })
	require.Equal(t, 2, numHandlerCalls)
	require.Equal(t, []string{"base", "scope2"}, lastHandledScope)

	// Check that the NaN happens before the Inf, and should be the one
	// reported.
	require.NotPanics(t, func() { e.Call([]float32{0.0, -1.0}) })
	require.Equal(t, 3, numHandlerCalls)
	require.Equal(t, []string{"scope1"}, lastHandledScope)
}
