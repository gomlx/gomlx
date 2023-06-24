package nanlogger

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestNanLogger(t *testing.T) {
	manager := graphtest.BuildTestManager()

	var numHandlerCalls, numNan, numInf int
	var lastHandledScope []string
	handler := func(nanType float64, info *Trace) {
		numHandlerCalls++
		if math.IsNaN(nanType) {
			numNan++
		}
		if math.IsInf(nanType, 0) {
			numInf++
		}
		lastHandledScope = info.Scope
	}

	// Create a NanLogger and a trivial executor that will trigger NaN and Inf.
	l := New()
	l.SetHandler(handler)
	e := NewExec(manager, func(values *Node) *Node {
		l.PushScope("scope1")
		v1 := Sqrt(values)
		l.Trace(v1)
		l.PopScope()
		l.PushScope("not_used")
		v2 := Inverse(values)
		l.Trace(v2, "scope2")
		l.PopScope()
		return Add(v1, v2)
	})
	l.Attach(e)

	// Checks that without any NaN, nothing happens.
	_, err := e.Call([]float32{1.0, 3.0})
	require.NoError(t, err)
	assert.Equal(t, 0, numHandlerCalls)

	// Check that NaN is observed, with the correct scope.
	_, err = e.Call([]float64{-1.0, 1.0})
	require.NoError(t, err)
	require.Equal(t, 1, numHandlerCalls)
	require.Equal(t, 1, numNan)
	require.Equal(t, 0, numInf)
	require.Equal(t, []string{"scope1"}, lastHandledScope)

	// Check now that Inf is observed, with correct scope.
	// Notice we are also using float32, it should just work.
	_, err = e.Call([]float32{0.0, 1.0})
	require.NoError(t, err)
	require.Equal(t, 2, numHandlerCalls)
	require.Equal(t, 1, numNan)
	require.Equal(t, 1, numInf)
	require.Equal(t, []string{"scope2"}, lastHandledScope)

	// Check that the NaN happens before the Inf, and should be the one
	// reported.
	_, err = e.Call([]float64{0.0, -1.0})
	require.NoError(t, err)
	require.Equal(t, 3, numHandlerCalls)
	require.Equal(t, 2, numNan)
	require.Equal(t, 1, numInf)
	require.Equal(t, []string{"scope1"}, lastHandledScope)

}
